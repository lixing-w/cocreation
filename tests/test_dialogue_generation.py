"""Terminal closure, edit scope, media causality and text-only Qwen contracts."""
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.planner import Planner, build_plan, validate_plan, observation, MODULES, apply_numbering
from scripts.dialogue_generation.generate import run
from scripts.dialogue_generation.conflicts import CONFLICTS
from tools.dialogue_personas import choose_personas, USER_PERSONAS
from scripts.dialogue_generation.asset_variations import (
    POOLS, RELATIVE_POOLS, asset_type, alternate, alternatives, AssetSpec,
)
import random


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "scripts/dialogue_generation/examples/tokyo_vlog.json"


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(EXAMPLE.read_text())

    def test_many_seeds_close_and_cover_all_user_behaviors(self):
        coverage, modules = set(), set()
        for seed in range(200):
            result = build_plan(self.source, seed=seed, reworks=seed % 3)
            self.assertTrue(validate_plan(result))
            self.assertFalse(result["media_generated"])
            coverage.update(result["behavior_coverage"])
            modules.update(e["module"] for e in result["events"])
        self.assertEqual(coverage, set(range(1, 16)))
        self.assertTrue(set(MODULES) <= modules)

    def test_fresh_conversation_prefix_survives_planning_with_or_without_rules(self):
        for rules in [self.source["constraints"], []]:
            source = deepcopy(self.source)
            source["constraints"] = rules
            for seed in range(12):
                plan = build_plan(source, seed=seed, modules=["temporary_rollback"], reworks=0)
                first = plan["turns"][0]["intent"]["description"]
                self.assertTrue(first.startswith("A fresh conversation starts. "))
                self.assertEqual(first.count("A fresh conversation starts."), 1)
                self.assertTrue(all("A fresh conversation starts." not in t["intent"]["description"]
                                    for t in plan["turns"][1:]))

    def test_same_subject_and_return_to_earlier_subject_have_different_descriptions(self):
        planner = Planner(self.source, seed=1)
        planner.emit("assistant", "deliver_asset", "Here is the tote bag.", assets=["bag"])
        self.assertEqual(planner.reference_label("bag"), "same tote bag")
        planner.emit("assistant", "confirm_selection", "Use this traveler.", assets=["woman"])
        self.assertEqual(planner.reference_label("bag"), "previous tote bag")
        result = build_plan(self.source, seed=1, reworks=2, p_compile=0.5)
        turn = next(t for t in result["turns"] if t["intent"]["action"] == "request_variant" and t["intent"]["asset_ids"] == ["bag"])
        self.assertEqual(turn["intent"]["action"], "request_variant")
        self.assertIn("Try the previous tote bag", turn["intent"]["description"])
        self.assertEqual(turn["intent"]["reference_policy"], "clear")

    def test_only_planned_reference_ambiguity_gets_relaxed_policy(self):
        result = build_plan(self.source, seed=1, reworks=2)
        vague = next(t for t in result["turns"] if t["intent"]["action"] == "ambiguous_reference")
        clarify = result["turns"][vague["id"]]
        self.assertEqual(vague["intent"]["reference_policy"], "ambiguous_subject")
        self.assertEqual(clarify["intent"]["reference_policy"], "clear")

    def test_reproducible_without_mutating_input(self):
        original = deepcopy(self.source)
        self.assertEqual(build_plan(self.source, seed=12), build_plan(self.source, seed=12))
        self.assertEqual(self.source, original)
        self.assertNotEqual(build_plan(self.source, seed=12), build_plan(self.source, seed=13))

    def test_one_to_four_shots_and_no_compilation_for_one(self):
        for n in range(1, 5):
            source = deepcopy(self.source)
            shots = source["shots"][:n]
            if n == 4:
                shots.append(dict(id="S4", description="The traveler stops at the side of the street", assets=["woman", "street"]))
            source["shots"] = shots
            used = {a for s in shots for a in s["assets"]}
            source["assets"] = [a for a in source["assets"] if a["id"] in used]
            for rule in source["constraints"]:
                rule["asset_attributes"] = {a: values for a, values in rule.get("asset_attributes", {}).items() if a in used}
            source["target_order"] = [s["id"] for s in reversed(shots)]
            result = build_plan(source, seed=6, reworks=1, p_compile=0)
            self.assertEqual(result["final"]["order"], source["target_order"])
            if n == 1:
                self.assertFalse(any(m["purpose"] == "compilation" for m in result["placeholders"]))

    def test_all_module_subsets_finish_even_cross_only(self):
        for module in MODULES:
            result = build_plan(self.source, modules=[module], seed=3, reworks=2)
            self.assertTrue(validate_plan(result))
            self.assertIn(module, [e["module"] for e in result["events"]])

    def test_rework_happens_after_shot_and_only_updates_confirmed_scope(self):
        local_example = None
        for seed in range(40):
            result = build_plan(self.source, seed=seed, reworks=1, p_compile=1)
            scope = next((t for t in result["turns"] if t["intent"].get("scope") == "shot"), None)
            if scope:
                local_example = result, scope
                break
        self.assertIsNotNone(local_example)
        result, scope = local_example
        shots = set(scope["intent"]["shot_ids"])
        self.assertFalse(any(t['intent']['action']=='synchronize_revision' for t in result['turns']))
        self.assertTrue(any(t['intent']['action']=='deliver_shot' and shots.intersection(t['intent']['shot_ids']) for t in result['turns'][:scope['id']-1]))
        delivery=result['turns'][scope['id']]
        self.assertEqual(delivery['intent']['action'],'deliver_reworked_clips')
        rows={m['placeholder']:m for m in result['placeholders']}
        self.assertEqual({s for token in delivery['intent']['updated_clips'] for s in rows[token]['shot_ids']},shots)

    def test_unaffected_shot_not_regenerated_and_final_compilation_is_current(self):
        result = build_plan(self.source, seed=5, reworks=1, p_compile=1, cross_turn_rules=False)
        cup_shots = [m for m in result["placeholders"] if m["purpose"] == "shot" and m["shot_ids"] == ["S2"]]
        self.assertEqual(len(cup_shots), 1)
        bag_shots = [m for m in result["placeholders"] if m["purpose"] == "shot" and "bag" in m["bindings"]]
        self.assertTrue(any(not m["goal_match"] for m in bag_shots))
        self.assertTrue(any(m["goal_match"] for m in bag_shots))
        table = {m["placeholder"]: m for m in result["placeholders"]}
        final = table[result["final"]["media"]]
        self.assertEqual(final["shot_versions"], {s: result["final"]["shots"][s]["media"] for s in result["target_order"]})

    def test_validator_rejects_future_reference_and_stale_compilation(self):
        result = build_plan(self.source, seed=2)
        broken = deepcopy(result)
        broken["turns"][0]["input_media"] = [result["final"]["media"]]
        with self.assertRaisesRegex(ValueError, "Future"):
            validate_plan(broken)
        table = {m["placeholder"]: m for m in result["placeholders"]}
        table[result["final"]["media"]]["shot_versions"]["S1"] = "[[video_0000]]"
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_plan(result)

    def test_ambiguous_reference_does_not_attach_the_hidden_answer(self):
        result = build_plan(self.source, seed=1, reworks=2, p_compile=1)
        vague = [t for t in result["turns"] if t["intent"]["action"] == "ambiguous_reference"]
        self.assertTrue(vague)
        self.assertTrue(all(not t["input_media"] for t in vague))
        self.assertTrue(all(t["intent"]["target_media"] for t in vague))
        questions = [t for t in result["turns"] if t["intent"]["action"] == "clarify_shot_number"]
        self.assertTrue(questions)
        for turn in questions:
            # The question names two different real shot contents, rather than
            # fabricating ambiguity when both numbering systems mean the same.
            description = turn["intent"]["description"]
            table = {m["placeholder"]: m for m in result["placeholders"]}
            current = table[turn["intent"].get("display_media", [])[0]]["shot_ids"]
            original = []
            for media in result["placeholders"]:
                if media["purpose"] == "shot":
                    shot = media["shot_ids"][0]
                    if shot in current and shot not in original:
                        original.append(shot)
            position = int(re.search(r"shot (\d+)", description).group(1)) - 1
            self.assertNotEqual(current[position], original[position])
            source = {s["id"]: s for s in self.source["shots"]}
            self.assertIn(source[current[position]]["description"], description)
            self.assertIn(source[original[position]]["description"], description)

    def test_bad_observations_fail_before_model_call(self):
        cases = []
        bad = deepcopy(self.source); bad["shots"] *= 2; cases.append(bad)
        bad = deepcopy(self.source); bad["assets"][0]["frames"] = [float("nan")]; cases.append(bad)
        bad = deepcopy(self.source); bad["shots"][0]["assets"].append("missing"); cases.append(bad)
        bad = deepcopy(self.source); bad["assets"][0]["id"] = "bag"; cases.append(bad)
        bad = deepcopy(self.source); bad["target_order"] = ["S1", "S1", "S2"]; cases.append(bad)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                observation(value)

    def test_confirmations_and_historical_requests_are_not_attachments(self):
        result = build_plan(self.source, seed=296, p_compile=1)
        for turn in result["turns"]:
            if not turn["output_media"]:
                self.assertEqual(turn["input_media"], turn["intent"].get("display_media", []))
            if turn["intent"]["action"] in ("adopt", "rollback", "confirm_selection", "correct_reference", "confirm_reference", "request_shot"):
                self.assertFalse(turn["intent"].get("display_media", []))
            if turn["intent"]["action"] == "clarify_reference":
                self.assertTrue(turn["intent"].get("display_media", []))
                self.assertNotRegex(turn["intent"]["description"], "keep everything else|darken|will do")
        deliveries = [t for t in result["turns"] if t["intent"]["action"] == "deliver_shot"]
        self.assertTrue(all(t["input_media"] and not t["intent"].get("display_media", []) for t in deliveries))

    def test_implicit_acceptance_keeps_state_without_dummy_messages(self):
        selections = []
        for seed in range(100):
            result = build_plan(self.source, seed=seed, modules=["clarify"], reworks=0, extra_behaviors=False)
            selections.extend(result["selections"])
            for selection in result["selections"]:
                if not selection["explicit"]:
                    next_turn = result["turns"][selection["after_message"]:][:1]
                    self.assertFalse(next_turn and next_turn[0]["intent"]["action"] == "adopt")
                    self.assertEqual(result["final"]["assets"][selection["asset_id"]], selection["target_media"])
        rate = sum(not s["explicit"] for s in selections) / len(selections)
        self.assertGreater(rate, 0.4)
        self.assertLess(rate, 0.6)

    def test_relative_variations_respect_asset_type_and_close(self):
        variations = set()
        for seed in range(60):
            result = build_plan(self.source, seed=seed, modules=["relative_preview"], reworks=1, extra_behaviors=False)
            for m in result["placeholders"]:
                if "variation" not in m:
                    continue
                kind = m["variation"]["type"]
                variations.add(kind)
                if kind == "size":
                    row = next(a for a in self.source["assets"] if a["id"] == m["asset_ids"][0])
                    self.assertTrue(row["kind"] == "object" or row["id"] in ("bag", "cup"))
                if m["variation"]["phase"] == "preview":
                    self.assertTrue(m["edit_parent"])
                    self.assertLessEqual(m["edit_depth"], 2)
                self.assertIn(kind, POOLS[m["asset_type"]])
        expected = {key for row in observation(self.source)["assets"]
                    for key in alternatives(row, relative=True)}
        self.assertEqual(variations, expected)

    def test_alternatives_and_relative_edits_never_use_lighting_for_people_or_objects(self):
        for seed in range(80):
            result = build_plan(self.source, seed=seed, modules=["relative_preview", "temporary_rollback"], reworks=2)
            for m in result["placeholders"]:
                if m.get("asset_type") in ("person", "object"):
                    self.assertNotRegex(m["description"], r"lighting|exposure|illumination|shot size")
            for turn in result["turns"]:
                if set(turn["intent"]["asset_types"].values()) <= {"person", "object"}:
                    self.assertNotRegex(turn["intent"]["description"], r"change only the lighting|overall exposure|a brighter frame")

    def test_person_changes_a_known_field_and_requires_explicit_kind(self):
        woman = observation(self.source)["assets"][0]
        varied = alternate(woman, AssetSpec(woman), random.Random(1))
        self.assertEqual(sum(varied.attributes[k] != v for k, v in woman["attributes"].items()), 1)
        self.assertNotIn("lighting", varied)
        with self.assertRaisesRegex(ValueError, "kind"):
            asset_type(dict(id="unknown", kind="identity", name="X", description="X"))
        self.assertEqual(asset_type(dict(id="x", kind="object", description="X")), "object")

    def test_shot_requests_and_conflict_resolution_have_natural_drafts(self):
        result = build_plan(self.source, seed=62, p_compile=1)
        requests = [t for t in result["turns"] if t["intent"]["action"] == "request_shot"]
        self.assertTrue(requests)
        for turn in requests:
            self.assertNotRegex(turn["intent"]["description"], r"assets|reference assignments|explicit requirements")
            self.assertIn("video", turn["intent"]["description"])
        conflict = next(t for t in result["turns"] if t["intent"]["action"] == "clarify_conflict")
        spec = CONFLICTS[conflict["intent"]["conflict_type"]]
        self.assertRegex(conflict["intent"]["description"], spec["reason"])
        reply = next(t for t in result["turns"] if t["intent"]["action"] == "resolve_conflict")
        self.assertEqual(reply["intent"]["conflict_type"], conflict["intent"]["conflict_type"])
        self.assertEqual(reply["intent"]["shot_description"], self.source["shots"][0]["description"])

    def test_numbering_is_a_concrete_operation_and_executes_after_clarification(self):
        result = build_plan(self.source, seed=296, p_compile=1)
        start = next(t for t in result["turns"] if t["intent"]["action"] == "ambiguous_shot_number")
        self.assertIn(start["intent"]["numbering_operation"]["type"], ("swap", "move_after"))
        question, resolution, delivery = result["turns"][start["id"]:start["id"] + 3]
        self.assertEqual(question["intent"]["action"], "clarify_shot_number")
        self.assertEqual(resolution["intent"]["numbering_basis"], "current")
        media = {m["placeholder"]: m for m in result["placeholders"]}
        before = media[question["intent"].get("display_media", [])[0]]["shot_ids"]
        self.assertEqual(media[delivery["output_media"][0]]["shot_ids"],
                         apply_numbering(before, start["intent"]["numbering_operation"]))

    def test_correction_records_shown_subject_separately_from_target_object(self):
        result = build_plan(self.source, seed=62, p_compile=1)
        t = next(t for t in result["turns"] if t["intent"]["action"] == "correct_reference")
        correction = t["intent"]["reference_correction"]
        self.assertIn(correction["shown_name"], {a["name"] for a in self.source["assets"]})
        self.assertEqual(correction["target_name"], "tote bag")
        self.assertNotEqual(correction["shown_media"], correction["target_media"])

    def test_four_shot_reordering_asks_about_a_genuinely_ambiguous_position(self):
        source = json.loads((EXAMPLE.parent / "alpine_kayak_vlog.json").read_text())
        # Previously the move named two unambiguous positions and crashed
        # while looking for a position to clarify.
        result = build_plan(source, seed=3, reworks=1, p_compile=1)
        self.assertTrue(validate_plan(result))
        self.assertEqual(result["final"]["order"], source["target_order"])
        start = next(t for t in result["turns"] if t["intent"]["action"] == "ambiguous_shot_number")
        question = result["turns"][start["id"]]
        table = {m["placeholder"]: m for m in result["placeholders"]}
        current = table[question["intent"].get("display_media", [])[0]]["shot_ids"]
        original = []
        for media in result["placeholders"]:
            if media["purpose"] == "shot":
                shot = media["shot_ids"][0]
                if shot not in original:
                    original.append(shot)
        position = int(re.search(r"shot (\d+)", question["intent"]["description"]).group(1)) - 1
        named = [p - 1 for p in start["intent"]["numbering_operation"]["positions"]]
        self.assertIn(position, named)
        self.assertNotEqual(current[position], original[position])

    def test_swap_and_move_after_and_validator_rejects_unconfirmed_edits(self):
        self.assertEqual(apply_numbering(["A", "B", "C"], dict(type="swap", positions=[1, 2])), ["B", "A", "C"])
        self.assertEqual(apply_numbering(["A", "B", "C", "D"], dict(type="move_after", positions=[4, 1])), ["A", "D", "B", "C"])
        result = build_plan(self.source, seed=296, p_compile=1)
        resolution = next(t for t in result["turns"] if t["intent"]["action"] == "resolve_shot_number")
        resolution["intent"]["action"] = "confirm_other_thing"
        with self.assertRaisesRegex(ValueError, "before numbering clarification"):
            validate_plan(result)

    def test_tokyo_rollback_surfaces_its_sampled_framing_change(self):
        from scripts.dialogue_generation.asset_variations import empty_attributes
        source = deepcopy(self.source)
        source['constraints'] = []
        for asset in source['assets']:
            distance = asset['attributes'].get('camera_distance')
            asset['attributes'] = empty_attributes(asset['kind'])
            if asset['id'] == 'street':
                asset['attributes']['camera_distance'] = distance
        from scripts.dialogue_generation.planner import Planner
        planner = Planner(source, seed=197, reworks=0, p_compile=1, cross_turn_rules=False)
        planner.cross('street', 'woman', 'temporary_rollback')
        result = planner.run()
        turns = [turn for turn in result["turns"]
                 if turn["intent"]["asset_ids"] == ["street"] and "required_attribute" in turn["intent"]]
        self.assertEqual([turn["intent"]["action"] for turn in turns],
                         ["request_asset", "request_variant", "deliver_variant", "rollback"])
        values = [turn['intent']['required_attribute']['value'] for turn in turns]
        self.assertEqual(values[0], values[3])
        self.assertEqual(values[1], values[2])
        self.assertNotEqual(values[0], values[1])
        original, request, delivery, rollback = turns
        table = {item["placeholder"]: item for item in result["placeholders"]}
        base = result["turns"][original["id"]]["output_media"][0]
        variant = table[delivery["output_media"][0]]
        self.assertEqual(variant["edit_parent"], base)
        self.assertEqual(variant["attribute_changes"],
                         {"camera_distance": {"before": values[0], "after": values[1]}})
        self.assertIn(base, rollback["intent"]["context_media"])
        for unchanged in ("clear skies", "somewhat dim", "shops line both sides"):
            self.assertNotIn(unchanged, request["intent"]["description"].lower())


    def test_validator_rejects_lighting_variation_on_a_person(self):
        result = build_plan(self.source, seed=0, modules=["relative_preview"])
        person = next(m for m in result["placeholders"] if m.get("asset_type") == "person" and "variation" in m)
        person["variation"]["type"] = "exposure"
        with self.assertRaisesRegex(ValueError, "asset type"):
            validate_plan(result)


class RealizationTests(unittest.TestCase):










    def test_personas_are_seeded_once_and_cover_five_styles(self):
        self.assertEqual(len(USER_PERSONAS), 5)
        self.assertEqual(choose_personas(8), choose_personas(8))
        self.assertEqual({choose_personas(s)["user"]["id"] for s in range(100)}, set(USER_PERSONAS))
        self.assertEqual(len({choose_personas(s)["assistant"]["id"] for s in range(100)}), 1)

















    def test_outputs_include_every_turn_intent_and_placeholder_catalog(self):
        class Writer:
            backend, model = "test", "stub-for-tests-only"
            def __init__(self): self.count = 0
            def __call__(self, turn, media, previous, *, seed):
                self_history = len(previous)
                assert self_history == self.count
                self.count += 1
                return "This is a test line."
        with tempfile.TemporaryDirectory() as tmp:
            writer = Writer()
            result = run(EXAMPLE, Path(tmp) / "result", realizer=writer, seed=2, progress=False)
            output = Path(tmp) / "result"
            self.assertEqual(writer.count, len(result["turns"]))
            self.assertEqual(result["realization"]["status"], "complete")
            self.assertTrue(all(t["content"] and t["intent"]["description"] for t in result["turns"]))
            self.assertEqual(len(json.loads((output / "placeholders.json").read_text())), len(result["placeholders"]))
            self.assertIn("Media placeholders", (output / "dialogue.md").read_text())
            self.assertFalse(any(p.suffix in (".png", ".mp4") for p in output.rglob("*")))
            for turn in result["turns"]:
                self.assertEqual("References: " in turn["content"], bool(turn["intent"].get("display_media", [])))
                for p in turn["input_media"]:
                    self.assertEqual(p in turn["content"], p in turn["intent"].get("display_media", []))

    def test_failed_model_call_keeps_plan_and_explicit_failed_checkpoint(self):
        def broken(*args, **kwargs):
            raise RuntimeError("intentional test outage")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed"
            with self.assertRaisesRegex(RuntimeError, "outage"):
                run(EXAMPLE, output, realizer=broken, progress=False)
            self.assertTrue((output / "plan.json").is_file())
            result = json.loads((output / "dialogue.json").read_text())
            self.assertEqual(result["realization"]["status"], "failed")
            self.assertEqual(result["realization"]["completed_turns"], 0)
            self.assertFalse((output / "dialogue.md").exists())

    def test_resume_preserves_full_prefix_and_checks_planning_options(self):
        previous_lengths = []
        def interrupted(turn, media, previous, *, seed):
            if len(previous) == 2:
                raise RuntimeError("test interruption")
            return "A line from earlier."
        def continued(turn, media, previous, *, seed):
            previous_lengths.append(len(previous))
            self.assertEqual(previous[0]["content"], "A line from earlier.")
            return "A test line after resuming."
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            with self.assertRaises(RuntimeError):
                run(EXAMPLE, output, realizer=interrupted, seed=2, progress=False)
            with self.assertRaisesRegex(ValueError, "same observation"):
                run(EXAMPLE, output, realizer=continued, seed=3, resume=True, progress=False)
            result = run(EXAMPLE, output, realizer=continued, seed=2, resume=True, progress=False)
            self.assertEqual(previous_lengths, list(range(2, len(result["turns"]))))
            self.assertEqual(result["realization"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
