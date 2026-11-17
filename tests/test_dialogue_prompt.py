"""Actual prompt boundaries and independent writer/verifier transport."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.dialogue_generation.planner import build_plan
from tools.dialogue_personas import choose_personas
from tools.dialogue_realizer import QwenRealizer, SYSTEM
from tools.dialogue_verifier import CRITERIA, SYSTEM as VERIFIER_SYSTEM, REPORT_SCHEMA

EXAMPLES = Path(__file__).resolve().parents[1]/"scripts/dialogue_generation/examples"


def approved():
    return dict(dict.fromkeys(CRITERIA, 2), critique="")


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return b'{"response":"ok","message":{"content":"ok"},"done_reason":"stop"}'


class PromptTests(unittest.TestCase):
    def model(self, **kwargs):
        kwargs.setdefault("backend", "ollama")
        model = QwenRealizer(**kwargs)
        model.verifier = Mock(return_value=approved())
        return model

    def test_persona_examples_remain_verbatim_and_intention_is_separate_from_tone(self):
        personas = choose_personas(1, "casual")
        model = self.model(personas=personas)
        turn = dict(id=7, role="user", intent=dict(action="request_asset",
                    description="Request the assistant to make a reference image of a traveler with short black hair."))
        with patch.object(model, "_request", side_effect=["{}", "Can we get a still of her with short black hair?"]) as writer:
            model(turn, {}, [dict(role="assistant", content="What should she look like?")])
        for invocation in writer.call_args_list:
            prompt = invocation.args[0]
            self.assertIn(personas["user"]["style"], prompt)
            self.assertIn(personas["user"]["example"], prompt)
            positions = [prompt.index(part) for part in ["Speak as the User", "Tone examples:",
                         "Previous messages", "Now, write", "Reply intention:",
                         "Content requirements"]]
            self.assertEqual(positions, sorted(positions))
            self.assertTrue(prompt.endswith("Return only the user's spoken line."))

    def test_formal_shot_category_is_shared_by_writer_and_verifier(self):
        model = self.model()
        turn = dict(id=1, role='user', intent=dict(action='request_shot',
            description='Make the planned shot.', shot_description='Slow push-in on the cup.',
            media_rule_context=['formal shots: black-and-white rendering']))
        with patch.object(model, '_request', return_value='Make a formal black-and-white shot with a slow push-in on the cup.') as writer:
            model(turn, {})
        self.assertIn('Required media category: formal shots.', writer.call_args.args[0])
        brief = model.verifier.call_args.args[1]
        self.assertEqual(brief['intent_facts']['requested_media_role'], 'shot')
        from tools.dialogue_verifier import build_prompt
        self.assertIn('Required media category: formal shots.', build_prompt('candidate', brief))

    def test_writer_system_and_prompt_reach_wire_without_copying_draft_into_prefill(self):
        prompt = "Speak as the User with this tone:\nCasual.\nDraft content requirements: UNIQUE_FACT."
        with tempfile.TemporaryDirectory() as tmp:
            model = QwenRealizer(backend="ollama", model="qwen3-vl-30b-thinking", output_dir=tmp)
            with patch("urllib.request.urlopen", return_value=Response()) as request:
                self.assertEqual(model._request(prompt, 123), "ok")
            body = json.loads(request.call_args.args[0].data)
            self.assertTrue(body["raw"])
            self.assertIn("<|im_start|>system\n"+SYSTEM+"<|im_end|>\n", body["prompt"])
            self.assertIn("<|im_start|>user\n"+prompt+"<|im_end|>\n", body["prompt"])
            self.assertTrue(body["prompt"].endswith("</think>\n\n"))
            self.assertIn("dialogue act has been decided", body["prompt"])
            self.assertEqual(body["prompt"].count("UNIQUE_FACT"), 1)
            self.assertEqual(json.loads((Path(tmp)/"requests.jsonl").read_text()), body)

    def test_action_control_is_system_only_and_reaches_ollama(self):
        from tools.dialogue_realizer import ACT_GUIDANCE
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model(output_dir=tmp)
            turn = dict(id=1, role="user", intent=dict(action="request_shot", description="A medium tracking shot.", shot_description="Traveler walks beside the lake."))
            with patch("urllib.request.urlopen", return_value=Response()) as request:
                model(turn, {})
            body = json.loads(request.call_args.args[0].data)
            system, user = body["prompt"].split("<|im_start|>user\n", 1)
            self.assertIn(ACT_GUIDANCE["request_shot"], system)
            self.assertNotIn(ACT_GUIDANCE["request_shot"], user)
            self.assertNotIn("Action: request_shot", user)

    def test_verifier_uses_same_model_distinct_system_and_structured_report_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = QwenRealizer(backend="ollama", model="qwen3-vl:30b", output_dir=tmp)
            with patch("urllib.request.urlopen", return_value=Response()) as request:
                self.assertEqual(model._verify_request("Evaluate this candidate.", 124), ("ok", "stop"))
            body = json.loads(request.call_args.args[0].data)
            self.assertEqual(body["model"], model.model)
            self.assertEqual(body["messages"][0], dict(role="system", content=VERIFIER_SYSTEM))
            self.assertTrue(body["think"])
            self.assertNotIn(SYSTEM, json.dumps(body["messages"]))
            self.assertEqual(body["format"], REPORT_SCHEMA)
            self.assertEqual(body["options"]["temperature"], .1)
            self.assertEqual(json.loads((Path(tmp)/"verifier/requests.jsonl").read_text()), body)

    def test_local_backend_keeps_systems_separate_on_one_shared_adapter(self):
        model = QwenRealizer(backend="ollama", model="qwen3-vl-30b-thinking")
        model.local = Mock(side_effect=[{"text":"ok"}, {"text":"Review evidence.\n</think>"+json.dumps(approved())}])
        model._request("Writer task", 1)
        model._verify_request("Judge task", 2)
        calls = model.local.call_args_list
        self.assertEqual([c.kwargs["system_prompt"] for c in calls], [SYSTEM, VERIFIER_SYSTEM])
        self.assertIn("dialogue act has been decided", calls[0].kwargs["assistant_prefill"])
        self.assertEqual(calls[1].kwargs["assistant_prefill"], "<think>\n")

    def test_verifier_never_adds_a_separate_report_finalization_call(self):
        for name in ("qwen3-vl-30b-instruct", "qwen3-vl-30b-thinking"):
            model = QwenRealizer(backend="ollama", model=name)
            with patch.object(model, "_generate", return_value=("", "length")) as request:
                self.assertEqual(model._verify_request("Evaluation data", 12), ("", "length"))
            request.assert_called_once()

    def test_instruct_has_no_thinking_prefill_and_uses_one_shared_model(self):
        with patch("cocreation.Qwen") as factory:
            factory.return_value.side_effect = [{"text": "User line."}, {"text": json.dumps(approved())}]
            model = QwenRealizer(backend="local")
        model._request("Write", 1)
        model._verify_request("Judge", 2)
        self.assertEqual(factory.call_args.kwargs["model_id"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(len(factory.return_value.call_args_list), 2)
        for call in factory.return_value.call_args_list:
            self.assertIsNone(call.kwargs["assistant_prefill"])

    def test_semantic_history_is_three_messages_and_older_style_is_short_excerpts_only(self):
        model = self.model()
        turn = dict(id=9, role="user", intent=dict(action="correct_reference", description="Clarify which earlier version was meant."))
        previous = [dict(role="user", content="Yeah, let's use this one. SECRET_OLDER_FACT"),
                    dict(role="assistant", content="OLD_ASSISTANT_FACT"),
                    dict(role="user", content="Yeah, let's try another version."),
                    dict(role="assistant", content="Here's the image.\n[[image_0001]]"),
                    dict(role="assistant", content="This is the requested version.")]
        with patch.object(model, "_request", return_value="Use this one.") as writer:
            model(turn, {}, previous)
        prompt = writer.call_args.args[0]
        self.assertNotIn("Recent same-speaker opening excerpts", prompt)
        self.assertNotIn("Yeah, let's use this one.", prompt)
        self.assertNotIn("SECRET_OLDER_FACT", prompt)
        self.assertNotIn("OLD_ASSISTANT_FACT", prompt)
        self.assertNotIn("image_0001", prompt)
        context = prompt.split("Previous messages", 1)[1].split("Now, write", 1)[0]
        self.assertEqual(context.count("Assistant:"), 2)
        self.assertEqual(context.count("User:"), 1)

    def test_decision_and_delivery_notes_do_not_seed_stock_spoken_lines(self):
        model = self.model()
        for role, action, description in [("user", "adopt", "Let's use this version."),
                                          ("assistant", "confirm_selection", "Acknowledge the selected version."),
                                          ("assistant", "deliver_asset", "Present the reference image.")]:
            turn = dict(id=1, role=role, intent=dict(action=action, description=description))
            with patch.object(model, "_request", return_value="A line.") as writer:
                model(turn, {})
            notes = writer.call_args.args[0].split("Content requirements", 1)[1]
            self.assertNotIn("Got it, we'll use", notes)
            self.assertNotIn("Here's the reference", notes)
            self.assertNotIn("Let's use this version.", notes)

    def test_numbered_selection_is_derived_from_media_bindings(self):
        model = self.model()
        turn = dict(id=3, role="user", intent=dict(action="adopt", description="Choose option two.",
                    context_media=["b"], decision="temporary"))
        previous = [dict(role="assistant", content="Here are both options.", output_media=["a", "b"])]
        with patch.object(model, "_request", return_value="Use the second one for now.") as writer:
            model(turn, {}, previous)
        self.assertIn("Selected option number: 2", writer.call_args.args[0])
        self.assertIn("Decision status: temporary", writer.call_args.args[0])

    def test_verifier_gets_separate_content_notes_and_actual_delivered_subjects(self):
        model = self.model(asset_names={"bag":"tote bag"})
        turn = dict(id=12, role="assistant", output_media=["bag_image"], intent=dict(
                    action="deliver_variant", description="Here's the test version with material: wood.",
                    asset_ids=["bag"], required_attribute=dict(label="main material", value="wood")))
        media = {"bag_image":dict(kind="image", description="Standalone wooden tote bag.", asset_ids=["bag"])}
        with patch.object(model, "_request", return_value="Here's the wooden tote."):
            model(turn, media)
        brief = model.verifier.call_args.args[1]
        self.assertEqual(brief["action"], "deliver_variant")
        self.assertEqual(brief["delivered_media"][0]["subjects"], ["tote bag"])
        self.assertNotIn("description", brief["delivered_media"][0])
        self.assertNotIn("Here's", " ".join(brief["draft_content_requirements"]))
        self.assertNotIn("Speak as", json.dumps(brief))

    def test_deliberately_vague_turn_never_exposes_recent_subject_identity(self):
        model = self.model(asset_names={"cup": "PRIVATE_TARGET"})
        turn = dict(id=53, role="user", intent=dict(action="ambiguous_reference", description="Change that thing in an earlier image.", asset_ids=["cup"]))
        previous = [dict(role="assistant", content="PRIVATE_RECENT_SUBJECT")]
        with patch.object(model, "_request", return_value="Can we change that thing from earlier?") as writer:
            model(turn, {}, previous)
        self.assertNotIn("PRIVATE_TARGET", writer.call_args.args[0])
        self.assertNotIn("PRIVATE_RECENT_SUBJECT", writer.call_args.args[0])

    def test_delivery_writer_receives_no_visual_caption_or_previous_request(self):
        model = self.model(asset_names={"bag": "PRIVATE_SUBJECT_NAME"})
        turn = dict(id=2, role="assistant", output_media=["image"], intent=dict(action="deliver_asset", description="Present the image.", asset_ids=["bag"]))
        media = {"image": dict(kind="image", description="PRIVATE_VISUAL_DETAILS", asset_ids=["bag"])}
        previous = [dict(role="user", content="PRIVATE_OLD_REQUEST")]
        with patch.object(model, "_request", return_value="Here's the reference image.") as writer:
            model(turn, media, previous)
        for withheld in ("PRIVATE_VISUAL_DETAILS", "PRIVATE_OLD_REQUEST", "PRIVATE_SUBJECT_NAME"):
            self.assertNotIn(withheld, writer.call_args.args[0])
        self.assertNotIn("PRIVATE_VISUAL_DETAILS", json.dumps(model.verifier.call_args.args[1]))

    def test_new_shot_writer_receives_only_current_shot_and_reference_names(self):
        model = self.model(asset_names={"bag": "tote bag"})
        turn = dict(id=5, role="user", intent=dict(action="request_shot", description="Make the shot.", shot_description="Medium tracking shot of the traveler walking.", asset_ids=["bag"], context_media=["bag"]))
        previous = [dict(role="user", content="PRIVATE_PREVIOUS_CAMERA")]
        media = {"bag": dict(kind="image", description="PRIVATE_OTHER_APPEARANCE", asset_ids=["bag"])}
        with patch.object(model, "_request", return_value="Can we get a medium tracking shot?") as writer:
            model(turn, media, previous)
        prompt = writer.call_args.args[0]
        self.assertIn("Medium tracking shot of the traveler walking", prompt)
        self.assertIn("tote bag", prompt)
        self.assertNotIn("PRIVATE_PREVIOUS_CAMERA", prompt)
        self.assertNotIn("PRIVATE_OTHER_APPEARANCE", prompt)

    def test_compilation_requirements_name_only_the_actual_existing_clips_in_order(self):
        model = self.model()
        turn = dict(id=57, role="user", intent=dict(action="request_compilation", description="Add the new clip to the preview.", context_media=["walk", "paddle"]))
        media = {"walk": dict(kind="video", description="Traveler walks to the water. References used: red case"),
                 "paddle": dict(kind="video", description="Traveler paddles away. References used: yellow kayak")}
        with patch.object(model, "_request", return_value="First the walk, then paddling.") as writer:
            model(turn, media)
        requirements = model.verifier.call_args.args[1]["draft_content_requirements"]
        self.assertEqual(requirements, ["Assemble the existing video clips in this order", "First: Traveler walks to the water", "Second: Traveler paddles away"])
        self.assertNotIn("red case", writer.call_args.args[0])

    def test_human_review_feedback_reaches_writer_but_does_not_bias_judge(self):
        model = self.model()
        turn = dict(id=1, role="user", intent=dict(action="request_shot", description="Ask for a tracking shot."))
        with patch.object(model, "_request", return_value="Can we get a tracking shot?") as writer:
            model(turn, {}, review_feedback="REVIEW_NOTE: rewrite the whole line in English.")
        self.assertIn("REVIEW_NOTE", model._writer_control)
        self.assertNotIn("REVIEW_NOTE", json.dumps(model.verifier.call_args.args[1]))

    def test_required_version_detail_is_content_for_both_writer_and_judge(self):
        model = self.model(asset_names={"paddle": "paddle"})
        turn = dict(id=31, role="user", intent=dict(action="rollback", description="Go back to the earlier matte paddle.",
                    asset_ids=["paddle"], required_attribute=dict(field="surface_finish", label="surface finish", value="matte")))
        with patch.object(model, "_request", return_value="Back to the matte paddle.") as writer:
            model(turn, {})
        self.assertIn("Required surface finish of this version: matte.", writer.call_args.args[0])
        self.assertIn("Required surface finish of this version: matte.", model.verifier.call_args.args[1]["draft_content_requirements"])

    def test_rollback_judge_gets_subject_recent_test_and_no_future_alternative(self):
        model = self.model(asset_names={"bag": "tote bag"})
        turn = dict(id=15, role="user", intent=dict(action="rollback", asset_ids=["bag"],
                    description="Return to the canvas tote bag.", context_media=["earlier"],
                    required_attribute=dict(label="main material", value="canvas", alternative="PRIVATE_ALTERNATIVE")))
        media = {"earlier":dict(kind="image", purpose="asset", asset_ids=["bag"])}
        previous = [dict(role="user", content="Let's test the wood version."),
                    dict(role="assistant", content="The test clip is ready.")]
        with patch.object(model, "_request", return_value="Back to the canvas tote bag.") as writer:
            model(turn, media, previous)
        brief = model.verifier.call_args.args[1]
        self.assertEqual(brief["subjects"], ["tote bag"])
        self.assertEqual(brief["previous_messages"], previous)
        self.assertEqual(brief["intent_facts"]["required_attribute"]["subject"], "tote bag")
        self.assertNotIn("PRIVATE_ALTERNATIVE", json.dumps(brief))
        self.assertIn("earlier reference of the tote bag", writer.call_args.args[0])

    def test_original_description_and_subject_transition_reach_judge(self):
        model = self.model(asset_names={"bag": "tote bag", "woman": "lead traveler"})
        turn = dict(id=11, role="user", intent=dict(action="request_variant", asset_ids=["bag"],
                    reference_policy="clear", description="Try the previous tote bag with wood."))
        previous = [dict(role="assistant", content="We'll use the lead traveler.", intent=dict(asset_ids=["woman"]))]
        with patch.object(model, "_request", return_value="Try that tote bag with wood."):
            model(turn, {}, previous)
        brief = model.verifier.call_args.args[1]
        self.assertEqual(brief["python_description"], turn["intent"]["description"])
        self.assertIn("preceding turn concerns lead traveler", brief["reference_transition"])
        self.assertIn("'that tote bag'", brief["reference_transition"])
        turn["intent"]["reference_policy"] = "ambiguous_subject"
        with patch.object(model, "_request", return_value="Try that with wood."):
            model(turn, {}, previous)
        self.assertNotIn("reference_transition", model.verifier.call_args.args[1])

    def test_continuing_reference_context_does_not_invent_other_properties(self):
        model = self.model(asset_names={"bag": "tote bag"})
        turn = dict(id=11, role="user", intent=dict(action="request_variant", asset_ids=["bag"], reference_policy="clear",
                    description="Try the same tote bag with a matte finish.",
                    required_attribute=dict(field="surface_finish", label="surface finish", value="matte")))
        previous = [dict(role="assistant", content="Here's the tote bag.", intent=dict(asset_ids=["bag"]))]
        with patch.object(model, "_request", return_value="Can we make that matte?"):
            model(turn, {}, previous)
        context = model.verifier.call_args.args[1]["reference_transition"]
        self.assertIn("continuation, not a topic switch", context)
        self.assertNotIn("wood", context)

    def test_single_reference_adoption_records_actual_choice_count(self):
        model = self.model(asset_names={"street": "Tokyo street"})
        turn = dict(id=19, role="user", intent=dict(action="adopt", asset_ids=["street"],
                    description="Accept the street reference.", context_media=["street_image"]))
        previous = [dict(role="assistant", content="Here's the reference.", output_media=["street_image"])]
        with patch.object(model, "_request", return_value="Use this one.") as writer:
            model(turn, {}, previous)
        self.assertIn("accept the single reference of the Tokyo street just shown", writer.call_args.args[0])
        brief = model.verifier.call_args.args[1]
        self.assertEqual(brief["offered_choice_count"], 1)
        self.assertEqual(brief["selected_choice_number"], 1)

    def test_rollback_contrast_uses_only_already_delivered_variant(self):
        model = self.model(asset_names={"bag": "tote bag"})
        turn = dict(id=15, role="user", intent=dict(action="rollback", asset_ids=["bag"],
                    description="Return to the canvas tote bag.", context_media=["earlier"],
                    required_attribute=dict(field="material", label="main material", value="canvas")))
        media = {key:dict(kind="image", asset_ids=["bag"], attributes=dict(material=value))
                 for key, value in [("earlier", "canvas"), ("newer", "wood"), ("future", "PRIVATE_FUTURE") ]}
        previous = [dict(role="assistant", content="Here's the variant.", output_media=["newer"])]
        with patch.object(model, "_request", return_value="Back to the canvas tote."):
            model(turn, media, previous)
        brief = model.verifier.call_args.args[1]
        self.assertEqual(brief["previous_variant"], dict(subject="tote bag", label="main material", value="wood"))
        self.assertNotIn("PRIVATE_FUTURE", json.dumps(brief))

    def test_wide_framing_hint_survives_neutral_assistant_content_notes(self):
        model = self.model()
        turn = dict(id=28, role="assistant", intent=dict(action="deliver_variant",
                    description="Present the variant.", required_attribute=dict(label="shot size", value="long shot")))
        with patch.object(model, "_request", return_value="Here's the wide shot.") as writer:
            model(turn, {})
        self.assertIn("Required shot size of this version: wide shot.", writer.call_args.args[0])

    def test_first_creation_gets_actual_subject_discussion_without_unrelated_history(self):
        model = self.model(asset_names={"paddle":"paddle"})
        turn = dict(id=7, role="user", intent=dict(action="request_asset", first_creation=True,
                    asset_ids=["paddle"], description="Request a still of a wooden paddle."))
        previous = [dict(role="user", content="Prepare the kayak.", intent=dict(asset_ids=["kayak"]))]
        with patch.object(model, "_request", return_value="Can we get a still of a wooden paddle?") as writer:
            model(turn, {}, previous)
        self.assertEqual(model.verifier.call_args.args[1]["prior_subject_discussion"], [])
        self.assertNotIn("has not been discussed", writer.call_args.args[0])
        previous.append(dict(role="assistant", content="What should the paddle look like?", intent=dict(asset_ids=["paddle"])))
        with patch.object(model, "_request", return_value="Can we get a still of a wooden paddle?"):
            model(turn, {}, previous)
        self.assertEqual(model.verifier.call_args.args[1]["prior_subject_discussion"],
                         [dict(role="assistant", content="What should the paddle look like?")])

    def test_clarification_gets_all_re_shown_candidates_without_hidden_answer(self):
        model = self.model(asset_names={"case":"waterproof case", "kayak":"kayak"})
        turn = dict(id=58, role="assistant", input_media=["one", "two"], intent=dict(
                    display_media=["one", "two"], action="clarify_reworks", asset_ids=["case"],
                    description="Ask whether the user means the kayak or waterproof case."))
        media = {"one":dict(kind="image", description="Existing kayak.", asset_ids=["kayak"]),
                 "two":dict(kind="image", description="Existing waterproof case.", asset_ids=["case"])}
        with patch.object(model, "_request", return_value="Did you mean the kayak or the case?"):
            model(turn, media)
        brief = model.verifier.call_args.args[1]
        self.assertEqual([row["subjects"] for row in brief["shown_media"]], [["kayak"], ["waterproof case"]])
        self.assertNotIn("target_media", json.dumps(brief))

    def test_ambiguous_batch_brief_excludes_oracle_targets_and_future_facts(self):
        source = json.loads((EXAMPLES/"alpine_kayak_vlog.json").read_text())
        for seed in range(100):
            plan = build_plan(source, seed=seed, reworks=2)
            if any(t["intent"]["action"] == "request_reworks" for t in plan["turns"]):
                break
        turn = next(t for t in plan["turns"] if t["intent"]["action"] == "request_reworks")
        model = self.model(asset_names={a["id"]:"SECRET_"+a["id"] for a in source["assets"]})
        media = {m["placeholder"]:m for m in plan["placeholders"]}
        with patch.object(model, "_request", return_value="Please make those edits.") as writer:
            model(turn, media)
        self.assertNotIn("SECRET_", writer.call_args.args[0])
        self.assertNotIn("target_media", writer.call_args.args[0])

    def test_context_overflow_is_explicit_instead_of_truncating_latest_message(self):
        model = self.model(context_tokens=2048)
        turn = dict(id=2, role="user", intent=dict(action="adopt", description="Use this version."))
        with self.assertRaisesRegex(RuntimeError, "context budget"):
            model(turn, {}, [dict(role="assistant", content="Important context. "*2000)])


if __name__ == "__main__":
    unittest.main()
