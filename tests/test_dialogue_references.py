"""Construction references must describe a realizable reverse/forward graph."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from scripts.dialogue_generation.planner import build_plan, validate_plan

EXAMPLES = Path(__file__).resolve().parents[1] / "scripts/dialogue_generation/examples"

class ReferenceTests(unittest.TestCase):
    def plan(self, seed=1):
        return build_plan(json.loads((EXAMPLES / "tokyo_vlog.json").read_text()), seed=seed, reworks=2, p_compile=.5)

    def test_target_root_reverse_edits_and_forward_exploration(self):
        plan = self.plan()
        rows = {m["placeholder"]: m for m in plan["placeholders"]}
        target = plan["final"]["media"]
        self.assertEqual(target, plan["placeholders"][-1]["placeholder"])
        self.assertEqual(rows[target]["reference_materials"], [])
        self.assertEqual(rows[target]["reference_depth"], 0)
        self.assertEqual(rows[target]["source_video"], plan["source_video"])
        # The cup starts gradient and ends plain: construct plain from target,
        # then gradient from plain, despite the dialogue's opposite chronology.
        cup = [m for m in rows.values() if m["purpose"] == "asset" and m["asset_ids"] == ["cup"]]
        before, after = cup[0], cup[-1]
        self.assertEqual(after["reference_materials"], [target])
        self.assertEqual(before["reference_materials"], [after["placeholder"]])
        self.assertEqual(after["inputs"], [before["placeholder"]])
        self.assertEqual((after["reference_depth"], before["reference_depth"]), (1, 2))
        for row in rows.values():
            if row["purpose"] == "shot" and row["placeholder"] != target:
                self.assertEqual(row["reference_materials"], [target])
            if row["purpose"] in {"exploration", "compilation"} and row["placeholder"] != target:
                self.assertEqual(row["reference_materials"], row["inputs"])
        trial = next(m for m in rows.values() if m["purpose"] == "exploration" and m["kind"] == "video")
        self.assertEqual(trial["reference_depth"], 1 + max(rows[p]["reference_depth"] for p in trial["inputs"]))
        self.assertTrue(any(m["reference_depth"] == 3 for m in rows.values() if m["purpose"] == "exploration"))

    def test_graph_closes_for_many_seeds_and_all_media_categories(self):
        purposes = set()
        for name in ("tokyo_vlog", "alpine_kayak_vlog"):
            source = json.loads((EXAMPLES / (name + ".json")).read_text())
            for seed in range(30):
                plan = build_plan(source, seed=seed, reworks=2, p_compile=.5)
                self.assertTrue(validate_plan(plan))
                purposes.update(m["purpose"] for m in plan["placeholders"])
        self.assertTrue({"asset", "shot", "exploration", "compilation"} <= purposes)
        # Force a late standalone preview after the final one-shot render.
        source = json.loads((EXAMPLES / "tokyo_vlog.json").read_text())
        source["shots"] = source["shots"][:1]; source["target_order"] = ["S1"]
        source["assets"] = [a for a in source["assets"] if a["id"] in source["shots"][0]["assets"]]
        source["constraints"] = []
        plan = build_plan(source, seed=12, reworks=0)
        self.assertEqual(plan["placeholders"][-1]["purpose"], "final_export")
        self.assertEqual(plan["placeholders"][-1]["reference_depth"], 0)

    def test_invalid_cycle_missing_reference_wrong_depth_and_policy_rejected(self):
        for failure in ("cycle", "missing", "depth", "policy"):
            plan = self.plan()
            row = plan["placeholders"][0]
            if failure == "cycle": row["reference_materials"] = [row["placeholder"]]
            if failure == "missing": row["reference_materials"] = ["[[video_9999]]"]
            if failure == "depth": row["reference_depth"] += 1
            if failure == "policy":
                row["reference_materials"] = [plan["final"]["shots"]["S1"]["media"]]
                row["reference_depth"] = 2
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_variant_selection_and_single_use_scope_are_explicit(self):
        plan = self.plan()
        turns = {t["id"]: t for t in plan["turns"]}
        self.assertIn("wood version", turns[13]["intent"]["description"])
        self.assertEqual(turns[13]["intent"]["required_attribute"]["value"], "wood")
        self.assertEqual(turns[57]["intent"]["scope"], "all_uses")
        self.assertIn("going forward", turns[57]["intent"]["description"])



















    def test_plan_validator_checks_exact_changed_references_per_clip(self):
        plan = self.plan()
        turn = next(t for t in plan["turns"] if t["intent"]["action"] == "request_shot_update")
        turn["intent"]["reference_updates"] = {}
        with self.assertRaisesRegex(ValueError, "revised references"):
            validate_plan(plan)




