"""Construction references must describe a realizable reverse/forward graph."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from scripts.dialogue_generation.planner import build_plan, validate_plan

EXAMPLES = Path(__file__).resolve().parents[1] / "scripts/dialogue_generation/examples"

class ReferenceTests(unittest.TestCase):
    def plan(self, seed=1):
        return build_plan(json.loads((EXAMPLES / "tokyo_vlog.json").read_text()), seed=seed, reworks=2, p_compile=.5, cross_turn_rules=False)

    def test_forward_inputs_and_shot_edits(self):
        plan = self.plan()
        rows, latest = {}, {}
        for row in plan['placeholders']:
            refs = row['reference_materials']
            self.assertTrue(all(ref in rows for ref in refs))
            self.assertNotIn('source_video',row)
            self.assertEqual(row['reference_depth'],max((rows[ref]['reference_depth']+1 for ref in refs),default=0))
            if row['purpose']=='shot':
                sid=row['shot_ids'][0]
                if sid in latest:
                    self.assertEqual(refs[0],latest[sid])
                latest[sid]=row['placeholder']
            rows[row['placeholder']]=row
        for turn in plan['turns']:
            self.assertNotIn('reference_media',turn)
            if turn['output_media']:
                expected=list(dict.fromkeys(ref for token in turn['output_media'] for ref in rows[token]['reference_materials'] if ref not in turn['output_media']))
                self.assertEqual(turn['input_media'],expected)
            for token in turn['intent'].get('display_media',[]):
                self.assertIn(token,turn['input_media'])

    def test_graph_closes_across_seeds_and_includes_video_to_image_edges(self):
        purposes, crop_edges = set(), []
        source=json.loads((EXAMPLES/'tokyo_vlog.json').read_text())
        for seed in range(12):
            plan=build_plan(source,seed=seed,reworks=2,p_compile=.5,modules=['candidates'])
            self.assertTrue(validate_plan(plan))
            rows={row['placeholder']:row for row in plan['placeholders']}
            for row in rows.values():
                purposes.add(row['purpose'])
                for ref in row['reference_materials']:
                    self.assertLessEqual(rows[ref]['introduced_at'],row['introduced_at'])
                    if row['kind']=='image' and rows[ref]['kind']=='video':
                        crop_edges.append((seed,ref,row['placeholder']))
        self.assertTrue({'asset','shot','exploration','compilation'}<=purposes)
        self.assertTrue(crop_edges)

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
        variant = next(t for t in plan['turns'] if t['intent']['action']=='request_variant')
        required=variant['intent']['required_attribute']
        delivered=next(t for t in plan['turns'][variant['id']:] if t['intent']['action']=='deliver_variant')
        output=next(m for m in plan['placeholders'] if m['placeholder']==delivered['output_media'][0])
        self.assertEqual(output['attributes'][required['field']],required['value'])
        scope = next((t for t in plan['turns'] if t['intent']['action']=='choose_update_scope'),None)
        for seed in range(12):
            if scope:break
            scope=next((t for t in self.plan(seed)['turns'] if t['intent']['action']=='choose_update_scope'),None)
        self.assertIsNotNone(scope)
        self.assertIn(scope['intent']['scope'], ('shot', 'all_uses'))
        self.assertNotIn('future',scope['intent']['description'])



















    def test_plan_validator_checks_exact_changed_references_per_clip(self):
        plan = self.plan()
        turn = next(t for t in plan['turns'] if t['intent']['action']=='deliver_reworked_clips')
        edit=turn['intent']['resolved_edits'][0]
        row=next(m for m in plan['placeholders'] if m['placeholder']==turn['intent']['updated_clips'][0])
        row['bindings'][edit['asset_id']]=edit['target_media']
        with self.assertRaisesRegex(ValueError, "confirmed revision"):
            validate_plan(plan)




