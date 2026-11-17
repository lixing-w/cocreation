"""Bundled edits resolve every ambiguity before producing media or updating shots."""
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.planner import build_plan, validate_plan
from scripts.dialogue_generation.conflicts import CONFLICTS
from tools.dialogue_realizer import QwenRealizer, DEFAULT_OLLAMA_MODEL

EXAMPLE = Path(__file__).resolve().parents[1] / 'scripts/dialogue_generation/examples/alpine_kayak_vlog.json'


class BatchRevisionTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(EXAMPLE.read_text())
        self.plan = build_plan(self.source, seed=0, reworks=2, p_compile=.5, cross_turn_rules=False)
        self.request = next(t for t in self.plan['turns'] if t['intent']['action'] == 'request_reworks')
        self.questions, self.answer, self.delivery = self.plan['turns'][self.request['id']:self.request['id'] + 3]

    def test_both_ambiguities_and_scopes_are_asked_in_one_reply(self):
        self.assertTrue(all(e['ambiguous'] for e in self.request['intent']['edits']))
        self.assertEqual(self.questions['intent']['action'], 'clarify_reworks')
        self.assertEqual(self.answer['intent']['action'], 'resolve_reworks')
        self.assertEqual(self.delivery['intent']['action'], 'deliver_reworked_clips')
        self.assertEqual({(q['edit_index'], q['kind']) for q in self.questions['intent']['clarifications']},
                         {(1, 'reference'), (2, 'reference'), (1, 'scope'), (2, 'scope')})
        self.assertFalse(self.request['input_media'] or self.request['intent'].get('display_media', []))
        self.assertTrue(self.questions['intent'].get('display_media', []))
        self.assertEqual(len(self.delivery['intent']['preview_media']),2)
        self.assertTrue(self.delivery['intent']['updated_clips'])
        self.assertTrue(validate_plan(self.plan))

    def test_overlapping_shots_update_once_after_both_edits(self):
        uses = [set(e['uses']) for e in self.request['intent']['edits']]
        self.assertTrue(uses[0] & uses[1])
        self.assertTrue(all(e['scope'] == 'all_uses' for e in self.answer['intent']['resolved_edits']))
        rows={m['placeholder']:m for m in self.plan['placeholders']}
        changed=[s for token in self.delivery['intent']['updated_clips'] for s in rows[token]['shot_ids']]
        self.assertEqual(set(changed), uses[0] | uses[1])
        self.assertEqual(len(changed), len(set(changed)))

    def test_validator_rejects_missing_question_unresolved_delivery_and_scope_swap(self):
        broken = deepcopy(self.plan)
        broken['turns'][self.questions['id'] - 1]['intent']['clarifications'].pop()
        with self.assertRaisesRegex(ValueError, 'omitted'):
            validate_plan(broken)
        broken = deepcopy(self.plan)
        broken['turns'].remove(broken['turns'][self.answer['id'] - 1])
        with self.assertRaisesRegex(ValueError, 'before all'):
            validate_plan(broken)
        broken = deepcopy(self.plan)
        edit = broken['turns'][self.delivery['id'] - 1]['intent']['resolved_edits'][0]
        edit['scope'] = 'shot' if edit['scope'] == 'all_uses' else 'all_uses'
        with self.assertRaisesRegex(ValueError, 'confirmed scope'):
            validate_plan(broken)

    def test_mixed_scopes_keep_each_assets_own_decision_and_close(self):
        found = False
        for seed in range(50):
            plan = build_plan(self.source, seed=seed, reworks=2, p_compile=1)
            reply = next((t for t in plan['turns'] if t['intent']['action'] == 'resolve_reworks'), None)
            if reply and {e['scope'] for e in reply['intent']['resolved_edits']} == {'shot', 'all_uses'}:
                self.assertTrue(validate_plan(plan))
                local = next(e for e in reply['intent']['resolved_edits'] if e['scope'] == 'shot')
                delivery=plan['turns'][reply['id']]
                self.assertEqual(delivery['intent']['action'],'deliver_reworked_clips')
                applied=next(e for e in delivery['intent']['resolved_edits'] if e['asset_id']==local['asset_id'])
                self.assertEqual(applied['applied_shots'],[local['target_shot']])
                self.assertFalse(any(t['intent']['action']=='synchronize_revision' for t in plan['turns'][reply['id']:]))
                found = True
                break
        self.assertTrue(found)








    def test_environment_only_shot_does_not_invent_an_object_conflict(self):
        source = deepcopy(self.source)
        source['assets'] = [a for a in source['assets'] if a['id'] == 'shore']
        source['shots'] = [dict(id='S1', description='Static establishing shot of the alpine lakeshore', assets=['shore'])]
        source['target_order'] = ['S1']
        for rule in source['constraints']:
            rule['asset_attributes'] = {a:v for a,v in rule.get('asset_attributes',{}).items() if a == 'shore'}
        seen = set()
        for seed in range(30):
            plan = build_plan(source,seed=seed,reworks=0,modules=['direct'])
            seen.update(t['intent']['conflict_type'] for t in plan['turns'] if t['intent']['action']=='clarify_conflict')
        self.assertTrue(seen)
        self.assertFalse(seen.intersection({'framing','camera_motion'}))

    def test_default_ollama_model_is_30b_and_8b_remains_selectable(self):
        self.assertEqual(DEFAULT_OLLAMA_MODEL, 'qwen3-vl:30b-a3b-instruct-q4_K_M')
        self.assertEqual(QwenRealizer(backend="ollama").model, 'qwen3-vl:30b-a3b-instruct-q4_K_M')
        self.assertEqual(QwenRealizer(backend='ollama', model='qwen3-vl:8b').model, 'qwen3-vl:8b')

    def test_five_conflict_types_are_sampled_with_correct_reasons(self):
        found = set()
        for seed in range(70):
            plan = build_plan(self.source, seed=seed, reworks=0)
            for t in plan['turns']:
                if t['intent']['action'] == 'clarify_conflict':
                    kind = t['intent']['conflict_type']
                    found.add(kind)
                    spec = CONFLICTS[kind]
                    for term in spec['terms']:
                        self.assertRegex(t['intent']['description'], term)
                    self.assertRegex(t['intent']['description'], spec['reason'])
            if found == set(CONFLICTS):
                break
        self.assertEqual(found, set(CONFLICTS))



if __name__ == '__main__':
    unittest.main()
