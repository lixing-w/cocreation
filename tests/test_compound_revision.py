"""Related requests share one scope-confirmed generation transaction."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock
from scripts.dialogue_generation.planner import Planner, validate_plan
from scripts.dialogue_generation.generation import build_tasks
from tools.dialogue_realizer import QwenRealizer
from tools.dialogue_verifier import CRITERIA

ROOT = Path(__file__).resolve().parents[1]


def source(two_assets=False):
    example = json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
    bag = deepcopy(next(a for a in example['assets'] if a['id'] == 'bag'))
    assets = [bag]
    if two_assets:
        other = deepcopy(bag)
        other.update(id='case', name='case', description='A distinct travel case.')
        assets.append(other)
    return dict(assets=assets, shots=[dict(id=s, description=f'A bag moves through scene {s}.',
                assets=[a['id'] for a in assets]) for s in ('S1','S2')],
                target_order=['S2','S1'], constraints=[])


def staged(seed=0, two_assets=False):
    p = Planner(source(two_assets), seed=seed, reworks=2 if two_assets else 1,
                p_compile=0, modules=['direct'], cross_turn_rules=False)
    p.conflict_done = True
    for a in p.assets:
        p.single(a, 'direct')
    p.render_due()
    p.rules.start(dict(scope='global', selector={}, effect=dict(season='winter'), text=''))
    return p


class CompoundRevisionTests(unittest.TestCase):
    def plan(self, seed=0, two_assets=False):
        p = staged(seed, two_assets)
        before = dict(p.rendered)
        p.compilation(['S2','S1'], 'request_compilation')
        plan = p.run()
        delivery = next(t for t in plan['turns'] if t['intent'].get('compound_request')
                        and t['intent']['action'] == 'deliver_reworked_clips')
        return plan, before, delivery

    def test_scope_is_confirmed_before_videos_and_each_clip_is_updated_once(self):
        plan, before, delivery = self.plan()
        request = next(t for t in plan['turns'] if t['intent'].get('compound_request')
                       and t['intent']['action'] == 'request_reworks')
        question, answer = plan['turns'][request['id']:request['id']+2]
        rows = {m['placeholder']:m for m in plan['placeholders']}
        self.assertIn('Put the existing clips together', request['intent']['description'])
        self.assertIn('requested cut', request['intent']['description'])
        self.assertIn('winter', question['intent']['description'])
        self.assertIn('assemble', question['intent']['description'])
        self.assertEqual(answer['intent']['action'], 'resolve_reworks')
        self.assertFalse(any(rows[p]['kind']=='video' for t in [request, question, answer] for p in t['output_media']))
        clips = delivery['intent']['updated_clips']
        self.assertEqual(len(clips), 2)
        self.assertEqual({rows[c]['shot_ids'][0] for c in clips}, {'S1','S2'})
        self.assertTrue(all(rows[c]['render_state']['season']=='winter' for c in clips))
        self.assertTrue(all(rows[c]['reference_materials'][0] == before[rows[c]['shot_ids'][0]] for c in clips))
        tasks = build_tasks(plan, source())
        self.assertFalse(tasks[delivery['intent']['updated_compilation']].get('clip_edits'))
        self.assertTrue(validate_plan(plan))

    def test_local_identity_scope_is_independent_of_cut_wide_season_changes(self):
        seen = set()
        for seed in range(20):
            plan, before, delivery = self.plan(seed)
            rows = {m['placeholder']:m for m in plan['placeholders']}
            edit = delivery['intent']['resolved_edits'][0]
            seen.add(edit['scope'])
            for token in delivery['intent']['updated_clips']:
                row = rows[token]
                shot = row['shot_ids'][0]
                expected = edit['new_media'] if shot in edit['applied_shots'] else rows[before[shot]]['bindings']['bag']
                self.assertEqual(row['bindings']['bag'], expected)
            self.assertTrue(validate_plan(plan))
        self.assertEqual(seen, {'shot','all_uses'})

    def test_later_independent_feedback_is_kept_as_another_transaction(self):
        p = staged(two_assets=True)
        p.rework_group(['bag'], compilation_order=['S2','S1'], render_updates=p.compilation_render_updates(['S2','S1']))
        plan = p.run()
        deliveries = [t for t in plan['turns'] if t['intent']['action']=='deliver_reworked_clips']
        self.assertEqual(len(deliveries), 2)
        self.assertEqual(len(deliveries[0]['intent']['resolved_edits']), 1)
        self.assertFalse(deliveries[1]['intent'].get('compound_request'))
        self.assertTrue(validate_plan(plan))

    def test_user_can_add_another_edit_while_answering_without_two_video_passes(self):
        plan, _, delivery = self.plan(two_assets=True)
        additions = [t for t in plan['turns'] if t['intent'].get('additional_edits')]
        self.assertEqual(len(additions), 1)
        self.assertEqual(len(delivery['intent']['resolved_edits']), 2)
        self.assertEqual(len(delivery['intent']['updated_clips']), 2)
        opening = next(t for t in plan['turns'] if t['intent']['action']=='request_reworks')
        self.assertEqual(len(opening['intent']['edits']), 1)
        self.assertGreater(additions[0]['id'], opening['id'])
        rows = {m['placeholder']:m for m in plan['placeholders']}
        self.assertFalse(any(rows[token]['kind']=='video' for t in plan['turns'][opening['id']-1:delivery['id']-1]
                             for token in t['output_media']))
        self.assertTrue(validate_plan(plan))

    def test_added_edit_must_be_resolved_before_delivery(self):
        plan, _, _ = self.plan(two_assets=True)
        extension = next(t for t in plan['turns'] if t['intent'].get('additional_edits'))
        final_answer = next(t for t in plan['turns'][extension['id']:] if t['intent']['action']=='resolve_reworks')
        plan['turns'].remove(final_answer)
        with self.assertRaisesRegex(ValueError, 'before all'):
            validate_plan(plan)

    def test_added_edit_cannot_silently_replace_the_confirmed_first_scope(self):
        plan, _, delivery = self.plan(two_assets=True)
        edit = delivery['intent']['resolved_edits'][0]
        edit['scope'] = 'shot' if edit['scope']=='all_uses' else 'all_uses'
        with self.assertRaisesRegex(ValueError, 'confirmed scope'):
            validate_plan(plan)

    def test_validator_rejects_omitted_render_update_and_wrong_assembly_order(self):
        plan, _, delivery = self.plan()
        for kind in ('render', 'order', 'scope'):
            broken = deepcopy(plan)
            turn = broken['turns'][delivery['id']-1]
            rows = {m['placeholder']:m for m in broken['placeholders']}
            if kind == 'render':
                rows[turn['intent']['updated_clips'][0]]['render_state'] = {}
            elif kind == 'order':
                turn['intent']['compilation_order'].reverse()
            else:
                edit = turn['intent']['resolved_edits'][0]
                edit['scope'] = 'shot' if edit['scope']=='all_uses' else 'all_uses'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_plan(broken)

    def test_assembly_preserves_both_old_and_new_clips_after_winter_rule(self):
        p = Planner(source(), seed=0, reworks=0, p_compile=0, extra_behaviors=False, cross_turn_rules=False)
        p.single('bag', 'direct')
        for sid in ('S1','S2'):
            if sid == 'S2':
                p.rules.start(dict(scope='global',selector={},effect=dict(season='winter'),text=''))
            ref = p.ready['bag']
            token = p.add_media('video', p.shots[sid]['description'], assets=['bag'], shots=[sid],
                                inputs=[ref], operation='generate_clip', bindings=dict(bag=ref))
            p.emit('assistant','deliver_shot','Here is the shot.',shots=[sid],outputs=[token])
            p.rendered[sid]=token; p.bindings[sid]=dict(bag=ref); p.render_order.append(sid)
        original = dict(p.rendered)
        p.compilation(['S2','S1'], 'request_compilation')
        plan = p.run()
        task = build_tasks(plan, source())[plan['final']['media']]
        self.assertFalse(task.get('clip_edits'))
        self.assertIsNone(task['backend'])
        self.assertIsNone(task['prompt'])
        self.assertEqual([clip['placeholder'] for clip in task['clips']], [original['S2'], original['S1']])
        self.assertNotIn('season',task['render_state'])

    def test_equivalent_video_ratios_do_not_trigger_a_compound_edit(self):
        p = staged()
        # Both presets mean landscape video; their image settings differ.
        for token in p.rendered.values():
            p.media[token]['render_state'] = dict(season='winter', aspect_ratio='landscape')
        p.rules.snapshot = Mock(return_value=(dict(season='winter',aspect_ratio='square_assets_landscape_video'), []))
        self.assertEqual(p.compilation_render_updates(['S1','S2']), {})

    def test_writer_keeps_acknowledgment_and_scope_question_together(self):
        plan, _, _ = self.plan()
        turn = next(t for t in plan['turns'] if t['intent'].get('compound_request')
                    and t['intent']['action']=='clarify_reworks')
        model = QwenRealizer(backend='ollama', asset_names={'bag':'bag'})
        model._request = Mock(return_value='I will assemble the clips and apply winter. Should I update the bag in one shot or all its shots?')
        model.verifier = Mock(return_value=dict(dict.fromkeys(CRITERIA,2), critique=''))
        media = {m['placeholder']:m for m in plan['placeholders']}
        model(turn, media, [], seed=1)
        prompt = model._request.call_args.args[0]
        self.assertIn('winter', prompt)
        self.assertIn('assemble', prompt)
        self.assertIn('outstanding scope/reference questions', model._writer_control)
        self.assertIn('pending, not completed', model._writer_control)


if __name__ == '__main__':
    unittest.main()
