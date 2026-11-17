"""Rule scope, independent intervals, real media effects, and causal validation."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.dialogue_generation.planner import Planner, build_plan, validate_plan, observation
from scripts.dialogue_generation.cross_turn_rule import (
    matches, normalize_rule, media_role, ROLES,
)

EXAMPLES = Path(__file__).resolve().parents[1] / 'scripts/dialogue_generation/examples'


def spec(effect, scope='global', roles=None):
    return dict(scope=scope, selector={} if scope == 'global' else dict(roles=roles), effect=effect, text='')


class CrossTurnRuleTests(unittest.TestCase):
    def planner(self):
        source = json.loads((EXAMPLES/'tokyo_vlog.json').read_text())
        source['constraints'] = []
        return Planner(source, seed=1, reworks=0, extra_behaviors=False, cross_turn_rules=False)

    def test_global_and_partial_cover_the_requested_media_roles(self):
        global_rule = spec(dict(color_mode='black_and_white'))
        partial = spec(dict(color_mode='black_and_white'), 'partial', ['asset_image'])
        for role in ROLES:
            row = dict(purpose='exploration', rule_role=role, kind='image' if role == 'asset_image' else 'video', asset_ids=['bag'])
            self.assertTrue(matches(global_rule, row))
            self.assertEqual(matches(partial, row), role == 'asset_image')
        self.assertEqual(media_role(dict(kind='image', purpose='exploration')), 'asset_image')
        self.assertEqual(media_role(dict(kind='video', purpose='exploration')), 'trial_video')

    def test_canonical_scopes_validate_selectors_and_legacy_inputs(self):
        p = self.planner()
        self.assertEqual(normalize_rule(dict(scope='all', effect=dict(season='fall')), p.assets)['scope'], 'global')
        self.assertEqual(normalize_rule(dict(scope='shots', effect=dict(season='fall')), p.assets)['scope'], 'partial')
        for bad in (dict(scope='partial', effect=dict(season='fall')),
                    dict(scope='global', selector=dict(roles=['asset_image']), effect=dict(season='fall')),
                    dict(scope='global', effect=dict(season='made-up')),
                    dict(scope='global', effect=[])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_rule(bad, p.assets)
        # Initial user rules need not already match the final target.
        source = deepcopy(p.source)
        source['constraints'] = [spec(dict(season='fall'))]
        self.assertEqual(observation(source)['constraints'][0]['effect'], dict(season='fall'))

    def test_overlapping_rules_can_end_in_either_order(self):
        for nested in (False, True):
            p = self.planner()
            first = p.rules.start(spec(dict(season='fall')))
            p.single('bag', 'direct')
            second = p.rules.start(spec(dict(color_mode='black_and_white'), 'partial', ['asset_image']))
            p.single('woman', 'direct')
            both = p.ready['woman']
            p.rules.end(second if nested else first)
            p.single('cup', 'direct')
            remaining = p.ready['cup']
            result = p.run()
            table = {m['placeholder']: m for m in result['placeholders']}
            self.assertEqual(table[both]['render_state'], dict(season='fall', color_mode='black_and_white'))
            self.assertEqual(table[remaining]['render_state'], dict(season='fall') if nested else dict(color_mode='black_and_white'))
            self.assertEqual(table[both]['render_state'], dict(season='fall', color_mode='black_and_white'))
            self.assertIn('Output requirements:', table[both]['description'])
            self.assertTrue(validate_plan(result))
            self.assertEqual(result['final']['assets']['woman'], both)
            self.assertEqual(sum(r['ended_at'] is None for r in result['rules']), 1)
            self.assertFalse(any(t['intent']['action'] == 'restore_rule_asset' for t in result['turns']))

    def test_once_is_one_request_and_does_not_leak_to_next_request(self):
        p = self.planner()
        rule = p.rules.start(spec(dict(color_mode='black_and_white')))
        p.single('bag', 'direct')
        source = p.ready['bag']
        p.rules.challenge(rule, outcome='once')
        rows = [m for m in p.media.values() if m['purpose'] == 'exploration']
        self.assertEqual([r['render_state']['color_mode'] for r in rows], ['original_colors', 'black_and_white'])
        # An exception wraps the complete creation transaction, including two candidates.
        with p.rules.exception({rule['id']: dict(color_mode='original_colors')}):
            pair = [p.add_media('image', 'Candidate', assets=['bag']) for _ in range(2)]
        self.assertTrue(all(p.media[t]['render_state']['color_mode'] == 'original_colors' for t in pair))
        del p.media[pair[0]]; del p.media[pair[1]]
        self.assertEqual(p.rules.snapshot(dict(kind='image', purpose='asset', asset_ids=['bag']))[0]['color_mode'], 'black_and_white')
        result = p.run()
        followup = next(t for t in result['turns'] if t['intent']['action']=='request_rule_followup')
        context = ' '.join(followup['intent']['media_rule_context'])
        self.assertIn('black-and-white rendering', context)
        self.assertNotIn('original-color rendering', context)
        self.assertIn('requested_media_role', followup['intent'])
        self.assertTrue(validate_plan(result))

    def test_keep_and_override_preserve_or_replace_the_actual_rule(self):
        for outcome in ('keep', 'override'):
            p = self.planner()
            rule = p.rules.start(spec(dict(color_mode='black_and_white'), 'partial', ['asset_image']))
            p.single('bag', 'direct')
            p.rules.challenge(rule, outcome=outcome)
            if outcome == 'keep':
                self.assertIn(rule['id'], p.rules.active)
                self.assertEqual(len(p.rules.records), 1)
            else:
                self.assertNotIn(rule['id'], p.rules.active)
                self.assertEqual(len(p.rules.records), 2)
                replacement = p.rules.records[-1]
                self.assertEqual(replacement['scope'], 'partial')
                self.assertEqual(replacement['selector'], rule['selector'])
                self.assertEqual(rule['ended_at'], replacement['started_at'])
            self.assertTrue(validate_plan(p.run()))

    def test_accepted_global_treatment_persists_without_terminal_restoration(self):
        p = self.planner()
        p.rules.start(spec(dict(color_mode='black_and_white')))
        p.single('bag', 'direct')
        before = p.ready['bag']
        result = p.run()
        table = {m['placeholder']:m for m in result['placeholders']}
        after = result['final']['assets']['bag']
        self.assertEqual(before, after)
        self.assertEqual(table[after]['render_state'], dict(color_mode='black_and_white'))
        self.assertIsNone(result['rules'][0]['ended_at'])
        self.assertEqual(table[before]['reference_materials'], [])

    def test_rules_are_reflected_in_exported_descriptions_and_resist_tampering(self):
        p = self.planner()
        rule = p.rules.start(spec(dict(color_mode='black_and_white')))
        p.single('bag', 'direct')
        p.rules.challenge(rule, outcome='once')
        p.rules.end(rule)
        result = p.run()
        for mode in ('reminder', 'resolution', 'state', 'description', 'cancellation', 'history', 'effective_attributes'):
            broken = deepcopy(result)
            if mode == 'reminder':
                next(t for t in broken['turns'] if t['intent']['action']=='clarify_rule_conflict')['intent']['action'] = 'confirm_selection'
            elif mode == 'resolution':
                next(t for t in broken['turns'] if t['intent']['action']=='resolve_rule_conflict')['intent']['rule_event']['ids'] = ['missing']
            elif mode == 'cancellation':
                next(t for t in broken['turns'] if t['intent']['action']=='confirm_rule_cancel')['intent']['rule_event']['id'] = 'missing'
            elif mode == 'history':
                broken['rules'][0]['ended_at'] = None
            else:
                row = next(m for m in broken['placeholders'] if m['purpose']=='asset' and m['render_state'])
                if mode == 'state': row['render_state'] = {}
                elif mode == 'description': row['description'] = row['base_description']
                else: row['effective_attributes']['color'] = 'unknown'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_plan(broken)

    def test_formal_shot_conflict_names_the_actual_changed_asset_and_category(self):
        p = self.planner()
        rule = p.rules.start(spec({'asset_attributes': {'street': {'time_of_day': 'night'}, 'cafe': {'time_of_day': 'night'}}}, 'partial', ['shot']))
        p.single('street', 'direct')
        p.single('cafe', 'direct')
        p.rules.challenge(rule, outcome='keep')
        request = next(t for t in p.turns if t['intent']['action']=='request_rule_conflict')
        reminder = next(t for t in p.turns if t['intent']['action']=='clarify_rule_conflict')
        changed = [a for a, attrs in reminder['intent']['rule_event']['proposed']['asset_attributes'].items()
                   if attrs != rule['effect']['asset_attributes'][a]]
        self.assertEqual(request['intent']['asset_ids'], changed)
        self.assertIn('a formal video shot', request['intent']['description'])
        self.assertNotIn('preview', request['intent']['description'])
        result = next(m for m in p.media.values() if m['purpose']=='exploration')
        self.assertEqual(result['rule_role'], 'shot')
        self.assertTrue(validate_plan(p.run()))

    def test_exception_and_override_resolve_every_overlapping_rule(self):
        for outcome in ('once', 'override'):
            p = self.planner()
            broad = p.rules.start(spec(dict(color_mode='black_and_white')))
            narrow = p.rules.start(spec(dict(color_mode='black_and_white', season='fall'), 'partial', ['asset_image']))
            p.single('bag', 'direct')
            # An asset-image request is simultaneously covered by both rules.
            from unittest.mock import patch
            with patch.object(p.rules, 'preview_context', return_value=dict(kind='image', purpose='exploration', rule_role='asset_image', asset_ids=['bag'], source=p.ready['bag'])):
                p.rules.challenge(broad, outcome=outcome)
            reminder = next(t for t in p.turns if t['intent']['action']=='clarify_rule_conflict')
            self.assertEqual(reminder['intent']['rule_event']['ids'], [broad['id'], narrow['id']])
            first = next(m for m in p.media.values() if m['purpose']=='exploration')
            self.assertEqual(first['render_state'], dict(color_mode='original_colors', season='fall'))
            if outcome == 'once':
                self.assertEqual(list(p.media.values())[-1]['render_state'], dict(color_mode='black_and_white', season='fall'))
            else:
                self.assertEqual({r['scope'] for r in p.rules.active.values()}, {'global', 'partial'})
                self.assertTrue(all(r['effect']['color_mode']=='original_colors' for r in p.rules.active.values()))
                self.assertTrue(any(r['effect'].get('season')=='fall' for r in p.rules.active.values()))
            self.assertTrue(validate_plan(p.run()))

    def test_partial_trial_rule_is_not_given_to_asset_image_requests(self):
        p = self.planner()
        p.rules.start(spec(dict(color_mode='black_and_white'), 'partial', ['trial_video']))
        p.single('bag', 'direct')
        result = p.run()
        request = next(t for t in result['turns'] if t['intent']['action']=='request_asset' and t['intent']['asset_ids']==['bag'])
        context = ' '.join(request['intent']['media_rule_context'])
        self.assertIn('standalone asset images', context)
        self.assertIn('no additional rule treatment applies', context)
        self.assertNotIn('black-and-white', context)

    def test_media_description_does_not_add_assets_outside_the_output(self):
        p = self.planner()
        p.rules.start(spec({'asset_attributes': {'street': {'time_of_day':'night'}, 'cafe': {'time_of_day':'night'}}}))
        p.single('cafe', 'direct')
        result = p.run()
        row = next(m for m in result['placeholders'] if m['purpose']=='asset' and m['asset_ids']==['cafe'])
        self.assertIn('coffee shop', row['description'])
        self.assertNotIn('Tokyo street', row['description'])
        request = next(t for t in result['turns'] if t['intent']['action']=='request_asset' and t['intent']['asset_ids']==['cafe'])
        self.assertNotIn('Tokyo street', ' '.join(request['intent']['media_rule_context']))

    def test_asset_scoped_rule_matches_cuts_and_exports_containing_that_asset(self):
        p = self.planner()
        p.rules.start(dict(scope='partial', selector=dict(asset_ids=['bag']), effect=dict(color_mode='black_and_white'), text=''))
        p.single('bag', 'direct')
        clip = p.add_media('video', 'Trial', assets=['bag'], inputs=[p.ready['bag']], purpose='exploration')
        p.emit('assistant','deliver_exploration','Here is the trial.',outputs=[clip])
        cut = p.add_media('video','Cut',inputs=[clip],purpose='compilation')
        p.emit('assistant','deliver_compilation','Here is the cut.',outputs=[cut])
        exported = p.add_media('video','Export',inputs=[cut],purpose='final_export')
        for token in (cut, exported):
            self.assertEqual(p.media[token]['asset_ids'], ['bag'])
            self.assertEqual(p.media[token]['render_state']['color_mode'], 'black_and_white')
            self.assertIn('bag',p.media[token]['source_frames'])

    def test_rule_question_judge_requires_permission_and_all_decisions(self):
        from tools.dialogue_verifier import build_prompt
        prompt = build_prompt('I will make an exception.', dict(speaker='assistant', action='clarify_rule_conflict'))
        self.assertIn('premature decision', prompt)
        self.assertIn('retain the earlier rule', prompt)
        prompt = build_prompt('Make a color clip.', dict(speaker='user', action='request_rule_conflict'))
        self.assertIn('deliberately conflicts', prompt)
        self.assertIn('not a grounding or coherence error', prompt)

    def test_many_seeds_cover_scopes_decisions_and_independent_lifetimes(self):
        outcomes, scopes, relations, appearances = set(), set(), set(), set()
        for name in ('tokyo_vlog', 'alpine_kayak_vlog'):
            source = json.loads((EXAMPLES/(name+'.json')).read_text())
            for seed in range(60):
                plan = build_plan(source, seed=seed, reworks=seed%3)
                self.assertTrue(validate_plan(plan))
                scopes.update(r['scope'] for r in plan['rules'])
                outcomes.update(t['intent']['rule_event']['outcome'] for t in plan['turns'] if t['intent'].get('rule_event',{}).get('type')=='resolve')
                appearances.update(tuple(sorted(r['render_state'])) for r in plan['placeholders'])
                rules = plan['rules']
                for a in rules:
                    for b in rules:
                        if a['started_at'] < b['started_at']:
                            relations.add('disjoint' if (a['ended_at'] or float('inf')) < b['started_at'] else
                                          'nested' if (b['ended_at'] or float('inf')) < (a['ended_at'] or float('inf')) else 'overlap')
        self.assertEqual(scopes, {'global', 'partial'})
        self.assertEqual(outcomes, {'keep', 'once', 'override'})
        self.assertTrue({'disjoint', 'nested', 'overlap'} <= relations)
        self.assertTrue(any('season' in keys and 'color_mode' in keys for keys in appearances))


if __name__ == '__main__':
    unittest.main()
