"""Persistent settings apply to images/videos; format exceptions compare actual output."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.planner import Planner, build_plan, validate_plan
from scripts.dialogue_generation.cross_turn_rule import normalize_rule, matches, output_aspect, different_for_kind

ROOT = Path(__file__).resolve().parents[1]


def spec(effect, *, kinds=None):
    return dict(scope='partial' if kinds else 'global', selector=dict(kinds=kinds) if kinds else {}, effect=effect, text='')


class CrossTurnRuleTests(unittest.TestCase):
    def planner(self):
        source=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        source['constraints']=[]
        return Planner(source,seed=1,reworks=0,extra_behaviors=False,cross_turn_rules=False)

    def test_rules_use_only_media_kind_selectors(self):
        p=self.planner()
        rule=normalize_rule(spec(dict(season='winter'),kinds=['image']),p.assets)
        for kind in ('image','video'):
            self.assertEqual(matches(rule,dict(kind=kind,asset_ids=['bag'])),kind=='image')
        with self.assertRaises(ValueError):
            normalize_rule(dict(scope='partial',selector=dict(roles=['trial_video']),effect=dict(season='winter')),p.assets)

    def test_retired_color_effect_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_rule(spec(dict(color_mode='black_and_white')),self.planner().assets)

    def test_invalid_scope_effect_and_selectors_are_rejected(self):
        p=self.planner()
        for bad in [spec(dict(season='made-up')),dict(scope='partial',effect=dict(season='spring')),
                    spec(dict(season='winter'),kinds=['trial']),dict(scope='global',selector=dict(kinds=['image']),effect=dict(season='spring'))]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):normalize_rule(bad,p.assets)

    def test_no_explicit_format_request_uses_defaults_without_rule_dialogue(self):
        p=self.planner();result=p.run()
        self.assertFalse(result['rules'])
        self.assertFalse(any('aspect_ratio' in t['intent']['description'] for t in result['turns']))
        from scripts.dialogue_generation.output_geometry import geometry
        self.assertEqual(geometry({},'image')[1],(1024,1024))
        self.assertEqual(geometry({},'video')[1],(1280,720))

    def test_ratio_cannot_be_canceled_and_past_media_remain_unchanged(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='portrait')))
        p.single('bag','direct');old=p.ready['bag']
        count=len(p.turns)
        with self.assertRaisesRegex(ValueError,'cannot be canceled'):p.rules.end(rule)
        self.assertEqual(len(p.turns),count)
        result=p.run();self.assertTrue(validate_plan(result))
        self.assertEqual(p.media[old]['render_state']['aspect_ratio'],'portrait')
        self.assertIsNone(result['rules'][0]['ended_at'])

    def test_season_can_be_canceled_prospectively(self):
        p=self.planner();rule=p.rules.start(spec(dict(season='fall')))
        p.single('bag','direct');old=p.ready['bag'];p.rules.end(rule);p.single('cup','direct')
        self.assertEqual(p.media[old]['render_state'],dict(season='fall'))
        self.assertEqual(p.media[p.ready['cup']]['render_state'],{})
        self.assertTrue(validate_plan(p.run()))

    def test_time_of_day_reaches_image_and_video_and_cancels_prospectively(self):
        from scripts.dialogue_generation.planner_rules import media_rules
        p = self.planner()
        rule = p.rules.start(spec(dict(time_of_day='dusk')))
        p.single('bag', 'direct')
        before = p.ready['bag']
        video = p.add_media('video', 'Bag in a courtyard', assets=['bag'], inputs=[before], operation='generate')
        p.emit('user', 'request_trial', 'Make a test video of the bag.', assets=['bag'])
        p.emit('assistant', 'deliver_trial', 'Show the test video.', outputs=[video], assets=['bag'])
        p.rules.end(rule)
        p.single('cup', 'direct')
        plan = p.run()
        self.assertTrue(validate_plan(plan))
        self.assertEqual(media_rules(plan, before), [{'requirement': 'Use dusk as the time of day.'}])
        self.assertEqual(media_rules(plan, video), [{'requirement': 'Use dusk as the time of day.'}])
        self.assertEqual(media_rules(plan, p.ready['cup']), [])
        self.assertIn('lighting or visible sky', p.media[video]['description'])

    def test_ratio_preset_equivalence_is_checked_for_the_requested_kind(self):
        self.assertFalse(different_for_kind(dict(aspect_ratio='square_assets_landscape_video'),dict(aspect_ratio='square_assets_portrait_video'),'image'))
        self.assertTrue(different_for_kind(dict(aspect_ratio='square_assets_landscape_video'),dict(aspect_ratio='square_assets_portrait_video'),'video'))
        self.assertFalse(different_for_kind(dict(aspect_ratio='square_assets_portrait_video'),dict(aspect_ratio='1:1'),'image'))

    def test_explicit_square_exception_applies_once_then_portrait_returns(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='portrait')));p.single('bag','direct')
        context=p.rules.preview_context(rule,kind='image')
        count=len(p.media)
        p.rules.one_time(rule,context,dict(aspect_ratio='1:1'))
        self.assertEqual(len(p.media),count)
        p.single('cup','direct');exception=p.ready['cup']
        p.single('woman','direct');followup=p.ready['woman']
        requests=[t for t in p.turns if t['intent'].get('rule_event',{}).get('type')=='exception']
        self.assertEqual(len(requests),1)
        self.assertEqual(requests[0]['intent']['action'],'request_asset')
        self.assertEqual(requests[0]['intent']['asset_ids'],['cup'])
        self.assertNotIn('clarify_rule_conflict',[t['intent']['action'] for t in p.turns])
        images=[p.media[exception],p.media[followup]]
        self.assertEqual([output_aspect(m['render_state']['aspect_ratio'],'image') for m in images],['1:1','9:16'])
        self.assertTrue(validate_plan(p.run()))

    def test_square_image_exception_is_not_requested_when_images_already_square(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='square_assets_portrait_video')));p.single('bag','direct')
        context=p.rules.preview_context(rule,kind='image');count=len(p.turns)
        with self.assertRaisesRegex(ValueError,'must visibly differ'):p.rules.one_time(rule,context,dict(aspect_ratio='1:1'))
        self.assertEqual(len(p.turns),count)
        for _ in range(20):
            chosen=p.rules.changed_effect(rule,kind='image',once=True)
            self.assertNotEqual(output_aspect(chosen['aspect_ratio'],'image'),'1:1')

    def test_exception_does_not_change_other_media_kind_in_the_same_request(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='portrait')))
        with p.rules.exception({rule['id']:dict(aspect_ratio='1:1')},kind='image'):
            image=p.rules.snapshot(dict(kind='image',asset_ids=['bag']))[0]
            video=p.rules.snapshot(dict(kind='video',asset_ids=['bag']))[0]
        self.assertEqual(image['aspect_ratio'],'1:1');self.assertEqual(video['aspect_ratio'],'portrait')
        self.assertEqual(p.rules.snapshot(dict(kind='image',asset_ids=['bag']))[0]['aspect_ratio'],'portrait')

    def test_explicit_replacement_is_permanent_and_preserves_scope(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='portrait')));p.single('bag','direct')
        p.rules.challenge(rule,outcome='override',kind='image')
        self.assertIn(rule['id'],p.rules.active)
        count=len(p.media)
        p.single('cup','direct')
        self.assertEqual(len(p.media),count+1)
        self.assertNotIn(rule['id'],p.rules.active)
        replacement=next(iter(p.rules.active.values()))
        self.assertEqual(replacement['selector'],rule['selector'])
        self.assertNotEqual(output_aspect(replacement['effect']['aspect_ratio'],'image'),'9:16')
        self.assertTrue(validate_plan(p.run()))

    def test_keep_does_not_replace_existing_setting(self):
        p=self.planner();rule=p.rules.start(spec(dict(season='winter')));p.single('bag','direct')
        p.rules.challenge(rule,outcome='keep',kind='image')
        count=len(p.media);p.single('cup','direct')
        self.assertEqual(len(p.media),count+1)
        self.assertEqual(p.rules.active[rule['id']]['effect'],dict(season='winter'))
        self.assertTrue(validate_plan(p.run()))

    def test_validator_rejects_ratio_cancel_and_redundant_exception(self):
        p=self.planner();rule=p.rules.start(spec(dict(aspect_ratio='portrait')));p.single('bag','direct')
        p.rules.one_time(rule,p.rules.preview_context(rule,kind='image'),dict(aspect_ratio='1:1'))
        p.single('cup','direct')
        result=p.run();broken=deepcopy(result)
        turn=next(t for t in broken['turns'] if t['intent'].get('rule_event',{}).get('type')=='exception')
        turn['intent']['rule_event']['effect']=dict(aspect_ratio='portrait')
        with self.assertRaisesRegex(ValueError,'repeats'):validate_plan(broken)

    def test_independent_draws_can_start_cancel_and_challenge_in_one_check(self):
        planner = self.planner()
        rules = planner.rules
        image_season = rules.start(spec(dict(season='winter'), kinds=['image']))
        video_season = rules.start(spec(dict(season='summer'), kinds=['video']))
        planner.single('bag', 'direct')
        rules.enabled = True
        choices = ['global', 'aspect_ratio', 'portrait', image_season, video_season, 'keep']
        with patch.object(rules.rng, 'random', side_effect=[.19, .29, .19]) as draw, \
                patch.object(rules.rng, 'choice', side_effect=choices) as choice:
            rules.advance()
        self.assertEqual(draw.call_count, 3)
        self.assertEqual(rules.records[-1]['effect'], dict(aspect_ratio='portrait'))
        self.assertNotIn(image_season['id'], rules.active)
        self.assertEqual(choice.call_args_list[-2].args[0], [video_season])
        self.assertEqual(rules.pending['rule_id'], video_season['id'])

    def test_cancel_draw_ignores_format_even_with_a_cancellable_season(self):
        rules = self.planner().rules
        ratio = rules.start(spec(dict(aspect_ratio='portrait')))
        season = rules.start(spec(dict(season='winter')))
        rules.enabled = True
        with patch.object(rules.rng, 'random', side_effect=[.20, .30, .19]), \
                patch.object(rules.rng, 'choice', return_value=ratio) as choice:
            rules.advance()
        choice.assert_called_once_with([ratio, season])
        self.assertEqual(list(rules.active), [ratio['id'], season['id']])

    def test_cancel_drops_pending_challenge_and_a_new_rule_is_ineligible(self):
        planner = self.planner()
        rules = planner.rules
        season = rules.start(spec(dict(season='winter')))
        planner.single('bag', 'direct')
        rules.challenge(season, outcome='keep')
        rules.enabled = True
        with patch.object(rules.rng, 'random', side_effect=[.19, .29, .19]), \
                patch.object(rules.rng, 'choice', side_effect=['global', 'aspect_ratio', 'landscape', season]):
            rules.advance()
        self.assertIsNone(rules.pending)
        self.assertEqual(list(rules.active.values()), [rules.records[-1]])
        self.assertEqual(rules.records[-1]['effect'], dict(aspect_ratio='landscape'))

    def test_incompatible_new_rule_does_not_block_challenge(self):
        planner = self.planner()
        rules = planner.rules
        season = rules.start(spec(dict(season='winter')))
        planner.single('bag', 'direct')
        rules.enabled = True
        choices = ['global', 'season', 'summer', season, 'keep']
        with patch.object(rules.rng, 'random', side_effect=[.19, .29, .20]), \
                patch.object(rules.rng, 'choice', side_effect=choices):
            rules.advance()
        self.assertEqual(rules.records, [season])
        self.assertEqual(rules.pending['rule_id'], season['id'])

    def test_seeded_plans_never_cancel_format_or_repeat_effective_exception(self):
        source=self.planner().source
        found=set()
        for seed in range(50):
            plan=build_plan(source,seed=seed,reworks=1)
            rules={r['id']:r for r in plan['rules']}
            for turn in plan['turns']:
                event=turn['intent'].get('rule_event',{})
                if event.get('type')=='end':self.assertNotIn('aspect_ratio',rules[event['id']]['effect'])
                if event.get('type')=='exception':
                    for rid in event['ids']:self.assertTrue(different_for_kind(rules[rid]['effect'],event['effect'],event['kind']))
                    found.add('once')
            found.update(r['effect'].get('aspect_ratio') for r in plan['rules'] if 'aspect_ratio' in r['effect'])
            self.assertTrue(validate_plan(plan))
        self.assertIn('once',found)
        self.assertTrue({'portrait','landscape'} <= found)


if __name__=='__main__':unittest.main()
