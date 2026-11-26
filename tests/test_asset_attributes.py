"""Finite observation vocabulary, actual version state, and rework scope."""
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import unittest
from unittest.mock import patch

from scripts.dialogue_generation.asset_variations import (
    CATALOG, CATEGORIES, POOLS, UNOBSERVED, DISABLED_VARIATIONS, VISIBLE_FIELDS, AssetSpec, alternatives, alternate,
    empty_attributes, validate_attributes, relative_variation, COMPARATIVES, register_attributes,
)
from scripts.dialogue_generation.observation_contract import input_schema, observation_prompt
from scripts.dialogue_generation.planner import observation, build_plan, validate_plan
from tools.dialogue_realizer import QwenRealizer

ROOT = Path(__file__).resolve().parents[1] / 'scripts/dialogue_generation'


class AttributeTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((ROOT / 'examples/tokyo_vlog.json').read_text())

    def test_catalog_and_generated_contract_are_in_sync(self):
        self.assertEqual(set(CATEGORIES), {'person', 'object', 'indoor_environment', 'outdoor_environment'})
        schema = input_schema()
        self.assertEqual(schema, json.loads((ROOT / 'observation.schema.json').read_text()))
        self.assertEqual(observation_prompt(), (ROOT / 'observation_prompt.md').read_text())
        self.assertEqual(len(re.findall(r'^Example \d+\. Input', observation_prompt(), re.M)), 3)
        examples = re.findall(r'```json\n(.*?)\n```', observation_prompt(), re.S)
        self.assertEqual(len(examples), 3)
        for example in examples:
            observation(json.loads(example))
        for category in CATEGORIES.values():
            for field, spec in category['attributes'].items():
                self.assertEqual(len(spec['values']), len(set(spec['values'])))
                self.assertTrue(UNOBSERVED <= set(spec['values']))
                self.assertGreater(len(set(spec['values']) - UNOBSERVED), 1)
                if 'ordered_values' in spec:
                    self.assertIn(field, COMPARATIVES)

    def test_missing_fields_fail_but_observed_extensions_are_accepted(self):
        for mode in ('missing','missing_map'):
            source=deepcopy(self.source)
            if mode=='missing': source['assets'][0]['attributes'].pop('hair_length')
            else: source['assets'][0].pop('attributes')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                observation(source)
        source=deepcopy(self.source)
        source['assets'][0]['attributes']['distinctive_motif']='bird-shaped pin'
        source['assets'][0]['attributes']['hair_color']='iridescent silver'
        asset=observation(source)['assets'][0]
        self.assertTrue(validate_attributes(asset))
        self.assertIn('distinctive_motif',asset['attribute_extensions'])
        self.assertIn('iridescent silver',asset['attribute_extensions']['hair_color']['values'])

    def test_every_catalog_field_can_drive_a_real_single_field_variation(self):
        for kind, category in CATEGORIES.items():
            for field, spec in category['attributes'].items():
                if field not in VISIBLE_FIELDS[kind] or field in DISABLED_VARIATIONS.get(kind, ()): continue
                row = dict(id='asset', kind=kind, name='test subject', description='invariant details',
                           attributes=empty_attributes(kind))
                row['attributes'][field] = spec['values'][0]
                before = AssetSpec(row)
                varied = alternate(row, before, random.Random(4))
                with self.subTest(kind=kind, field=field):
                    self.assertEqual(varied.change['field'], field)
                    changed = [key for key in before.attributes if before.attributes[key] != varied.attributes[key]]
                    self.assertEqual(changed, [field])
                    self.assertNotIn(varied.attributes[field], UNOBSERVED)
                    self.assertTrue(validate_attributes(row, varied.attributes))
                    self.assertIn(varied.attributes[field], str(varied))
                    self.assertEqual(row['attributes'], before.attributes)

    def test_relative_edits_are_adjacent_and_return_to_the_given_state(self):
        for kind, category in CATEGORIES.items():
            for field, spec in category['attributes'].items():
                if 'ordered_values' not in spec or field not in VISIBLE_FIELDS[kind] or field in DISABLED_VARIATIONS.get(kind, ()): continue
                row = dict(id='asset', kind=kind, name='test subject', description='invariant details',
                           attributes=empty_attributes(kind))
                row['attributes'][field] = spec['ordered_values'][1]
                variation, earlier, request, question = relative_variation(row, AssetSpec(row), random.Random(2))
                self.assertEqual(variation, field)
                positions = spec['ordered_values']
                self.assertEqual(abs(positions.index(earlier.attributes[field]) - positions.index(row['attributes'][field])), 1)
                self.assertTrue(request and question)

    def test_incompatible_values_and_forbidden_variation_fields_are_filtered(self):
        row = deepcopy(self.source['assets'][-1])
        row['attributes']['transparency'] = 'transparent'
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            validate_attributes(row)
        row['attributes'].update(material='glass', surface_finish='glossy')
        choices = alternatives(row)
        self.assertNotIn('wood', choices.get('material', []))
        self.assertNotIn('shape', choices)
        self.assertNotIn('texture', choices)
        person = deepcopy(self.source['assets'][0])
        person['attributes']['hair_length'] = 'bald'
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            validate_attributes(person)

    def test_identity_extension_can_be_sampled_from_another_known_version(self):
        row = dict(id='bag', kind='object', name='bag', description='A bag.',
                   attributes={**empty_attributes('object'), 'color': 'petrol teal'})
        register_attributes(row, row['attributes'])
        baseline = AssetSpec(row, {**row['attributes'], 'color': 'red'})
        options = alternatives(row, baseline)
        self.assertIn('petrol teal', options['color'])
        self.assertNotIn('petrol teal', CATEGORIES['object']['attributes']['color']['values'])
        self.assertNotIn('red', options['color'])
        for seed in range(40):
            varied = alternate(row, baseline, random.Random(seed))
            if varied.attributes['color'] == 'petrol teal':
                break
        else:
            self.fail('The identity extension was never sampled')
        validate_attributes(row, varied.attributes)

    def test_extensions_do_not_enable_unknown_or_forbidden_fields(self):
        row = dict(id='person', kind='person', name='person', description='A person.',
                   attributes={**empty_attributes('person'), 'hair_style': 'neatly cut', 'clothing_color': 'unknown'},
                   attribute_extensions={'hair_style': dict(values=['unknown', 'neatly cut']),
                                         'clothing_color': dict(values=['unknown', 'petrol teal'])})
        self.assertEqual(alternatives(row), {})
        row['attributes']['clothing_color'] = 'red'
        choices = alternatives(row)
        self.assertNotIn('hair_style', choices)
        self.assertIn('petrol teal', choices['clothing_color'])
        self.assertNotIn('unknown', choices['clothing_color'])

    def test_extension_aliases_do_not_duplicate_visual_choices_or_invent_order(self):
        row = dict(id='scene', kind='outdoor_environment', name='scene', description='A landscape.',
                   attributes={**empty_attributes('outdoor_environment'), 'season': 'winter'},
                   attribute_extensions={'season': dict(values=['unknown', 'autumn'])})
        self.assertEqual(alternatives(row)['season'], ['spring', 'summer', 'fall'])
        person = dict(id='person', kind='person', name='person', description='A person.',
                      attributes={**empty_attributes('person'), 'age_group': 'centenarian'},
                      attribute_extensions={'age_group': dict(values=['unknown', 'centenarian'])})
        self.assertNotIn('age_group', alternatives(person, relative=True))

    def test_unknown_attributes_are_never_invented_for_sampling(self):
        source = deepcopy(self.source)
        source['constraints'] = []
        for row in source['assets']:
            row['attributes'] = empty_attributes(row['kind'])
        result = build_plan(source, seed=4, reworks=2, modules=['relative_preview'])
        self.assertTrue(validate_plan(result))
        self.assertEqual({e['module'] for e in result['events']}, {'direct'})
        self.assertTrue(all(not m.get('variation') for m in result['placeholders']))

    def test_final_state_checks_attributes_not_only_a_goal_flag(self):
        plan = build_plan(self.source, seed=5)
        asset = next(m for m in plan['placeholders'] if m['operation'] == 'create_reference' and m['goal_match'])
        row = plan['target_assets'][asset['asset_ids'][0]]
        edited = alternate(row, AssetSpec(row, asset['attributes']), random.Random(0))
        asset['attributes'] = edited.attributes
        with self.assertRaisesRegex(ValueError, 'different attributes'):
            validate_plan(plan)

    def test_rework_can_ask_to_update_two_existing_shots_or_only_one(self):
        source = deepcopy(self.source)
        source['constraints'] = []
        for row in source['assets']:
            attrs = empty_attributes(row['kind'])
            if row['id'] == 'bag': attrs['color'] = 'blue'
            row['attributes'] = attrs
        scopes, multi_shot = set(), []
        for seed in range(60):
            plan = build_plan(source, seed=seed, modules=['direct'], reworks=1, p_compile=1, cross_turn_rules=False)
            event = next(e for e in plan['events'] if e['module'] == 'later_revision')
            request = next(t for t in plan['turns'] if t['intent']['action'] == 'request_rework')
            scopes.update(t['intent']['scope'] for t in plan['turns'] if t['intent']['action'] == 'choose_update_scope')
            self.assertIn('main object color to blue', request['intent']['description'])
            for shot in event['rendered_uses']:
                self.assertTrue(any(t['intent']['action'] == 'deliver_shot' and shot in t['intent']['shot_ids']
                                    for t in plan['turns'][:request['id'] - 1]))
            if len(event['rendered_uses']) == 2:
                multi_shot.append(plan)
                question = next(t for t in plan['turns'] if t['intent']['action'] == 'clarify_update_scope')
                self.assertIn('same tote bag', question['intent']['description'])
                self.assertIn('all existing shots', question['intent']['description'])
        self.assertTrue(multi_shot)
        self.assertEqual(scopes, {'shot', 'all_uses'})

    def test_locked_scene_attributes_cannot_leak_into_formal_shots(self):
        for seed in range(20):
            plan = build_plan(self.source, seed=seed, reworks=2, p_compile=1)
            table = {m['placeholder']: m for m in plan['placeholders']}
            for shot in (m for m in table.values() if m['operation'] == 'generate_clip'):
                for asset, token in shot['bindings'].items():
                    if asset in ('street', 'cafe'):
                        self.assertEqual(table[token]['attributes']['time_of_day'], 'night')

    def test_whole_environment_requirement_is_sent_to_semantic_verifier(self):
        from unittest.mock import Mock
        from tools.dialogue_verifier import CRITERIA
        model = QwenRealizer(asset_names={'cafe': 'coffee shop'})
        report = dict(dict.fromkeys(CRITERIA, 2), critique="")
        model.verifier = Mock(return_value=report)
        turn = dict(id=1, role='user', intent=dict(action='reject_and_edit',
                    description='Make the entire coffee shop tidy.', asset_ids=['cafe'],
                    asset_types={'cafe': 'indoor_environment'}))
        with patch.object(model, '_request', return_value='Tidy the entire coffee shop.'):
            self.assertEqual(model(turn, {}), 'Tidy the entire coffee shop.')
        brief = model.verifier.call_args.args[1]
        self.assertIn('Make the entire coffee shop tidy.', brief['draft_content_requirements'])


if __name__ == '__main__':
    unittest.main()
