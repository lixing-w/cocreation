"""Prompt provenance, deterministic cuts, plain attributes and pipeline compatibility."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.dialogue_generation.planner import build_plan, validate_plan
from scripts.dialogue_generation.asset_variations import CATEGORIES, empty_attributes
from tools.prompt_observer import (PromptObserver, attributes_schema, parse_attributes, load_record,
                                  prepare_source, validate_inventory)
from tools.prompt_observer_prompts import SYSTEM, attributes_prompt, inventory_prompt
from tools.video_observer_prompts import SYSTEM as VIDEO_SYSTEM
from tools.video_observer import VideoObserver


H3 = dict(prompt='subject_definitions: A woman wears a blue coat.\n\n'
    'summary: [Shot 1] Reference summary. [Shot 2] Reference summary.\n\n'
    'retention_analysis: Her blue coat changes to red.\n\n'
    'detailed_description: Cinematic. [Shot 1] A woman in a red coat stands. '
    '[Shot 2] At 00:02.000, she walks.\n\n'
    'overall_soundscape: Loud thunder.\n\nnon_diegetic_music: Jazz.',
    num_frames=124, width=1344, height=768, video_path='ref2va_videos/example.mp4')


def inventory():
    return dict(assets=[dict(id='A1', kind='person', name='woman',
                            description='A woman in a red coat.')],
                shots=[dict(number=1, description='A woman stands.', assets=['A1']),
                       dict(number=2, description='She walks.', assets=['A1'])])


def attribute_reply(known=None, kind='person'):
    values = {key: 'unknown' for key in CATEGORIES[kind]['attributes']}
    values.update(known or {})
    return values


def separated_inventory():
    return dict(assets=[dict(id='A1',description='Person')],
                shots=[dict(id=f'S{i+1}',description=row['description']) for i,row in enumerate(inventory()['shots'])])


class SourceTests(unittest.TestCase):
    def test_ref_target_overrides_reference_and_does_not_count_summary_markers(self):
        source = prepare_source(H3, fps=24)
        self.assertEqual(len(source['shots']), 2)
        self.assertEqual(source['shots'][1]['start'], 2)
        self.assertAlmostEqual(source['shots'][-1]['end'], 124/24)
        self.assertNotIn('blue coat', source['visual_text'])
        self.assertNotIn('thunder', source['visual_text'])
        self.assertEqual(source['geometry'], dict(width=1344, height=768))

    def test_openvid_actions_and_camera_moves_do_not_invent_cuts(self):
        source = prepare_source(dict(caption='A man walks, then runs. The camera pans.', seconds='6.1'))
        self.assertEqual(source['format'], 'openvid')
        self.assertEqual(len(source['shots']), 1)
        self.assertEqual(source['geometry'], {})
        self.assertEqual(source['shots'][0]['end'], 6.1)

    def test_base_h3_and_explicit_duration_override(self):
        source = prepare_source(dict(prompt='integrated_multimodal_description: [Shot 1] A dog.', seconds=5), duration=7)
        self.assertEqual(source['shots'][0]['end'], 7)
        self.assertEqual(source['duration_basis'], 'override')

    def test_rejects_unknown_duration_and_does_not_guess_h3_fps(self):
        for record in ('A person stands.', H3):
            with self.subTest(record=type(record)), self.assertRaisesRegex(ValueError, 'duration'):
                prepare_source(record)
        self.assertEqual(prepare_source(dict(caption='A dog.', frame='90', fps='30'))['duration'], 3)

    def test_rejects_missing_repeated_out_of_order_or_out_of_bounds_cuts(self):
        descriptions = [
            '[Shot 1] A woman. [Shot 2] She walks.',
            '[Shot 1] A woman. [Shot 1] She walks.',
            '[Shot 2] At 00:01.000, a woman.',
            '[Shot 1] A woman. [Shot 2] At 00:05.000, she walks.',
            '[Shot 1] A woman. [Shot 2] At 00:02.000, she walks. [Shot 3] At 00:01.000, she runs.',
            ' '.join(f'[Shot {i}] At 00:0{i-1}.000, a woman.' for i in range(1, 6)),
        ]
        for body in descriptions:
            with self.subTest(body=body), self.assertRaises(ValueError):
                prepare_source('integrated_multimodal_description: '+body, duration=5)

    def test_invalid_metadata_fails_before_a_model_call(self):
        for value in (True, float('nan'), 0, -1, 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_source('A dog.', duration=value)
        with self.assertRaises(ValueError):
            prepare_source('A dog.', duration=5, width=1.5)

    def test_reads_real_dataset_container_formats_and_zero_based_rows(self):
        rows = [dict(caption='A cat.', seconds='4'), dict(caption='A dog.', seconds='5')]
        with tempfile.TemporaryDirectory() as tmp:
            for ext, content in (
                ('json', json.dumps(rows)),
                ('jsonl', '\n'.join(map(json.dumps, rows))),
                ('csv', 'caption,seconds\nA cat.,4\nA dog.,5\n'),
            ):
                path = Path(tmp)/('data.'+ext)
                path.write_text(content)
                self.assertEqual(load_record(path, 1), rows[1])
                with self.assertRaisesRegex(ValueError, 'No row'):
                    load_record(path, 2)


class ObservationTests(unittest.TestCase):
    def test_plain_values_allow_new_vocabulary_in_existing_fields(self):
        asset=dict(kind='person')
        values=parse_attributes(asset,dict(clothing_color='peacock blue',clothing_type='linen wrap coat'))
        self.assertEqual(values['clothing_color'],'peacock blue')
        self.assertEqual(values['hair_color'],'unknown')
        self.assertIn('peacock blue',asset['attribute_extensions']['clothing_color']['values'])
        self.assertEqual(set(values),set(CATEGORIES['person']['attributes']))

    def test_inventory_accepts_natural_paraphrases(self):
        value=inventory()
        value['assets'][0]['description']='The central person in the scene.'
        self.assertEqual(validate_inventory(value,prepare_source(H3,fps=24),8),value)

    def test_end_to_end_uses_plain_attributes_and_remains_planner_compatible(self):
        qwen=Mock(model_id='test-model',side_effect=[dict(text=json.dumps(v),finish_reason='stop') for v in (
            inventory(),attribute_reply(dict(clothing_color='red')),
            attribute_reply(dict(clothing_color='red')),separated_inventory(),
            dict(attribute_separation=2,action_preservation=2,detail_preservation=2,correction=''))])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            result=PromptObserver(qwen=qwen).observe(H3,path,fps=24,progress=False)
            self.assertEqual(result['assets'][0]['attributes']['clothing_color'],'red')
            self.assertEqual(result['target_order'],['S1','S2'])
            self.assertEqual(json.loads((path/'asset_01/attributes.json').read_text())['clothing_color'],'red')
            validate_plan(build_plan(deepcopy(result),seed=1))
            for call in qwen.call_args_list:
                self.assertEqual(call.args[1],[])
        self.assertEqual(VideoObserver.system_prompt,VIDEO_SYSTEM)

    def test_observation_rules_have_the_same_wire_prefix_for_different_assets(self):
        qwen = Mock(model_id='test-model', return_value=dict(text='{}', finish_reason='stop'))
        observer = PromptObserver(qwen=qwen)
        with tempfile.TemporaryDirectory() as tmp:
            for index, name in enumerate(('cup', 'ball')):
                asset = dict(name=name, kind='object', description=f'The {name} is green.')
                prompt = attributes_prompt(asset, f'The {name} falls.')
                observer._ask(prompt, [], {}, lambda value: value, Path(tmp)/str(index))
        first, second = qwen.call_args_list
        self.assertEqual(first.kwargs['system_prompt'], second.kwargs['system_prompt'])
        self.assertIn('Fields and suggested vocabulary', first.kwargs['system_prompt'])
        self.assertIn('The cup falls.', first.args[0])
        self.assertNotIn('The cup falls.', str(first.kwargs))
        self.assertIn('The ball falls.', second.args[0])
        one = inventory_prompt(dict(common='', shots=[{}]), 3)
        two = inventory_prompt(dict(common='', shots=[{}, {}]), 8)
        self.assertEqual(one.instructions, two.instructions)
        self.assertIn('Shot count: 2', two.data)

    def test_review_can_recover_values_and_discard_a_first_pass_addition(self):
        proposed=attribute_reply(dict(clothing_color='peacock blue',hair_style='invented hairstyle'))
        reviewed=attribute_reply(dict(clothing_color='peacock blue',clothing_type='linen wrap coat'))
        qwen=Mock(model_id='test-model',side_effect=[dict(text=json.dumps(v)) for v in (
            inventory(),proposed,reviewed,separated_inventory(),
            dict(attribute_separation=2,action_preservation=2,detail_preservation=2,correction=''))])
        with tempfile.TemporaryDirectory() as tmp:
            result=PromptObserver(qwen=qwen).observe(H3,Path(tmp)/'run',fps=24,progress=False)
        asset=result['assets'][0]
        self.assertEqual(asset['attributes']['clothing_color'],'peacock blue')
        self.assertEqual(asset['attributes']['hair_style'],'unknown')
        self.assertNotIn('hair_style',asset['attribute_extensions'])
        self.assertIn('linen wrap coat',asset['attribute_extensions']['clothing_type']['values'])


if __name__=='__main__':
    unittest.main()


