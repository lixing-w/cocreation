import copy
import io
import os
from contextlib import redirect_stderr
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cocreation import Qwen, Reference
from cocreation.worker import qwen_messages
from tools.extract_attribute_nodes import extract_attribute_nodes, parse_attributes, validate_attributes
from tools.extract_attribute_nodes_prompts import (
    VIDEO_EXAMPLE, IMAGE_EXAMPLE, DEFINITIONS, CATEGORY_RULES, extraction_system_prompt)


class ExtractionTests(unittest.TestCase):
    def test_examples_match_contract(self):
        for example in (VIDEO_EXAMPLE, IMAGE_EXAMPLE):
            self.assertEqual(validate_attributes(example, example['artifact_id'], example['kind']), example)
        self.assertNotIn('Fictional video example', DEFINITIONS)
        self.assertNotIn('Fictional image example', DEFINITIONS)
        self.assertIn('identity.description', DEFINITIONS)
        self.assertEqual(len(__import__('tools.extract_attribute_nodes_prompts', fromlist=['TOP_LEVEL_CATEGORIES']).TOP_LEVEL_CATEGORIES), 9)
        prompt=extraction_system_prompt()
        self.assertLess(prompt.index('## Definitions'), prompt.index('Video JSON example:'))
        self.assertLess(prompt.index('Video JSON example:'), prompt.index('Image JSON example:'))

    def test_invalid_contracts_return_none(self):
        mutations = [
            lambda d: d.update(artifact_id='wrong'),
            lambda d: d['attributes'].update(unknown=[]),
            lambda d: d['attributes'].update(style=None),
            lambda d: d['attributes'].update(style=[]),
            lambda d: d['attributes'].update(style='word '*16),
            lambda d: d['attributes'].update(style='unknown'),
            lambda d: d['attributes'].update(viewpoint_change=''),
            lambda d: d['attributes'].update({'identity.appearance': ['red']}),
            lambda d: d['attributes']['identity'][0].update(appearance=[]),
            lambda d: d['attributes']['identity'][0].update(appearance=[{'description':'red'}]),
            lambda d: d['attributes']['identity'][0].update(pose_change=None),
            lambda d: d['attributes']['identity'][0].update(parent_id='missing'),
            lambda d: d['attributes']['identity'][0].update(time_range=[0,2]),
            lambda d: d['attributes']['identity'][0].pop('uid'),
            lambda d: d['attributes']['identity'][0].pop('description'),
            lambda d: d['attributes']['environment'][0].update(uid='woman0_id'),
            lambda d: d['attributes']['environment'][0].update(weather=['snow']),
            lambda d: d['attributes']['environment'][0].update(space_type='indoor'),
            lambda d: d['attributes']['visible_text'][0].update(links=['woman0_id']),
            lambda d: d['attributes']['viewpoint'][0].update(viewpoint_motion='Pushes closer.'),
            lambda d: d['attributes']['identity_relations'][0].update(links=['woman0_id']*2),
            lambda d: d['attributes']['identity_relations'][0].update(links=['woman0_id','missing']),
            lambda d: d['attributes']['identity_relations'][0].update(links=['woman0_id']),
            lambda d: d['attributes']['identity_relations'][0].pop('links'),
            lambda d: d['attributes']['identity_background_relations'][0].update(links=['office0_env','woman0_id']),
        ]
        for mutate in mutations:
            d=copy.deepcopy(VIDEO_EXAMPLE); mutate(d)
            self.assertIsNone(parse_attributes(json.dumps(d),'example_video','video'))
        for text in ('bad', '{}', 'null', '{"x":1,"x":2}', json.dumps(VIDEO_EXAMPLE)[:-1]):
            self.assertIsNone(parse_attributes(text,'example_video','video'))
        d=copy.deepcopy(VIDEO_EXAMPLE);d['kind']='image'
        self.assertIsNone(parse_attributes(json.dumps(d),'example_video','image'))
        self.assertEqual(parse_attributes('```json\n'+json.dumps(IMAGE_EXAMPLE)+'\n```',
                                         'example_image','image'),IMAGE_EXAMPLE)

    def test_sparse_categories_and_optional_metadata(self):
        d=copy.deepcopy(IMAGE_EXAMPLE)
        d['attributes'].pop('style')
        self.assertEqual(parse_attributes(json.dumps(d),'example_image','image'),d)
        d['attributes']={}
        self.assertEqual(parse_attributes(json.dumps(d),'example_image','image'),d)
        for example in (IMAGE_EXAMPLE,VIDEO_EXAMPLE):
            self.assertNotIn(': null',json.dumps(example))
            self.assertNotIn(': []',json.dumps(example))
        prompt=extraction_system_prompt()
        self.assertNotIn('"time_range":null',prompt)
        self.assertNotIn('contains EVERY',prompt)

    def test_owned_and_shared_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.png';path.touch()
            ref=Reference('example_image',path,'image')
            model=Mock(return_value={'text':json.dumps(IMAGE_EXAMPLE)})
            self.assertEqual(extract_attribute_nodes(ref,qwen=model),IMAGE_EXAMPLE)
            model.close.assert_not_called()
            self.assertIn('system_prompt',model.call_args.kwargs)
            self.assertTrue(str(model.call_args.kwargs['system_cache_path']).endswith(
                'kvcache/extract_attribute_nodes/system_prompt.kvcache'))
            extract_attribute_nodes(ref,qwen=model,use_system_cache=False)
            self.assertIsNone(model.call_args.kwargs['system_cache_path'])
            self.assertEqual(model.call_args.args[1],[Reference(ref.artifact_id,path.resolve(),'image')])
            with patch('tools.extract_attribute_nodes.Qwen', return_value=model) as factory:
                self.assertEqual(extract_attribute_nodes(ref),IMAGE_EXAMPLE)
                factory.assert_called_once();model.close.assert_called_once()
            model.reset_mock();model.return_value={'text':'invalid'}
            with patch('tools.extract_attribute_nodes.Qwen',return_value=model):
                self.assertIsNone(extract_attribute_nodes(ref))
                model.close.assert_called_once()
            model.reset_mock();model.side_effect=RuntimeError('inference failed')
            with patch('tools.extract_attribute_nodes.Qwen',return_value=model):
                with self.assertRaises(RuntimeError):extract_attribute_nodes(ref)
                model.close.assert_called_once()

    def test_system_prompt_is_per_call_and_plain_text(self):
        model=Qwen()
        one=model('user',system_prompt='Rules <Image 1>',dry_run=True)
        two=model('next',dry_run=True)
        self.assertEqual(qwen_messages(one)[0],{'role':'system','content':'Rules <Image 1>'})
        self.assertEqual([m['role'] for m in qwen_messages(one)],['system','user'])
        self.assertEqual([m['role'] for m in qwen_messages(two)],['user'])
        with self.assertRaises(TypeError):model('user',system_prompt=[],dry_run=True)
        with self.assertRaises(ValueError):model('user',system_cache_path='/tmp/cache',dry_run=True)
        request=model('user',system_prompt='Rules',system_cache_path='/tmp/cache',dry_run=True)
        self.assertEqual(request['options']['system_cache_path'],str(Path('/tmp/cache').resolve()))

    def test_error_messages_identify_location(self):
        bad=copy.deepcopy(IMAGE_EXAMPLE)
        bad['attributes']['identity'][0]['appearance']=[None]
        for value,expected in [(json.dumps(bad),"$.attributes['identity'][0].appearance[0]"),
                               ('{\n"x":}', 'line 2, column'),
                               ('{"x":1,"x":2}', "Duplicate JSON key 'x'"),
                               (None, 'response text is missing')]:
            stream=io.StringIO()
            with redirect_stderr(stream):
                self.assertIsNone(parse_attributes(value,'example_image','image'))
            self.assertIn(expected,stream.getvalue())

    def test_output_root_owned_and_shared(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'COCREATION_SCRATCH':tmp}):
            path=Path(tmp)/'image.png';path.touch()
            ref=Reference('example_image',path,'image')
            model=Mock(return_value={'text':json.dumps(IMAGE_EXAMPLE)})
            default=Path(tmp).resolve()/'runs/extract_attribute_nodes'
            with patch('tools.extract_attribute_nodes.Qwen',return_value=model) as factory:
                extract_attribute_nodes(ref)
                factory.assert_called_once_with(output_dir=default)
                self.assertEqual(model.call_args.kwargs['output_dir'],default)
            extract_attribute_nodes(ref,qwen=model)
            self.assertEqual(model.call_args.kwargs['output_dir'],default)
            custom=Path(tmp).resolve()/'custom'
            extract_attribute_nodes(ref,qwen=model,output_dir=custom)
            self.assertEqual(model.call_args.kwargs['output_dir'],custom)

    def test_unique_uids_and_definition_snapshot(self):
        from tools.extract_attribute_nodes_prompts import OBJECT_FIELDS
        for example in (VIDEO_EXAMPLE, IMAGE_EXAMPLE):
            entries = [entry for category, group in example['attributes'].items()
                       if category in OBJECT_FIELDS for entry in group]
            uids = [entry['uid'] for entry in entries]
            self.assertEqual(len(uids), len(set(uids)))
        docs = (Path(__file__).resolve().parents[1]/'docs/cpg.md').read_text()
        self.assertIn(DEFINITIONS.strip(), docs)

    def test_each_temporal_field_rejected_for_images(self):
        for key, rule in CATEGORY_RULES.items():
            if rule['temporal'] != 'Yes':
                continue
            d = copy.deepcopy(IMAGE_EXAMPLE)
            if '.' in key:
                category, field = key.split('.')
                d['attributes'][category][0][field] = ['Moves visibly.'] if rule['multiple'] else 'Changes visibly.'
            else:
                d['attributes'][key] = 'Changes visibly.'
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_attributes(d, 'example_image', 'image')

    def test_minimal_objects_and_forward_relations(self):
        d = copy.deepcopy(IMAGE_EXAMPLE)
        for entry in d['attributes']['identity']:
            entry.pop('appearance')
        d['attributes'] = dict(reversed(list(d['attributes'].items())))
        self.assertIs(validate_attributes(d, 'example_image', 'image'), d)
