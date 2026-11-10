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
from tools.extract_attribute_nodes import extract_attribute_nodes, parse_attributes, validate_attributes, repair_attributes
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
            lambda d: d['attributes'].update({'identity.appearance': ['red']}),
            lambda d: d['attributes']['identity'][0].update(appearance=[{'description':'red'}]),
            lambda d: d['attributes']['identity'][0].update(parent_id='missing'),
            lambda d: d['attributes']['identity'][0].update(time_range=[0,2]),
            lambda d: d['attributes']['identity'][0].pop('uid'),
            lambda d: d['attributes']['environment'][0].update(uid='woman0_id'),
            lambda d: d['attributes']['environment'][0].update(space_type='indoor'),
            lambda d: d['attributes']['visible_text'][0].update(links=['woman0_id']),
            lambda d: d['attributes']['viewpoint'][0].update(viewpoint_motion='Pushes closer.'),
            lambda d: d['attributes']['identity_relations'][0].update(links=['woman0_id']*2),
            lambda d: d['attributes']['identity_relations'][0].update(links=['woman0_id','missing']),
            lambda d: d['attributes']['identity_relations'][0].update(links=['woman0_id']),
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
        bad['attributes']['identity'][0]['appearance']=[42]
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


class RepairTests(unittest.TestCase):
    def test_optional_placeholders_and_empty_items_repaired_in_place(self):
        value=copy.deepcopy(IMAGE_EXAMPLE)
        attrs=value['attributes']
        attrs.update(style=' UNKNOWN ', environment_change=None, unused=[], viewpoint_change='null')
        subject=attrs['identity'][0]
        subject.update(expression=['none',None,'  ',[],{}], appearance=[' Blue glaze. ',None,'NULL'])
        subject['body_pose']='Upright.'
        self.assertIs(repair_attributes(value),value)
        self.assertEqual(subject['appearance'],['Blue glaze.'])
        self.assertEqual(subject['body_pose'],['Upright.'])
        self.assertNotIn('expression',subject)
        for key in ('style','environment_change','unused','viewpoint_change'):
            self.assertNotIn(key,attrs)
        self.assertIs(validate_attributes(value,'example_image','image'),value)
        snapshot=copy.deepcopy(value)
        self.assertEqual(repair_attributes(value),snapshot)

    def test_scalar_wrapping_and_long_descriptions_not_truncated(self):
        value=copy.deepcopy(IMAGE_EXAMPLE)
        value['attributes']['environment'][0]['lighting']=[None,' Soft window light. ','unknown']
        value['attributes']['style']=['  Watercolor. ']
        value['attributes']['identity'][0]['description']='word '*30
        result=parse_attributes(json.dumps(value),'example_image','image')
        self.assertEqual(result['attributes']['style'],'Watercolor.')
        self.assertEqual(result['attributes']['environment'][0]['lighting'],'Soft window light.')
        self.assertEqual(len(result['attributes']['identity'][0]['description'].split()),30)

    def test_missing_description_prunes_object_and_dependent_relations(self):
        value=copy.deepcopy(IMAGE_EXAMPLE)
        removed=value['attributes']['identity'][0]['uid']
        value['attributes']['identity'][0]['description']=None
        result=parse_attributes(json.dumps(value),'example_image','image')
        self.assertIsNotNone(result)
        self.assertNotIn(removed,[e['uid'] for e in result['attributes']['identity']])
        for category in ('identity_relations','identity_background_relations'):
            self.assertTrue(all(removed not in e['links'] for e in result['attributes'].get(category,[])))

    def test_placeholder_relation_endpoint_drops_whole_relation(self):
        value=copy.deepcopy(IMAGE_EXAMPLE)
        value['attributes']['identity_relations'][0]['links'][1]='unknown'
        result=parse_attributes(json.dumps(value),'example_image','image')
        self.assertIsNotNone(result)
        self.assertNotIn('identity_relations',result['attributes'])

    def test_missing_links_removes_only_that_relation(self):
        value=copy.deepcopy(IMAGE_EXAMPLE)
        value['attributes']['identity_relations'][0].pop('links')
        result=parse_attributes(json.dumps(value),'example_image','image')
        self.assertNotIn('identity_relations',result['attributes'])
        self.assertEqual(result['attributes']['identity'],IMAGE_EXAMPLE['attributes']['identity'])
        self.assertEqual(result['attributes']['identity_background_relations'],
                         IMAGE_EXAMPLE['attributes']['identity_background_relations'])

    def test_pruned_viewpoint_removes_timeline_without_inventing_transition(self):
        value=copy.deepcopy(VIDEO_EXAMPLE)
        value['attributes']['viewpoint'][1]['description']='None'
        result=parse_attributes(json.dumps(value),'example_video','video')
        self.assertEqual(len(result['attributes']['viewpoint']),2)
        self.assertNotIn('viewpoint_change',result['attributes'])
        value['attributes']['viewpoint']=[]
        result=parse_attributes(json.dumps(value),'example_video','video')
        self.assertNotIn('viewpoint_change',result['attributes'])

    def test_no_guessing_metadata_unknown_fields_or_multiple_scalar_alternatives(self):
        for mutate in (
            lambda d:d.update(kind=None),
            lambda d:d['attributes'].update(mystery='Visible feature.'),
            lambda d:d['attributes']['environment'][0].update(lighting=['Warm.','Cold.']),
            lambda d:d['attributes']['identity_relations'][0].update(links=['cup0_id','missing']),
        ):
            value=copy.deepcopy(IMAGE_EXAMPLE);mutate(value)
            self.assertIsNone(parse_attributes(json.dumps(value),'example_image','image'))
        self.assertEqual(parse_attributes('{"artifact_id":"empty","kind":"image","attributes":null}',
                                         'empty','image')['attributes'],{})

    def test_strict_validator_does_not_repair_existing_observations(self):
        value=copy.deepcopy(IMAGE_EXAMPLE)
        value['attributes']['style']=None
        before=copy.deepcopy(value)
        with self.assertRaises(ValueError): validate_attributes(value,'example_image','image')
        self.assertEqual(value,before)

    def test_viewpoint_examples_include_continuous_and_cut_transitions(self):
        from tools.attribute_verifier_prompts import REVISE, INHERIT, VIEWPOINT_EXAMPLE
        for example in (VIDEO_EXAMPLE['attributes'],VIEWPOINT_EXAMPLE):
            self.assertEqual(len(example['viewpoint']),3)
            self.assertIn('Continuous push-in',example['viewpoint_change'])
            self.assertIn('hard cut',example['viewpoint_change'])
            for stage in example['viewpoint']:
                self.assertIn(stage['uid'],example['viewpoint_change'])
        for prompt in (extraction_system_prompt(),REVISE,INHERIT):
            self.assertIn('view0_vp',prompt)
            self.assertIn('view2_vp',prompt)


class DraftTests(unittest.TestCase):
    def test_draft_removed_before_validation_and_not_promoted(self):
        from tools.attribute_observation_prompts import DRAFT_KEY
        value=copy.deepcopy(IMAGE_EXAMPLE)
        value['attributes']={DRAFT_KEY:{'identity':[{'uid':'invented','description':'Unused draft object.'}]},
                             **value['attributes']}
        result=parse_attributes(json.dumps(value),'example_image','image')
        self.assertEqual(result,IMAGE_EXAMPLE)
        self.assertNotIn(DRAFT_KEY,result['attributes'])
        for draft in (None,'freeform',[],{'arbitrary':'data'}):
            value=copy.deepcopy(IMAGE_EXAMPLE)
            value['attributes'][DRAFT_KEY]=draft
            self.assertEqual(repair_attributes(value),IMAGE_EXAMPLE)

    def test_prompt_examples_draft_first_and_short_descriptions(self):
        from tools.attribute_observation_prompts import DRAFT_KEY, with_draft
        for value in (IMAGE_EXAMPLE,VIDEO_EXAMPLE):
            example=with_draft(value)
            self.assertEqual(next(iter(example['attributes'])),DRAFT_KEY)
            self.assertNotIn(DRAFT_KEY,value['attributes'])
            self.assertEqual(repair_attributes(example),value)
        prompt=extraction_system_prompt()
        # Full human-facing documentation is intentionally not sent to the model.
        self.assertNotIn('### Conservative observation repair',prompt)
        self.assertIn('< 10 words',prompt)
        self.assertIn(DRAFT_KEY,prompt)
