"""Prompt writers: source/target mapping, strict outputs, lifecycle and skill snapshots."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cocreation import Qwen, Reference, VDNH3, Flux
from cocreation.worker import qwen_messages
from tools._prompter import prepare_inputs, parse_prompt
from tools.flux_prompter import flux_prompter
from tools.h3_prompter import h3_prompter
from tools.flux_prompter_prompts import flux_system_prompt
from tools.h3_prompter_prompts import h3_system_prompt

REF_PROMPT = '''subject_definitions:
<Subject 1> is the red coat from <Video 1>.
<Subject 2> is the garden from <Picture 1>.
summary:
A five-second portrait in the garden.
retention_analysis:
<Subject 1> transfers the coat color; <Subject 2> transfers the garden setting.
detailed_description:
[Shot 1] A woman wearing <Subject 1> stands in <Subject 2> under morning light.
[Shot 2] At 00:02.000, a close view shows her gently turning toward the camera.
overall_soundscape:
N/A
non_diegetic_music:
N/A'''
BASE_PROMPT = '''integrated_multimodal_description:
[Shot 1] An abstract violet circle slowly brightens during five seconds.
overall_soundscape:
N/A
non_diegetic_music:
N/A'''
FLUX_PROMPT = 'A seated portrait wearing the red coat from <Image 1> (image 1), in the garden from <Image 2> (image 2), soft morning light.'


def response(prompt):
    return {'text':json.dumps({'prompt':prompt})}


class Prompters(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        for name in ('video.mp4','garden.png'):(self.root/name).touch()
        self.refs=[Reference('video',self.root/'video.mp4','video'),
                   Reference('garden',self.root/'garden.png','image')]
        self.attrs=[dict(id='video_coat',category='identity.appearance',text='Red coat.',artifact_ids=['video']),
                    dict(id='garden_environment',category='environment',text='Rose garden.',artifact_ids=['garden']),
                    dict(id='new_light',category='environment.lighting',text='Soft morning light.')]

    def test_direct_video_observation_and_flux_target_slots(self):
        model=Mock(return_value=response(FLUX_PROMPT))
        original=copy.deepcopy(self.attrs)
        self.assertEqual(flux_prompter(self.attrs,self.refs,qwen=model),FLUX_PROMPT)
        user,refs=model.call_args.args;kwargs=model.call_args.kwargs
        self.assertEqual([r.kind for r in refs],['video','image'])
        self.assertIn('"generator_label":"<Image 1>"',user)
        self.assertIn('"generator_label":"<Image 2>"',user)
        self.assertIn('"artifact_ids":[]',user)
        self.assertTrue(str(kwargs['system_cache_path']).endswith('/flux_prompter/system_prompt.kvcache'))
        planned=Qwen()(user,refs,system_prompt=kwargs['system_prompt'],dry_run=True)
        content=qwen_messages(planned)[1]['content']
        self.assertEqual([x['type'] for x in content if x['type']!='text'],['video','image'])
        model.close.assert_not_called();self.assertEqual(self.attrs,original)
        # Generator accepts final all-image list in those same slots.
        final=[Reference('video',self.root/'garden.png','image'),self.refs[1]]
        self.assertEqual(Flux()(FLUX_PROMPT,final,dry_run=True)['references'][1]['label'],'<Image 2>')

    def test_h3_mapping_and_sections_match_generator(self):
        model=Mock(return_value=response(REF_PROMPT))
        self.assertEqual(h3_prompter(self.attrs,self.refs,qwen=model,seconds=5),REF_PROMPT)
        user,refs=model.call_args.args;kwargs=model.call_args.kwargs
        self.assertIn('"target_seconds":5',user)
        self.assertIn('"generator_label":"<Picture 1>"',user)
        self.assertTrue(str(kwargs['system_cache_path']).endswith('/h3_prompter/system_prompt.kvcache'))
        self.assertEqual([r['label'] for r in VDNH3()(REF_PROMPT,refs,dry_run=True)['references']],
                         ['<Video 1>','<Picture 1>'])
        model.close.assert_not_called()

    def test_source_free_new_attributes(self):
        attrs=[dict(id='new_circle',category='environment',text='An abstract violet circle.')]
        for tool,prompt in [(flux_prompter,'An abstract violet circle.'),(h3_prompter,BASE_PROMPT)]:
            model=Mock(return_value=response(prompt))
            self.assertEqual(tool(attrs,qwen=model,use_system_cache=False),prompt)
            self.assertEqual(model.call_args.args[1],[])
            self.assertIsNone(model.call_args.kwargs['system_cache_path'])

    def test_bad_associations_fail_before_model(self):
        model=Mock()
        cases=[(self.attrs[:1],self.refs),
               (self.attrs,self.refs[:1]),
               (self.attrs,self.refs+self.refs[:1]),
               ([],[]),
               ([dict(id='x',category='style',text='Oil paint.',artifact_ids=['missing'])],[])]
        for attrs,refs in cases:
            with self.assertRaises(ValueError):flux_prompter(attrs,refs,qwen=model)
        for duration in (0,16,True,float('nan')):
            with self.assertRaises(ValueError):h3_prompter(self.attrs,self.refs,qwen=model,seconds=duration)
        with self.assertRaises(ValueError):
            flux_prompter(self.attrs,[Reference('audio',self.root/'garden.png','audio')],qwen=model)
        model.assert_not_called()

    def test_owned_lifecycle_success_null_and_failure(self):
        for target,tool,good in [('flux',flux_prompter,FLUX_PROMPT),('h3',h3_prompter,REF_PROMPT)]:
            for output in (response(good),response(None),{'text':'truncated'}):
                model=Mock(return_value=output)
                with patch('tools.'+target+'_prompter.Qwen',return_value=model) as factory:
                    result=tool(self.attrs,self.refs)
                    self.assertEqual(result,good if output==response(good) else None)
                    factory.assert_called_once();model.close.assert_called_once()
            model=Mock(side_effect=RuntimeError('inference'))
            with patch('tools.'+target+'_prompter.Qwen',return_value=model):
                with self.assertRaises(RuntimeError):tool(self.attrs,self.refs)
                model.close.assert_called_once()

    def test_output_label_and_timeline_validation(self):
        _,_,flux=prepare_inputs(self.attrs,self.refs,'flux')
        _,_,h3=prepare_inputs(self.attrs,self.refs,'h3')
        for bad in (FLUX_PROMPT.replace('<Image 1>','<Video 1>'),
                    FLUX_PROMPT.replace('<Image 2>','<Image 9>'),
                    FLUX_PROMPT.replace('<Image 2>',''), '```'+FLUX_PROMPT+'```'):
            self.assertIsNone(parse_prompt(response(bad),flux,'flux'))
        for bad in (REF_PROMPT.replace('<Picture 1>','<Image 1>'),
                    REF_PROMPT.replace('00:02.000','00:06.000'),
                    REF_PROMPT.replace('[Shot 2]','[Shot 3]'),
                    REF_PROMPT.replace('<Subject 1> is','<Subject 3> is'),
                    REF_PROMPT.replace('summary:','unknown_section:'),
                    REF_PROMPT+'\n<Audio 1>'):
            self.assertIsNone(parse_prompt(response(bad),h3,'h3',5))
        self.assertIsNone(parse_prompt({'text':'{"prompt":"a","prompt":"b"}'},[],'flux'))

    def test_official_files_intact_and_static_system_prefixes(self):
        root=Path(__file__).resolve().parents[1]/'skills'
        for name in ('flux-image-best-practices','h3-prompt-writing'):
            manifest=json.loads((root/name/'SOURCE.json').read_text())
            for path,row in manifest['files'].items():
                self.assertEqual(hashlib.sha256((root/name/path).read_bytes()).hexdigest(),row['sha256'])
        self.assertEqual(flux_system_prompt(),flux_system_prompt())
        self.assertEqual(h3_system_prompt(),h3_system_prompt())
        self.assertNotEqual(flux_system_prompt(),h3_system_prompt())
        self.assertIn('Official guide: flux-image-best-practices/SKILL.md',flux_system_prompt())
        self.assertIn('Official guide: h3-prompt-writing/references/ref-en.txt',h3_system_prompt())

    def test_global_ids_preserve_origin_and_bindings(self):
        attrs=[dict(id='video_bear',category='identity',text='Brown bear.',artifact_ids=['video']),
               dict(id='garden_bear',category='identity',text='White bear.',artifact_ids=['garden']),
               dict(id='video_pose',category='identity.body_pose',text='Standing.',parent_id='video_bear'),
               dict(id='new_relation',category='identity_relations',text='Facing each other.',
                    links=['video_bear','garden_bear'])]
        rows,_,_=prepare_inputs(attrs,self.refs,'h3')
        self.assertEqual(rows[2]['parent_id'],'video_bear')
        self.assertEqual(rows[3]['links'],['video_bear','garden_bear'])
        for bad in ({'artifact_id':'video','attributes':{}},
                    [dict(id='bear',category='identity',text='Bear.')],
                    [dict(id='video_pose',category='identity.body_pose',text='Standing.',parent_id='bear')]):
            with self.assertRaises(ValueError):prepare_inputs(bad,[],'flux')

    def test_h3_new_sound_policy(self):
        prompt=h3_system_prompt()
        self.assertIn('Design plausible NEW sound',prompt)
        self.assertNotIn('Unless the brief explicitly requests new generated',prompt)
        self.assertIn('all six Ref2VA sections',prompt)
        self.assertIn('all three T2VA sections',prompt)
