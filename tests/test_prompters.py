"""Prose inputs, direct expert outputs, media ordering and generator constraints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from cocreation import Reference
from cocreation.references import prepare
from cocreation.worker import qwen_messages
from tools._prompter import prepare_inputs, parse_prompt
from tools.flux_prompter import flux_prompter, flux_generation_prompt
from tools.h3_prompter import h3_prompter
from tools.flux_prompter_prompts import flux_system_prompt
from tools.h3_prompter_prompts import h3_system_prompt

REF_PROMPT = '''subject_definitions:
<Subject 1> is the red coat like in <Video 1>.
<Subject 2> is the garden like in <Picture 1>.
summary:
A five-second portrait in the garden.
retention_analysis:
Preserve the coat and garden only.
detailed_description:
[Shot 1] A woman wearing <Subject 1> stands in <Subject 2>.
[Shot 2] At 00:02.000, a close view shows her turning toward the camera.
overall_soundscape:
Soft wind.
non_diegetic_music:
N/A'''
BASE_PROMPT = '''integrated_multimodal_description:
[Shot 1] An abstract violet circle slowly brightens during five seconds.
overall_soundscape:
N/A
non_diegetic_music:
N/A'''
FLUX_PROMPT = 'A portrait with the red coat like in <Video 1>, in the garden like in <Image 1>, soft light.'


class Prompters(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        for name in ('video.mp4','garden.png'): (self.root/name).touch()
        self.refs=[Reference('video',self.root/'video.mp4','video'),
                   Reference('garden',self.root/'garden.png','image')]
        self.description='A red coat. (from <Video 1>)\nA rose garden. (from <Image 1>)\nSoft morning light.'

    def test_direct_text_and_mixed_media_maps(self):
        for target, tool, prompt in [('flux',flux_prompter,FLUX_PROMPT),('h3',h3_prompter,REF_PROMPT)]:
            with self.subTest(target=target):
                model=Mock(return_value={'text':prompt})
                self.assertEqual(tool(self.description,self.refs,qwen=model,output_dir=self.root/target),prompt)
                user, refs=model.call_args.args
                self.assertIn(self.description, user)
                if target == 'flux':
                    self.assertNotIn(' -> ',user)
                    self.assertNotIn('Reference mapping',user)
                    self.assertNotIn('<Image 2>',user)
                    self.assertIn('do not rename or renumber',user)
                else:
                    self.assertIn('<Video 1> -> <Video 1>',user)
                    self.assertIn('<Image 1> -> <Picture 1>',user)
                self.assertNotIn('"attributes"',user)
                self.assertEqual(refs,self.refs)
                self.assertTrue((self.root/target/'input.txt').is_file())
                model.close.assert_not_called()

    def test_media_inserted_once_in_observation_order_before_mapping(self):
        model=Mock(return_value={'text':FLUX_PROMPT})
        flux_prompter(self.description,self.refs,qwen=model)
        user,refs=model.call_args.args
        request=prepare(user,refs)
        request['options']=dict(max_pixels=123,video_fps=2,video_max_frames=32)
        items=qwen_messages(request)[0]['content']
        self.assertEqual([x['type'] for x in items if x['type']!='text'],['video','image'])
        self.assertTrue(items[0]['text'].endswith('<Video 1>'))

    def test_source_free_generation_and_owned_lifecycle(self):
        for name,tool,prompt in [('flux',flux_prompter,'A luminous pearl sphere.'),('h3',h3_prompter,BASE_PROMPT)]:
            model=Mock(return_value={'text':prompt})
            with patch(f'tools.{name}_prompter.Qwen',return_value=model):
                self.assertEqual(tool('A pearl sphere.',use_system_cache=False),prompt)
            model.close.assert_called_once()
            self.assertIsNone(model.call_args.kwargs['system_cache_path'])
        model=Mock(side_effect=RuntimeError('failed'))
        with patch('tools.flux_prompter.Qwen',return_value=model), self.assertRaises(RuntimeError):
            flux_prompter('A pearl sphere.')
        model.close.assert_called_once()

    def test_input_rejects_dangling_and_unused_sources(self):
        for text,refs in [('A red coat.',self.refs),('A coat. (from <Image 2>)',self.refs),
                          ([],[]),('A coat like in <Video 1>.',self.refs),
                          ('A coat. (from <Audio 1>)',self.refs)]:
            with self.subTest(text=text),self.assertRaises(ValueError):
                prepare_inputs(text,refs,'flux')
        with self.assertRaises(ValueError):
            prepare_inputs(self.description,self.refs+self.refs,'h3')

    def test_direct_prompt_validation(self):
        _,_,catalog=prepare_inputs(self.description,self.refs,'flux')
        for text in (json.dumps({'prompt':FLUX_PROMPT}), '```'+FLUX_PROMPT+'```',
                     'CANNOT_FULFILL', FLUX_PROMPT.replace('<Video 1>','<Video 2>'),
                     'A coat like in <Image 1>.', FLUX_PROMPT+' word'*201):
            with self.subTest(text=text):
                self.assertIsNone(parse_prompt({'text':text},catalog,'flux'))
        self.assertEqual(parse_prompt({'text':FLUX_PROMPT},catalog,'flux'),FLUX_PROMPT)

    def test_sparse_foliage_prompt_and_inline_source_contract(self):
        description='Green foliage is visible on the right. (from <Video 1>)'
        _,_,catalog=prepare_inputs(description,self.refs[:1],'flux')
        correct='Green foliage is visible on the right, like in <Video 1>.'
        self.assertEqual(parse_prompt({'text':correct},catalog,'flux'),correct)
        self.assertIsNone(parse_prompt({'text':'Green foliage is visible on the right.'},catalog,'flux'))
        self.assertIsNone(parse_prompt({'text':'Green foliage is visible on the right. References: <Video 1>'},catalog,'flux'))
        self.assertIsNone(parse_prompt({'text':correct.replace('like in','from')},catalog,'flux'))
        for system in (flux_system_prompt(),h3_system_prompt()):
            self.assertIn('This is NOT an image',system)
            self.assertIn('ALL visible elements and attributes not explicitly selected',system)
            self.assertIn('EVERY sourced requirement',system)
            self.assertTrue(system.endswith('Return only the requested direct plain text.\n'))
        self.assertIn('NO minimum',flux_system_prompt())
        model=Mock(return_value={'text':correct})
        self.assertEqual(flux_prompter(description,self.refs[:1],qwen=model),correct)
        self.assertIn('After EACH sourced requirement',model.call_args.args[0])
        _,_,h3_catalog=prepare_inputs(self.description,self.refs,'h3')
        self.assertIsNone(parse_prompt({'text':REF_PROMPT.replace('like in','from')},h3_catalog,'h3',5))

    def test_flux_label_conversion_is_simultaneous_and_checks_slots(self):
        frame=self.root/'frame.png'; frame.touch()
        images=[Reference('video',frame,'image'),self.refs[1]]
        raw=FLUX_PROMPT+' Keep the coat like in <Video 1>.'
        self.assertEqual(flux_generation_prompt(raw,self.refs,images),
            'A portrait with the red coat like in <Image 1>, in the garden like in <Image 2>, soft light. Keep the coat like in <Image 1>.')
        self.assertEqual(flux_generation_prompt('A pearl sphere.',[],[]),'A pearl sphere.')
        with self.assertRaises(ValueError): flux_generation_prompt(raw,self.refs,list(reversed(images)))
        with self.assertRaises(ValueError): flux_generation_prompt(raw,self.refs,self.refs)
        with self.assertRaises(ValueError): flux_generation_prompt(raw+' <Video 9>',self.refs,images)
        # Image-first order, two video slots and repeated mentions also stay distinct.
        other=Reference('second_video',self.root/'video.mp4','video')
        originals=[self.refs[1],self.refs[0],other]
        images=[self.refs[1],Reference('video',frame,'image'),Reference('second_video',frame,'image')]
        self.assertEqual(flux_generation_prompt('<Video 2> beside <Image 1> and <Video 1>.',originals,images),
                         '<Image 3> beside <Image 1> and <Image 2>.')

    def test_h3_structure_shots_and_times_remain_validated(self):
        _,_,catalog=prepare_inputs(self.description,self.refs,'h3')
        for text in (REF_PROMPT.replace('00:02.000','00:05.000'),
                     REF_PROMPT.replace('[Shot 2]','[Shot 3]'),
                     REF_PROMPT.replace('summary:','wrong:'),
                     REF_PROMPT.replace('[Shot 1]','[Shot 1] At 00:00.000,'),
                     REF_PROMPT.replace('<Subject 2> is','A garden is')):
            with self.subTest(text=text):
                self.assertIsNone(parse_prompt({'text':text},catalog,'h3',5))
        self.assertEqual(parse_prompt({'text':REF_PROMPT},catalog,'h3',5),REF_PROMPT)
        self.assertEqual(parse_prompt({'text':BASE_PROMPT},[],'h3',5),BASE_PROMPT)

    def test_role_and_task_examples(self):
        self.assertIn('<Video N> stays <Video N>',flux_system_prompt())
        self.assertNotIn('never\n<Video N>',flux_system_prompt())
        for system in (flux_system_prompt(),h3_system_prompt()):
            self.assertTrue(system.startswith('You are '))
            self.assertIn('example',system.lower())
            self.assertIn('direct plain text',system)
            self.assertNotIn('{"prompt":',system)


if __name__ == '__main__': unittest.main()
