"""Image integration contracts: source scope, slot order, defaults and legacy Flux."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from cocreation import Flux, MiniMaxH3Image, Reference
from cocreation.h3_image import build_graph, size_for, TRANSFORMER, ENCODER
from tools.h3_image_prompter import h3_image_prompter, h3_image_generation_prompt
from tools.h3_image_prompter_prompts import h3_image_system_prompt
from tools.placeholder_prompter import writer_input, writer_system, rewrite_placeholder
from scripts.dialogue_generation.generation import build_tasks
from scripts.dialogue_generation.planner import build_plan
from scripts.dialogue_generation.materialize import materialize, load_contract, execution_key

ROOT=Path(__file__).resolve().parents[1]


class H3ImageTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.media=Path(tmp.name)

    def source(self,name):
        path=self.media/name
        path.touch()
        return path

    def test_defaults_and_legacy_flux_are_independent(self):
        request=MiniMaxH3Image()('A small ceramic vessel.',dry_run=True)
        self.assertEqual(request['backend'],'h3_image')
        self.assertEqual((request['options']['width'],request['options']['height']),(4096,4096))
        self.assertEqual(request['options']['steps'],6)
        self.assertEqual(Flux()('A vessel.',dry_run=True)['options']['steps'],4)
        self.assertEqual(size_for('9:16'),(3072,5472))
        with self.assertRaises(ValueError):
            MiniMaxH3Image()('A vessel.',steps=4,dry_run=True)
        with self.assertRaises(ValueError):
            MiniMaxH3Image()('A vessel.',references=[Reference('v','x.mp4','video')],dry_run=True)

    def test_graph_uses_all_ordered_references_and_native_schedule(self):
        refs=[Reference('a','a.png','image'),Reference('b','b.png','image')]
        request=MiniMaxH3Image()('Use {{artifact:b}} and {{artifact:a}}.',refs,dry_run=True)
        self.assertEqual(request['prompt'],'Use <Picture 2> and <Picture 1>.')
        graph=build_graph(request,['a.png','b.png'])
        self.assertEqual(graph['condition']['inputs']['images.image_1'],['image_1',0])
        self.assertEqual(graph['sample']['inputs']['scheduler'],'simple')
        self.assertEqual(graph['sample']['inputs']['steps'],6)
        self.assertEqual(graph['sample']['inputs']['cfg'],1.)
        self.assertEqual(graph['model']['inputs']['unet_name'],TRANSFORMER)
        self.assertEqual(graph['encoder']['inputs']['clip_name'],ENCODER)

    def test_prompt_writer_sees_only_selected_text_and_matching_mode(self):
        model=Mock(return_value=dict(text='The shape from <Picture 1> defines <Subject 1>. The target image preserves that shape.'))
        model.supports_system_cache=False
        ref=Reference('a',self.source('unread-media.png'),'image')
        result=h3_image_prompter('The shape is retained. (from <Image 1>)',[ref],qwen=model)
        self.assertIn('<Picture 1>',result)
        args,options=model.call_args
        self.assertEqual(args[1],[])
        self.assertNotIn('unread-media.png',args[0])
        self.assertNotIn('A scene with physical contact',options['system_prompt'])
        self.assertNotIn('red cup',options['system_prompt'])
        self.assertNotIn('Reference-image edit',h3_image_system_prompt(False))
        self.assertNotIn('Text-to-image prompt writing guide',h3_image_system_prompt(True))

    def test_mixed_source_labels_remap_once_after_frame_selection(self):
        refs=[Reference('v',self.source('v.mp4'),'video'),Reference('i',self.source('i.png'),'image')]
        images=[Reference('v',self.source('still.png'),'image'),refs[1]]
        text='Use <Video 1> for <Subject 1> and <Picture 1> for the selected surface.'
        self.assertEqual(h3_image_generation_prompt(text,refs,images),
            'Use <Picture 1> for <Subject 1> and <Picture 2> for the selected surface.')

    def test_edit_input_omits_previous_values_and_unrelated_prose(self):
        task=dict(backend='h3_image',operation='local_edit',layout='product',
            desired_description='UNWANTED_SCENE',desired_attributes={'color':'STALE_COLOR'},
            changes=[dict(field='color',before='OLD_COLOR',after='blue')],
            references=[dict(label='<Picture 1>',kind='image',role='base to edit',description='OLD_CAPTION')])
        context=writer_input(task)
        for word in ('UNWANTED','STALE','OLD_','young girl','white hair'):
            self.assertNotIn(word,context)
        request=Mock(return_value=('The bag in <Picture 1> is <Subject 1>. The target image has a blue bag. Preserve every other detail.', 'stop'))
        self.assertIn('blue',rewrite_placeholder(task,request))
        self.assertEqual(request.call_count,1)

    def test_image_guide_reused_when_request_changes_layout_orientation_and_accents(self):
        task=dict(backend='h3_image', kind='image', operation='local_edit', references=[],
                  layout='product', presentation=dict(seasonal_accents='small blue blossoms'),
                  render_state=dict(aspect_ratio='portrait'))
        other=dict(task, layout='portrait', presentation=dict(seasonal_accents='orange fallen leaves'),
                   render_state=dict(aspect_ratio='landscape'))
        self.assertEqual(writer_system(task),writer_system(other))
        self.assertNotIn('orange fallen leaves',writer_system(other))
        data=json.loads(writer_input(other).split('instructions.\n',1)[1])
        self.assertEqual(data['layout'],'portrait')
        self.assertEqual(data['output_format']['orientation'],'landscape')
        self.assertEqual(data['presentation']['seasonal_accents'],'orange fallen leaves')

    def test_new_plans_use_h3_image_and_picture_labels(self):
        observation=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        tasks=build_tasks(build_plan(observation,seed=68,reworks=2),observation)
        images=[t for t in tasks.values() if t.get('backend')=='h3_image']
        self.assertTrue(images)
        for task in images:
            self.assertEqual(task['model_id'],'MiniMaxAI/MiniMax-H3-Image')
            self.assertTrue(all(r['label'].startswith('<Picture ') for r in task['references']))
            self.assertEqual(execution_key(task)[0],'h3_image')

    def test_materializer_recommends_16mp_and_preserves_flux_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'dialogue.md').write_text('[[image_0001]]')
            task=dict(placeholder='[[image_0001]]',kind='image',operation='text_generation',
                backend='h3_image',prompt='A blue bag.',dependencies=[],references=[],render_state={})
            (root/'placeholders.json').write_text(json.dumps([task]))
            report=materialize(root,dry_run=True,progress=False)
            self.assertEqual(report['output_sizes']['[[image_0001]]'],[4096,4096])
            task['render_state']={'aspect_ratio':'portrait'}
            (root/'placeholders.json').write_text(json.dumps([task]))
            self.assertEqual(materialize(root,dry_run=True,progress=False)['output_sizes']['[[image_0001]]'],[3072,5472])
            copied=dict(placeholder='[[image_0002]]',kind='image',operation='copy',
                dependencies=['[[image_0001]]'],references=[],render_state={'aspect_ratio':'portrait'})
            (root/'placeholders.json').write_text(json.dumps([task,copied]))
            self.assertEqual(materialize(root,dry_run=True,progress=False)['output_sizes']['[[image_0002]]'],[3072,5472])
            task.update(backend='flux',render_state={})
            (root/'placeholders.json').write_text(json.dumps([task]))
            self.assertEqual(materialize(root,dry_run=True,progress=False)['output_sizes']['[[image_0001]]'],[1024,1024])


if __name__=='__main__':
    unittest.main()
