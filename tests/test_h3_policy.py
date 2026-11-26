"""Operation routing, actual graph recipes and resident-worker isolation."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cocreation import H3Runtime, MiniMaxH3
from cocreation.h3 import build_graph
from cocreation.h3_policy import inference_recipe
from scripts.dialogue_generation.materialize import materialize, materialization_order
from tests.test_materialize import FakeModels
from tests.test_video_tone import task, token


class H3PolicyTests(unittest.TestCase):
    def test_mixed_recipes_and_fifty_step_fallback(self):
        opts=dict(new_turbo=True,new_steps=6,edit_steps=20,edit_sampler='res_multistep')
        for op in ('text_generation','reference_generation'):
            r=inference_recipe(op,**opts)
            self.assertTrue(r['turbo']);self.assertEqual(r['steps'],6)
        r=inference_recipe('video_edit',**opts)
        self.assertFalse(r['turbo']);self.assertEqual((r['steps'],r['sampler']),(20,'res_multistep'))
        self.assertEqual(inference_recipe('video_edit',new_turbo=True)['steps'],50)
        self.assertEqual(inference_recipe('reference_generation',turbo=True)['steps'],6)
        for opts in (dict(new_turbo=True,turbo=True),dict(edit_steps=0),dict(new_steps=True),dict(edit_sampler='invalid')):
            with self.assertRaises(ValueError):inference_recipe('video_edit',**opts)

    def test_graph_base_has_no_lora_and_uses_selected_sampler(self):
        request=MiniMaxH3(runtime=H3Runtime(turbo=False))('A park.',seconds=4,steps=20,sampler='res_multistep',dry_run=True)
        graph=build_graph(request,[])
        self.assertNotIn('turbo_lora',graph)
        self.assertEqual(graph['sample']['inputs']['sampler_name'],'res_multistep')
        self.assertEqual(graph['sample']['inputs']['steps'],20)
        self.assertEqual(graph['shift']['inputs']['model'],['model',0])
        request=MiniMaxH3(runtime=H3Runtime(turbo=True))('A park.',seconds=4,steps=6,dry_run=True)
        graph=build_graph(request,[])
        self.assertIn('turbo_lora',graph)
        self.assertFalse(graph['turbo_lora']['inputs']['low_vram'])
        self.assertEqual(graph['turbo_sampler']['class_type'],'MiniMaxH3TurboSampler')
        self.assertEqual(graph['sigmas']['inputs']['steps'],6)
        with self.assertRaises(ValueError):
            MiniMaxH3(runtime=H3Runtime(turbo=True))('x',sampler='res_multistep',dry_run=True)

    def test_scheduler_groups_ready_modes(self):
        a=task(1);b=task(2,'video_edit',[1]);c=task(3);d=task(4,'video_edit',[3])
        for row in (a,b,c,d):row['backend']='minimax_h3'
        tasks={r['placeholder']:r for r in (a,b,c,d)}
        recipes={id(r):inference_recipe(r['operation'],new_turbo=True) for r in tasks.values()}
        order=materialization_order(tasks,recipes=recipes)
        flags=[recipes[id(tasks[t])]['turbo'] for t in order]
        self.assertEqual(flags,[True,True,False,False])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class MixedMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name)
        rows=[task(1),task(2,'video_edit',[1]),task(3,refs=[2]),task(4,'video_edit',[3])]
        for r in rows:r['backend']='minimax_h3'
        (self.run/'placeholders.json').write_text(json.dumps(rows))
        (self.run/'dialogue.md').write_text(' '.join(token(n) for n in range(1,5)))
        self.opts=dict(new_video_turbo_lora=True,new_video_steps=6,edit_video_steps=20,edit_video_sampler='res_multistep',
                       output_width=96,output_height=64,progress=False)
    def test_switch_closes_previous_worker_and_resume_keeps_unchanged_new_root(self):
        fake=FakeModels();events=[]
        def factory(backend,model_id,output_dir,*,video_turbo_lora=False):
            events.append(('load',video_turbo_lora))
            model=fake(backend,model_id,output_dir);original=model.close
            def close():events.append(('close',video_turbo_lora));original()
            model.close=close
            return model
        with patch('scripts.dialogue_generation.materialize.default_model',side_effect=factory), \
             patch('scripts.dialogue_generation.materialize.preflight_models',return_value={}):
            report=materialize(self.run,**self.opts)
            self.assertEqual(events,[(kind,flag) for flag in (True,False,True,False) for kind in ('load','close')])
            self.assertEqual([c['options']['steps'] for c in fake.calls],[6,20,6,20])
            for n in (2,4):
                row=report['media'][token(n)]
                self.assertFalse(row['inference_recipe']['turbo'])
                self.assertEqual(row['options']['sampler'],'res_multistep')
            root_time=(self.run/'media/video_0001.mp4').stat().st_mtime_ns
            count=len(fake.calls)
            materialize(self.run,resume=True,**self.opts)
            self.assertEqual(len(fake.calls),count)
            materialize(self.run,resume=True,**dict(self.opts,edit_video_steps=50))
            self.assertEqual((self.run/'media/video_0001.mp4').stat().st_mtime_ns,root_time)
            self.assertEqual(fake.calls[count]['options']['steps'],50)
    def test_uniform_base_forty_reaches_every_request(self):
        report=materialize(self.run,dry_run=True,video_steps=40,edit_video_sampler='res_multistep',
                           output_width=96,output_height=64,progress=False)
        self.assertEqual(report['settings']['video_steps_by_backend']['minimax_h3'],40)
        self.assertFalse(report['settings']['video_turbo_lora'])
        recipes=[recipe for rows in report['h3_request_recipes'].values() for recipe in rows]
        self.assertEqual(len(recipes),4)
        self.assertTrue(all(recipe['steps']==40 and recipe['turbo'] is False for recipe in recipes))
        self.assertEqual([r['sampler'] for r in recipes],['euler','res_multistep','euler','res_multistep'])
        self.assertFalse((self.run/'media').exists())

    def test_dry_run_reports_new_and_edit_recipes_without_calling_models(self):
        report=materialize(self.run,dry_run=True,**self.opts)
        self.assertEqual(report['h3_request_recipes'][token(1)][0]['steps'],6)
        self.assertEqual(report['h3_request_recipes'][token(2)][0]['steps'],20)
        self.assertFalse(report['h3_request_recipes'][token(2)][0]['turbo'])
        self.assertFalse((self.run/'media').exists())


if __name__=='__main__':unittest.main()
