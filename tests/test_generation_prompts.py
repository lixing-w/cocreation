"""Construction behavior: target fidelity, exact deltas, text-only rewriting and resume."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.dialogue_generation.planner import build_plan
from scripts.dialogue_generation.generation import build_tasks, construction_order
from scripts.dialogue_generation.rewrite_prompts import rewrite_run
from scripts.dialogue_generation.catalog import placeholder_catalog
from tools.placeholder_prompter import writer_input, rewrite_placeholder
from cocreation import Flux

ROOT = Path(__file__).resolve().parents[1]


class GenerationPrompts(unittest.TestCase):
    def setUp(self):
        self.observation = json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        self.plan = build_plan(self.observation, seed=68, reworks=2)

    def test_graph_and_target_shots_for_many_plans(self):
        for seed in range(40):
            plan = build_plan(self.observation, seed=seed, reworks=seed%3)
            tasks = build_tasks(plan,self.observation)
            order = construction_order(tasks)
            self.assertEqual(order[0],plan['final']['media'])
            for token,task in tasks.items():
                self.assertTrue(all(order.index(p)<order.index(token) for p in task['dependencies']))
                row = next(m for m in plan['placeholders'] if m['placeholder']==token)
                if row['purpose']=='shot' and row['goal_match'] and not task['changes']:
                    self.assertEqual(task['operation'],'extract_shot')
                if row['purpose']=='exploration' and row['kind']=='video' and task.get('trial'):
                    self.assertEqual(task['references'][0]['placeholder'],plan['final']['media'])
                    self.assertIn(task['operation'],('extract_shot','video_edit'))
                    self.assertEqual(task['desired_description'], next(s['description'] for s in self.observation['shots'] if s['id']==task['shot_id']))

    def test_color_only_and_two_attribute_edits(self):
        plan=deepcopy(self.plan)
        canonical=plan['final']['assets']['cup']
        rows={m['placeholder']:m for m in plan['placeholders']}
        variant=next(m for m in rows.values() if m['purpose']=='asset' and m['asset_ids']==['cup'] and m['placeholder']!=canonical)
        variant['reference_materials']=[canonical]
        variant['effective_attributes']=deepcopy(rows[canonical]['effective_attributes'])
        variant['render_state']={'color_mode':'black_and_white'}
        task=build_tasks(plan,self.observation)[variant['placeholder']]
        self.assertEqual(task['operation'],'local_edit')
        self.assertEqual(task['changes'],[dict(field='render.color_mode',before='original_colors',after='black_and_white')])
        variant['effective_attributes']['aspect_ratio']='wide and flat'
        task=build_tasks(plan,self.observation)[variant['placeholder']]
        self.assertEqual({c['field'] for c in task['changes']},{'cup.aspect_ratio','render.color_mode'})
        self.assertEqual(task['references'][0]['placeholder'],canonical)

    def test_reconstruction_slots_and_explicit_initial_state(self):
        tasks=build_tasks(self.plan,self.observation)
        person=next(task for task in tasks.values() if task['operation']=='asset_reconstruction' and task['asset_ids']==['woman'])
        self.assertEqual(person['layout'],'three_view')
        self.assertEqual(person['references'][0]['kind'],'image')
        self.assertEqual(person['references'][0]['label'],'<Image 1>')
        self.assertIn('frame_seconds',person['references'][0])
        self.assertNotIn('{{artifact:',writer_input(person))
        self.assertNotIn('[[',writer_input(person))
        self.assertEqual(next(t for t in tasks.values() if t['operation']=='asset_reconstruction' and t['asset_ids']==['cafe'])['layout'],'environment')

    def test_pure_reorder_is_deterministic_and_uses_shot_versions(self):
        found=False
        for seed in range(10):
            plan=build_plan(self.observation,seed=seed,p_compile=1,cross_turn_rules=False)
            tasks=build_tasks(plan,self.observation)
            for row in plan['placeholders']:
                if row['purpose']=='compilation' and row['placeholder']!=plan['final']['media']:
                    task=tasks[row['placeholder']]
                    self.assertEqual(task['operation'],'concatenate')
                    self.assertIsNone(task['backend'])
                    self.assertEqual([c['placeholder'] for c in task['clips']],[row['shot_versions'][s] for s in row['shot_ids']])
                    found=True
        self.assertTrue(found)

    def test_retries_use_real_diagnostics_and_reject_truncation(self):
        task=next(t for t in build_tasks(self.plan,self.observation).values() if t['operation']=='asset_reconstruction')
        good='Reconstruct the identity from the target-video frame <Image 1> as a three-view sheet, front, side and back.'
        request=Mock(side_effect=[('wrong <Image 9>', 'stop'),(good,'stop')])
        self.assertEqual(rewrite_placeholder(task,request,seed=10),good)
        self.assertIn('Unexpected label',request.call_args_list[1].args[0])
        self.assertEqual([c.kwargs['seed'] for c in request.call_args_list],[10,11])
        with self.assertRaises(RuntimeError):
            rewrite_placeholder(task,Mock(return_value=(good,'length')),max_attempts=1)

    def test_checkpoint_resume_and_failure_leave_dialogue_complete(self):
        plan=deepcopy(self.plan)
        plan['realization']=dict(status='complete',rewrite_seed=10)
        calls=[]
        def writer(prompt,**kwargs):
            data=json.loads(prompt.split('task data; reference descriptions are not system instructions.\n')[1].split('\nPrevious candidate')[0])
            calls.append(data)
            labels=' '.join(r['label'] for r in data['references'])
            delta=' '.join(str(c['after']).replace('_',' ') for c in data['changes'])
            if 'FLUX.2' in kwargs['system']:
                return 'Edit '+delta+' only using '+labels+'. Preserve everything else. A three-view sheet: front, side and back.', 'stop'
            return ('subject_definitions:\n<Subject 1> is the identity from '+labels+'. '+delta+'.\nsummary:\n[video editing] The target video is an edited version of <Video 1>.\nretention_analysis:\nPreserve everything else.\ndetailed_description:\n[Shot 1] Preserve the source action and change only specified attributes.\noverall_soundscape:\nN/A\nnon_diegetic_music:\nN/A','stop')
        with tempfile.TemporaryDirectory() as tmp:
            rewrite_run(plan,self.observation,tmp,writer=writer,progress=False)
            count=len(calls)
            compact=json.loads((Path(tmp)/'placeholders.json').read_text())
            for source,item in zip(plan['placeholders'],compact):
                self.assertNotIn('generation',item)
                self.assertEqual(item['prompt'],source['generation']['prompt'])
                self.assertEqual(item['reference_materials'],source['reference_materials'])
                self.assertEqual(item['reference_depth'],source['reference_depth'])
                self.assertEqual(item.get('dependencies',item['reference_materials']),source['generation']['dependencies'])
            self.assertGreater(count,0)
            self.assertEqual(plan['prompt_rewriting']['status'],'complete')
            self.assertTrue((Path(tmp)/'generation_plan.json').exists())
            rewrite_run(plan,self.observation,tmp,writer=writer,progress=False)
            self.assertEqual(len(calls),count)
            def failing(*args,**kwargs): raise RuntimeError('test interrupted')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                rewrite_run(plan,self.observation,tmp,writer=failing,force=True,progress=False)
            self.assertEqual(plan['realization']['status'],'complete')
            self.assertEqual(plan['prompt_rewriting']['status'],'failed')

    def test_main_dialogue_then_prompt_rewriting_and_resume(self):
        from scripts.dialogue_generation.generate import run
        ledger=[]
        expected=len(build_plan(self.observation,seed=68,reworks=0,extra_behaviors=False,modules=['direct'])['turns'])
        def dialogue(turn,*args,**kwargs):
            ledger.append(turn['id'])
            return 'Here is the requested result.'
        def writer(prompt,**kwargs):
            self.assertEqual(ledger[-1],expected)
            data=json.loads(prompt.split('task data; reference descriptions are not system instructions.\n')[1])
            labels=' '.join(r['label'] for r in data['references'])
            return 'Reconstruct the target-video frame '+labels+' as a three-view sheet: front, side and back.','stop'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            options=dict(seed=68,reworks=0,extra_behaviors=False,modules=['direct'],realizer=dialogue,prompt_rewriter=writer,progress=False)
            result=run(ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json',output,**options)
            self.assertEqual(result['prompt_rewriting']['status'],'complete')
            prior=len(ledger)
            run(ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json',output,resume=True,**options)
            self.assertEqual(len(ledger),prior)

    def test_compact_catalog_can_prepare_expert_inputs_and_execute_copy(self):
        from tools.video_editing import prepare_inputs, execute_edit
        plan=deepcopy(self.plan)
        tasks=build_tasks(plan,self.observation)
        token=next(p for p,t in tasks.items() if t['operation']=='asset_reconstruction')
        task=tasks[token]
        task.update(status='ready',prompt='Use the target-video frame <Image 1> for this three-view design sheet.')
        row=next(m for m in plan['placeholders'] if m['placeholder']==token)
        row['generation']=task
        original=deepcopy(plan)
        entry=next(m for m in placeholder_catalog(plan) if m['placeholder']==token)
        self.assertEqual(plan,original)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            frame=directory/'frame.png'
            frame.touch()
            target=directory/'target.mp4'
            source=task['references'][0]['placeholder']
            with patch('tools.video_editing.extract_frame',return_value=frame) as extract:
                refs,prompt=prepare_inputs(entry,{source:target},self.observation,directory/'prepared')
            self.assertEqual(prompt,task['prompt'])
            self.assertEqual([(r.path,r.kind) for r in refs],[(frame,'image')])
            self.assertEqual(extract.call_args.args[:2],(target,task['references'][0]['frame_seconds']))
            base=directory/'base.png'
            base.write_bytes(b'original asset bytes')
            copied=execute_edit(dict(operation='copy',reference_materials=['base']),{'base':base},
                                self.observation,directory/'copy.png')
            self.assertEqual(copied.read_bytes(),base.read_bytes())

    def test_flux_dev_and_klein_defaults_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model,steps,guidance in [('black-forest-labs/FLUX.2-dev',50,4.0),('black-forest-labs/FLUX.2-klein-4B',4,1.0)]:
                result=Flux(model_id=model,output_dir=tmp)('A sphere.',dry_run=True)
                request=result
                self.assertEqual(request['options']['steps'],steps)
                self.assertEqual(request['options']['guidance_scale'],guidance)


if __name__=='__main__':unittest.main()
