"""Forward contracts, concise text rewriting, exact edits and checkpoint resume."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from scripts.dialogue_generation.planner import build_plan, Planner, observation
from scripts.dialogue_generation.references import assign_reference_materials
from scripts.dialogue_generation.generation import build_tasks, construction_order
from scripts.dialogue_generation.generation_requirements import resolve_requirements
from scripts.dialogue_generation.rewrite_prompts import rewrite_run
from tools.placeholder_prompter import writer_input, writer_system, writer_contract, rewrite_placeholder
ROOT=Path(__file__).resolve().parents[1]


def writer(prompt,**kwargs):
    data=json.loads(prompt.split('task data; reference descriptions are not system instructions.\n')[1].split('\nPrevious candidate')[0])
    labels=' '.join(r['label'] for r in data['references'])
    delta=' '.join(str(c['after']).replace('_',' ') for c in data['changes'])
    delta+=' '+ ' '.join(a['attributes'].get('material','') for r in data['references'] for a in r.get('appearance', []))
    if data.get('output_color_mode')=='black_and_white': delta+=' Render the entire output in black and white.'
    if 'layout' in data:
        return 'Edit '+labels+'. '+delta+'. Preserve all other details. One coherent image. Neutral studio backdrop. '+data.get('presentation',{}).get('seasonal_accents',''), 'stop'
    if not data['references']:
        return 'integrated_multimodal_description: [Shot 1] A traveler walks. '+delta+'\noverall_soundscape: N/A\nnon_diegetic_music: N/A','stop'
    return ('subject_definitions: '+labels+'\nsummary: [video editing] Edit <Video 1>. '+delta+
        '\nretention_analysis: Preserve everything else.\ndetailed_description: [Shot 1] Retain source action. '+delta+
        '\noverall_soundscape: N/A\nnon_diegetic_music: N/A','stop') if data['operation']=='video_edit' else (
        'subject_definitions: '+labels+'\nsummary: [reference generation] A traveler walks. '+delta+
        '\nretention_analysis: Preserve identities.\ndetailed_description: [Shot 1] A traveler walks.\noverall_soundscape: N/A\nnon_diegetic_music: N/A','stop')


class GenerationPrompts(unittest.TestCase):
    def setUp(self):
        self.source=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        self.plan=build_plan(self.source,seed=68,reworks=2)

    def test_no_guide_inputs_and_chronological_construction(self):
        for seed in range(12):
            plan=build_plan(self.source,seed=seed,reworks=2)
            tasks=build_tasks(plan,self.source)
            self.assertEqual(construction_order(tasks),[r['placeholder'] for r in plan['placeholders']])
            self.assertNotIn('source',[t['operation'] for t in tasks.values()])
            self.assertTrue(any(t['operation']=='text_generation' and not t['references'] for t in tasks.values()))
            for task in tasks.values():
                self.assertNotIn('source_video',task)
                for ref in task['references']:
                    self.assertTrue(ref.get('placeholder') in task['dependencies'] or task.get('preprocess'))

    def test_initial_assets_and_cropped_video_parents(self):
        found=False
        for seed in range(12):
            plan=build_plan(self.source,seed=seed,reworks=2,modules=['candidates'])
            rows={r['placeholder']:r for r in plan['placeholders']}
            for token,task in build_tasks(plan,self.source).items():
                if task['operation']=='text_generation' and task['backend']=='flux':
                    self.assertEqual(task['references'],[])
                    self.assertIn('desired_attributes',task)
                for ref in task['references']:
                    if ref.get('placeholder') and rows[ref['placeholder']]['kind']=='video' and ref['kind']=='image':
                        self.assertTrue(ref['crop_query'])
                        self.assertNotIn('frame_seconds',ref)
                        found=True
        self.assertTrue(found)

    def test_reorder_is_deterministic(self):
        plan=build_plan(self.source,seed=1,p_compile=1,cross_turn_rules=False)
        tasks=build_tasks(plan,self.source)
        for row in plan['placeholders']:
            if row['purpose']=='compilation':
                task=tasks[row['placeholder']]
                self.assertEqual(task['operation'],'concatenate')
                self.assertEqual([c['placeholder'] for c in task['clips']],[row['shot_versions'][sid] for sid in row['shot_ids']])
                self.assertIsNone(task['backend'])

    def test_source_action_and_subject_bindings_for_edits(self):
        tasks=build_tasks(self.plan,self.source)
        edits=[t for t in tasks.values() if t['operation']=='video_edit' and t.get('shot_id')]
        self.assertTrue(edits)
        for task in edits:
            contract=writer_contract(task)
            self.assertEqual(contract['source_action'],task['source_action'])
            self.assertEqual(contract['references'][0]['kind'],'video')
            self.assertTrue(any(r.get('subjects') for r in contract['references'][1:]))

    def test_final_acceptance_does_not_cancel_and_rebuild_rule_media(self):
        plan=build_plan(self.source,seed=0,reworks=2)
        tasks=build_tasks(plan,self.source)
        self.assertFalse(any(t['intent']['action']=='deliver_restored_rule_asset' for t in plan['turns']))
        self.assertFalse(any(t['intent'].get('rule_event', {}).get('cleanup') for t in plan['turns']))
        self.assertTrue(any(r['ended_at'] is None for r in plan['rules']))

    def test_delta_only_context(self):
        task=dict(backend='flux',operation='local_edit',layout='product',generation_requirements={},
            desired_description='UNRELATED',desired_attributes={'color':'OLD_COLOR'},
            changes=[dict(field='color',before='OLD_COLOR',after='yellow')],
            references=[dict(label='<Image 1>',kind='image',role='base to edit',description='OLD_STATE')])
        context=writer_input(task)
        self.assertIn('yellow',context)
        self.assertNotIn('OLD_',context)
        self.assertNotIn('UNRELATED',context)
        self.assertNotIn('before',context)

    def test_prompt_validation_retries_and_truncation(self):
        task=dict(backend='flux',operation='local_edit',layout='product',generation_requirements={},
            changes=[dict(field='color',after='red')],references=[dict(label='<Image 1>',kind='image',role='base to edit')])
        good='Edit <Image 1>. Change the bag to red. Preserve everything else.'
        request=Mock(side_effect=[('Use <Image 9>.','stop'),(good,'stop')])
        self.assertEqual(rewrite_placeholder(task,request,seed=10),good)
        self.assertNotIn('<Image 9>',request.call_args_list[1].args[0])
        with self.assertRaises(RuntimeError):
            rewrite_placeholder(task,Mock(return_value=(good,'length')),max_attempts=1)

    def test_flux_edit_binds_an_omitted_known_source_without_model_retry(self):
        task=dict(backend='flux',operation='local_edit',layout='product',
            changes=[dict(field='color',after='blue')],
            references=[dict(label='<Image 1>',kind='image',role='base to edit')])
        request=Mock(return_value=('A blue bag on a neutral studio backdrop.', 'stop'))
        result=rewrite_placeholder(task,request,max_attempts=1)
        self.assertTrue(result.startswith('Edit <Image 1>.'))
        self.assertIn('blue', result)
        self.assertIn('except the requested edits and current output rendering requirements', result)
        self.assertEqual(request.call_count,1)

    def test_flux_reference_generation_binds_studio_presentation_to_identity(self):
        task=dict(backend='flux',operation='reference_generation',layout='product',changes=[],
            presentation=dict(season='summer',seasonal_accents='one small lush tree'),
            references=[dict(label='<Image 1>',kind='image',role='identity/design reference')])
        text='A studio product photograph of a leather bag on a neutral backdrop with a small lush tree.'
        request=Mock(return_value=(text,'stop'))
        result=rewrite_placeholder(task,request,max_attempts=1)
        self.assertIn('<Image 1>',result)
        self.assertIn(text,result)
        self.assertEqual(request.call_count,1)

    def test_flux_source_binding_still_rejects_missing_changes_and_declines(self):
        task=dict(backend='flux',operation='local_edit',layout='product',
            changes=[dict(field='color',after='blue')],
            references=[dict(label='<Image 1>',kind='image',role='base to edit')])
        for text in ('A red bag on a neutral studio backdrop.', 'CANNOT_FULFILL', '', '{"prompt": "blue bag"}'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                rewrite_placeholder(task,Mock(return_value=(text,'stop')),max_attempts=1)

    def test_flux_source_binding_keeps_monochrome_priority(self):
        task=dict(backend='flux',operation='local_edit',layout='product',
            changes=[dict(field='shape',after='triangular')],render_state=dict(color_mode='black_and_white'),
            references=[dict(label='<Image 1>',kind='image',role='base to edit')])
        with self.assertRaises(RuntimeError):
            rewrite_placeholder(task,Mock(return_value=('A blue triangular bag in black and white.','stop')),max_attempts=1)

    def test_flux_text_generation_does_not_invent_an_edit_source(self):
        task=dict(backend='flux',operation='text_generation',layout='product',changes=[],references=[])
        text='A blue bag on a neutral studio backdrop.'
        self.assertEqual(rewrite_placeholder(task,Mock(return_value=(text,'stop')),max_attempts=1),text)

    def test_prompt_hashes_and_checkpoint_resume(self):
        plan=deepcopy(self.plan);plan['realization']=dict(status='complete',rewrite_seed=10)
        calls=Mock(side_effect=writer)
        with tempfile.TemporaryDirectory() as tmp:
            rewrite_run(plan,self.source,tmp,writer=calls,progress=False)
            first=calls.call_count
            rewrite_run(plan,self.source,tmp,writer=calls,progress=False)
            self.assertEqual(calls.call_count,first)
            for row in plan['placeholders']:
                task=row['generation']
                if task['backend']:
                    self.assertEqual(task['writer_input_hash'],hashlib.sha256(writer_input(task).encode()).hexdigest())
            self.assertEqual(plan['prompt_rewriting']['status'],'complete')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                rewrite_run(plan,self.source,tmp,writer=Mock(side_effect=RuntimeError('interrupted')),force=True,progress=False)
            self.assertEqual(plan['realization']['status'],'complete')
            self.assertEqual(plan['prompt_rewriting']['status'],'failed')

    def test_extensions_persist_through_plan_and_expert_contract(self):
        source=deepcopy(self.source)
        source['assets'][0]['attributes']['distinctive_motif']='white bird emblem'
        source['assets'][0]['attributes']['hair_color']='iridescent silver'
        normalized=observation(source)
        asset=normalized['assets'][0]
        self.assertIn('distinctive_motif',asset['attribute_extensions'])
        plan=build_plan(source,seed=0,reworks=0,modules=['direct'],extra_behaviors=False)
        self.assertEqual(plan['target_assets'][asset['id']]['attributes']['distinctive_motif'],'white bird emblem')
        task=next(t for t in build_tasks(plan,source).values() if t['asset_ids']==[asset['id']])
        self.assertIn('white bird emblem',writer_input(task))
        self.assertIn('iridescent silver',writer_input(task))

    def test_pose_and_composition_inheritance(self):
        requirements=resolve_requirements('image','asset',['person'],requested=dict(pose='seated in profile'))
        self.assertEqual(resolve_requirements('image','asset',['person'],parent=requirements),requirements)
        self.assertEqual(resolve_requirements('image','asset',['object']),{})


if __name__=='__main__':
    unittest.main()
