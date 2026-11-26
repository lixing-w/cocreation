from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from tools.candidate_prompter import rewrite_candidates
from tests.test_single_clip_candidates import task,prompt,REQUEST

class CandidatePrompterTests(unittest.TestCase):
    def test_joint_call_keeps_alternatives_separate_and_hides_planner_fields(self):
        calls=[]
        def request(user,**options):
            calls.append((user,options))
            return json.dumps({'prompts':[dict(output_position=i,prompt=prompt(finish))
                for i,finish in enumerate(('dark','beige'),1)]}),'stop'
        result=rewrite_candidates([task(1),task(2)],request)
        self.assertEqual(result,{'[[video_0002]]':prompt('dark'),'[[video_0003]]':prompt('beige')})
        self.assertEqual(len(calls),1)
        self.assertNotIn('PLANNER_SECRET',str(calls))
        self.assertNotIn('Write only output 1 of 2',calls[0][1]['system'])
        self.assertEqual(json.loads(calls[0][0].split('\n',1)[1])['current_user_request'],REQUEST)

    def test_image_alternatives_use_one_joint_call_and_image_feedback(self):
        tasks=[task(1),task(2)]
        for index,t in enumerate(tasks,1):
            t.update(backend='h3_image',kind='image',operation='text_generation',references=[],placeholder=f'[[image_{index:04d}]]')
        calls=[]
        def request(user,**options):
            calls.append((user,options))
            values=('CANNOT_FULFILL','A cyan horse passing by.') if len(calls)==1 else ('A white horse passing by.','A cyan horse passing by.')
            return json.dumps(dict(prompts=[dict(output_position=i,prompt=value) for i,value in enumerate(values,1)])),'stop'
        result=rewrite_candidates(tasks,request)
        self.assertEqual(result,{'[[image_0001]]':'A white horse passing by.','[[image_0002]]':'A cyan horse passing by.'})
        self.assertEqual(len(calls),2)
        self.assertIn('one complete image',calls[1][0])
        self.assertNotIn('[Shot 1]',calls[1][0])
        self.assertNotIn('PLANNER_SECRET',str(calls))

    def test_group_system_reused_when_reference_positions_subjects_and_formats_change(self):
        calls=[]
        def request(user,**options):
            payload=json.loads(user.split('\n',1)[1])
            calls.append((payload,options['system']))
            prompts=[]
            for output,text in zip(payload['outputs'],('A white horse passing by.','A cyan horse passing by.')):
                if output['references']:
                    text='Use <Picture 1>. '+text
                prompts.append(dict(output_position=output['output_position'],prompt=text))
            return json.dumps(dict(prompts=prompts)),'stop'
        first, second = task(1), task(2)
        for index,item in enumerate((first,second),1):
            item.update(backend='h3_image',kind='image',operation='text_generation',references=[],
                        placeholder=f'[[image_{index:04d}]]')
        # Keep the cache profile mixed in both calls, while switching which
        # option carries the reference and changing its subject and canvas.
        first['references'] = [dict(kind='image',label='<Picture 1>',subjects=['saddle A'])]
        second_run = deepcopy([first,second])
        second_run[0]['references'] = []
        second_run[1]['references'] = [dict(kind='image',label='<Picture 1>',subjects=['saddle B'])]
        second_run[1]['render_state']['aspect_ratio'] = '1:1'

        rewrite_candidates([first,second],request)
        rewrite_candidates(second_run,request)
        self.assertEqual(calls[0][1],calls[1][1])
        for subject, (payload,system), index in zip(('saddle A','saddle B'),calls,(0,1)):
            self.assertNotIn(subject,system)
            self.assertEqual(payload['outputs'][index]['references'][0]['subjects'],[subject])
        self.assertEqual(calls[1][0]['outputs'][1]['output_format']['orientation'],'square')

    def test_invalid_group_retries_every_alternative(self):
        for failure in ('duplicate','order','multishot'):
            calls=[]
            def request(user,**options):
                calls.append(user)
                items=[dict(output_position=i,prompt=prompt(f)) for i,f in enumerate(('dark','beige'),1)]
                if len(calls)==1:
                    if failure=='duplicate':items[1]['prompt']=items[0]['prompt']
                    elif failure=='order':items.reverse()
                    else:items[0]['prompt']=prompt('dark')+' [Shot 2] At 00:02.000, beige.'
                return json.dumps(dict(prompts=items)),'stop'
            with self.subTest(failure=failure):
                result=rewrite_candidates([task(1),task(2)],request)
                self.assertEqual(len(calls),2)
                self.assertEqual(result['[[video_0003]]'],prompt('beige'))

    def test_production_rewrite_groups_actual_video_tasks_and_reuses_checkpoint(self):
        from scripts.dialogue_generation.planner import Planner
        from scripts.dialogue_generation.asset_variations import AssetSpec
        from scripts.dialogue_generation.rewrite_prompts import rewrite_run
        from tests.test_generation_prompts import ROOT,writer
        source=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        p=Planner(source,seed=0,reworks=0,extra_behaviors=False,cross_turn_rules=False)
        desired=AssetSpec(p.assets['bag']);alternate=p.alternate('bag',desired)
        p.emit('user','request_candidates','Make two short clips of the bag, dark first and beige second.',assets=['bag'])
        tokens=[p.asset_media('bag',description=desired,kind='video'),
                p.asset_media('bag',description=alternate,kind='video',desired=False)]
        p.emit('assistant','deliver_candidates','Here are the clips.',assets=['bag'],outputs=tokens)
        p.adopt('bag',tokens[0],choice_required=True)
        plan=p.run();plan['realization']=dict(status='complete',rewrite_seed=10)
        for turn in plan['turns']:
            if not turn.get('content'):turn['content']='Please make this.' if turn['role']=='user' else 'Here it is.'
        class Writer:
            model='candidate-integration'
            def __init__(self):self.groups=[]
            def generation_request(self,user,**options):return writer(user,**options)
            def generation_candidates(self,tasks,**options):
                self.groups.append([t['placeholder'] for t in tasks])
                return {t['placeholder']:prompt('dark' if i==0 else 'beige') for i,t in enumerate(tasks)}
        model=Writer()
        with tempfile.TemporaryDirectory() as temp:
            rewrite_run(plan,source,temp,writer=model,progress=False)
            self.assertEqual(model.groups.count(tokens),1)
            rows={r['placeholder']:r for r in plan['placeholders']}
            self.assertEqual(rows[tokens[0]]['generation']['prompt'],prompt('dark'))
            self.assertEqual(rows[tokens[1]]['generation']['prompt'],prompt('beige'))
            count=len(model.groups)
            rewrite_run(plan,source,temp,writer=model,progress=False)
            self.assertEqual(len(model.groups),count)

    def test_production_rewrite_groups_actual_image_tasks_and_reuses_checkpoint(self):
        from scripts.dialogue_generation.planner import Planner
        from scripts.dialogue_generation.asset_variations import AssetSpec
        from scripts.dialogue_generation.rewrite_prompts import rewrite_run
        from tests.test_generation_prompts import ROOT,writer
        source=json.loads((ROOT/'scripts/dialogue_generation/examples/tokyo_vlog.json').read_text())
        p=Planner(source,seed=0,reworks=0,extra_behaviors=False,cross_turn_rules=False)
        desired=AssetSpec(p.assets['bag']);alternate=p.alternate('bag',desired)
        p.emit('user','request_candidates','Make two images of the bag, dark first and beige second.',assets=['bag'])
        tokens=[p.asset_media('bag',description=desired,kind='image'),
                p.asset_media('bag',description=alternate,kind='image',desired=False)]
        p.emit('assistant','deliver_candidates','Here are the clips.',assets=['bag'],outputs=tokens)
        p.adopt('bag',tokens[0],choice_required=True)
        plan=p.run();plan['realization']=dict(status='complete',rewrite_seed=10)
        for turn in plan['turns']:
            if not turn.get('content'):turn['content']='Please make this.' if turn['role']=='user' else 'Here it is.'
        class Writer:
            model='candidate-integration'
            def __init__(self):self.groups=[]
            def generation_request(self,user,**options):return writer(user,**options)
            def generation_candidates(self,tasks,**options):
                self.groups.append([t['placeholder'] for t in tasks])
                return {t['placeholder']:('A dark bag.' if i==0 else 'A beige bag.') for i,t in enumerate(tasks)}
        model=Writer()
        with tempfile.TemporaryDirectory() as temp:
            rewrite_run(plan,source,temp,writer=model,progress=False)
            self.assertEqual(model.groups.count(tokens),1)
            rows={r['placeholder']:r for r in plan['placeholders']}
            self.assertEqual(rows[tokens[0]]['generation']['prompt'],'A dark bag.')
            self.assertEqual(rows[tokens[1]]['generation']['prompt'],'A beige bag.')
            count=len(model.groups)
            rewrite_run(plan,source,temp,writer=model,progress=False)
            self.assertEqual(len(model.groups),count)

if __name__=='__main__':unittest.main()
