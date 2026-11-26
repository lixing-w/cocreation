"""Current planned output, historical rule scope, and public tool-call metadata."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.dialogue_generation.formal_sequence import generation_context, generation_segment, operation_segments
from scripts.dialogue_generation.planner_rules import rule_applicability
from tools.placeholder_prompter import writer_contract
from tools.dialogue_realizer import QwenRealizer
from tools.prompt_reasoner import PromptReasoner


def result():
    return dict(target_assets={'person':dict(name='person')},
        rules=[dict(id='R1',scope='global',selector={},effect=dict(aspect_ratio='landscape'),started_at=2,ended_at=None),
               dict(id='R2',scope='partial',selector=dict(kinds=['video']),effect=dict(season='fall'),started_at=4,ended_at=None),
               dict(id='R3',scope='global',selector={},effect=dict(season='winter'),started_at=20,ended_at=None)],
        placeholders=[dict(placeholder='[[image_0004]]',kind='image',asset_ids=['person'],
            render_state=dict(aspect_ratio='landscape'),introduced_at=16,
            applied_rules=[dict(id='R1',effect=dict(aspect_ratio='landscape'))])],
        turns=[dict(id=7,role='user',content='Choose the misty waterfall.'),
               dict(id=13,role='user',content='A person in blue traditional clothes holding a sword.'),
               dict(id=15,role='user',content='Keep landscape 16:9.'),
               dict(id=16,role='assistant',content='Here is the character.',output_media=['[[image_0004]]'])])


THINK=dict(intent='I will create this character image.',rules='The format applies; the video-only season does not.',
           references='This standalone character request does not need an earlier scene image.')


class PlannedPromptContextTests(unittest.TestCase):
    def test_character_contract_retains_plan_and_history_without_inventing_bindings(self):
        task=dict(operation='text_generation',backend='h3_image',kind='image',subject='person',
            render_state=dict(aspect_ratio='16:9'),references=[],
            desired_description='A person wearing traditional clothing, holding a sword.',
            desired_attributes=dict(clothing_color='blue',clothing_type='traditional clothing'),
            sequence_context=generation_context(result(),'[[image_0004]]'))
        contract=writer_contract(task)
        self.assertEqual(contract['planned_output']['desired_description'],task['desired_description'])
        self.assertEqual(contract['planned_output']['desired_attributes'],task['desired_attributes'])
        self.assertEqual(contract['references'],[])
        self.assertIn('misty waterfall',contract['visible_sequence'][0]['content'])
        self.assertNotIn('waterfall',json.dumps(contract['planned_output']))
        self.assertEqual(contract['output_subjects'],['person'])

    def test_image_excludes_video_only_rule_without_exposing_future_rule(self):
        view=rule_applicability(result(),'[[image_0004]]')
        self.assertEqual([r['id'] for r in view['effective']],['R1'])
        self.assertEqual([r['id'] for r in view['inactive']],['R2'])
        self.assertEqual(view['inactive'][0]['scope'],'videos')
        self.assertIn('fall',view['inactive'][0]['requirement'])
        self.assertIn('image',view['inactive'][0]['reason'])
        self.assertNotIn('winter',json.dumps(view))

    def test_video_applies_same_season_rule(self):
        data=result();row=data['placeholders'][0]
        row['kind']='video';row['render_state']['season']='fall'
        row['applied_rules'].append(dict(id='R2',effect=dict(season='fall')))
        view=rule_applicability(data,'[[image_0004]]')
        self.assertEqual([r['id'] for r in view['effective']],['R1','R2'])
        self.assertEqual(view['inactive'],[])

    def test_canceled_and_excepted_rules_remain_explained_but_not_enforced(self):
        data=result();data['rules'][1]['ended_at']=12
        data['placeholders'][0]['applied_rules'][0]['effect']['aspect_ratio']='1:1'
        data['placeholders'][0]['render_state']['aspect_ratio']='1:1'
        view=rule_applicability(data,'[[image_0004]]')
        self.assertEqual(view['effective'],[])
        self.assertEqual([r['id'] for r in view['inactive']],['R1','R2'])
        self.assertIn('exception',view['inactive'][0]['reason'])
        self.assertIn('ended before',view['inactive'][1]['reason'])
        self.assertIn('1:1',str(view['effective_requirements']))

    def test_assembly_keeps_canvas_part_and_excludes_new_content_part(self):
        data=result();data['rules']=data['rules'][:1]
        data['rules'][0]['effect']['season']='fall'
        view=rule_applicability(data,'[[image_0004]]',operation='concatenate')
        self.assertIn('16:9',view['effective'][0]['requirement'])
        self.assertNotIn('fall',view['effective'][0]['requirement'])
        self.assertIn('fall',view['inactive'][0]['requirement'])
        self.assertIn('assembles existing clips',view['inactive'][0]['reason'])

    def test_rules_reach_single_and_group_reasoner_requests(self):
        context=generation_context(result(),'[[image_0004]]')
        adapter=QwenRealizer.__new__(QwenRealizer)
        adapter._generate=Mock(return_value=(json.dumps(dict(think=THINK)),'stop'))
        with tempfile.TemporaryDirectory() as tmp:
            adapter.generation_reasoning(context,'Character only.','image',seed=8,output_dir=tmp)
        payload=json.loads(adapter._generate.call_args.args[0])
        self.assertEqual(payload['rule_applicability'],context['rule_applicability'])
        captured=[]
        def model(user,refs,**options):
            captured.append(json.loads(user))
            return dict(text=json.dumps(dict(think=THINK)),finish_reason='stop')
        steps=[dict(placeholder=f'[[image_{i:04d}]]',kind='image',prompt='Waterfall.',operation='text_generation',
                    references=[],sequence_context=context,latest_rules=context['latest_rules']) for i in (1,2)]
        with tempfile.TemporaryDirectory() as tmp:
            PromptReasoner(model).write_delivery(context['visible_sequence'],steps,seed=8,output_dir=tmp)
        self.assertEqual(captured[0]['generation_steps'][1]['rule_applicability'],context['rule_applicability'])

    def test_export_hides_id_and_seed_without_mutating_execution_data(self):
        attempt=dict(attempt=1,backend='h3_image',kind='image',prompt='Character.',path='out.png',references=[],
                     effective_parameters=dict(seed=8,width=5472,steps=6))
        before=deepcopy(attempt)
        segment=generation_segment(attempt,'[[image_0004]]',{})
        call=segment['tool_calls'][0]
        self.assertNotIn('id',call)
        self.assertEqual(call['arguments']['parameters'],dict(width=5472,steps=6))
        self.assertEqual(attempt,before)
        operation=dict(tool_call=dict(id='internal',name='assemble',arguments=dict(clips=['a.mp4','b.mp4'],parameters=dict(seed=10))))
        shown=operation_segments(operation)[1]['tool_calls'][0]
        self.assertNotIn('id',shown)
        self.assertNotIn('seed',shown['arguments']['parameters'])
        self.assertEqual(operation['tool_call']['arguments']['parameters']['seed'],10)


if __name__=='__main__':
    unittest.main()
