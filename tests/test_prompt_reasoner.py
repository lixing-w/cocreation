import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from tools.prompt_reasoner import PromptReasoner, SYSTEM
from tools.dialogue_realizer import QwenRealizer
from scripts.dialogue_generation.formal_sequence import export_sequence


def plan():
    return dict(intent="Let's analyze what the user is asking for. I will make one new fabric city image, showing the requested textile structure. I am preparing the first alternative independently.",
        rules="Now let's look at which cross-turn rules are effective. I have no active format requirement. The user hasn't specified an aspect ratio, so I'll use the defaults: 1:1 for images and landscape 16:9 for videos. I will use the image default here.",
        references="Now let's gather visual references for this generation. I am creating a new identity rather than editing an earlier image. I do not need a previous source because this request establishes the first city design.")


class PromptReasonerTests(unittest.TestCase):
    def test_prospective_plan_uses_final_prompt_but_excludes_private_state(self):
        calls=[]
        def model(user,refs,**options):
            calls.append((json.loads(user),refs,options))
            return dict(text=json.dumps(dict(think=plan())),finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            report=PromptReasoner(model).write([dict(role='user',content='Make a fabric city.',media=[],hidden='PLANNER_SECRET')],
                'FINAL_PROMPT_SECRET',[],kind='image',seed=1,output_dir=temp)
            self.assertEqual(calls[0][0]['generation_prompt']['text'],'FINAL_PROMPT_SECRET')
            self.assertNotIn('PLANNER_SECRET',json.dumps(calls,default=str))
            self.assertEqual(calls[0][0]['generation_prompt']['references'],[])
            self.assertFalse(calls[0][2]['enable_thinking'])
            self.assertEqual(calls[0][2]['max_new_tokens'],3072)
            self.assertEqual(report['generation_prompt'],'FINAL_PROMPT_SECRET')
            self.assertEqual(report['think'],'\n\n'.join(plan().values()))
            self.assertIn('about 150 words total',SYSTEM)

    def test_rule_trace_uses_the_planned_aspect_source(self):
        from scripts.dialogue_generation.planner_rules import rule_applicability

        token = '[[image_0001]]'
        for state, source, ratio in [({}, 'system_default', '1:1'),
                                     ({'aspect_ratio': 'portrait'}, 'user_override', '9:16')]:
            result = dict(placeholders=[dict(placeholder=token, kind='image', render_state=state)])
            applicability = rule_applicability(result, token)
            self.assertEqual(applicability['aspect_ratio'], dict(source=source, ratio=ratio))
            data = plan()
            data['rules'] = 'I will use the governing aspect ratio.'
            model = Mock(return_value=dict(text=json.dumps(dict(think=data)), finish_reason='stop'))
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temp:
                report = PromptReasoner(model).write([], 'city', applicability['effective_requirements'],
                    kind='image', rule_applicability=applicability, seed=1, output_dir=temp)
                self.assertEqual('system default for aspect ratio' in report['steps']['rules'],
                                 source == 'system_default')
                self.assertEqual(json.loads(model.call_args.args[0])['rule_applicability'], applicability)

    def test_future_prompt_composition_is_not_rejected_as_retrospective(self):
        for sentence in ['I will construct the video prompt to integrate these subjects.',
                         'I will construct a video prompt that integrates these subjects into a cohesive narrative shot.']:
            data=plan();data['intent']+=' '+sentence
            model=Mock(return_value=dict(text=json.dumps(dict(think=data)),finish_reason='stop'))
            with self.subTest(sentence=sentence),tempfile.TemporaryDirectory() as temp:
                report=PromptReasoner(model).write([], 'city', [],seed=1,output_dir=temp)
                self.assertIn(sentence,report['think'])
                self.assertEqual(model.call_count,1)

    def test_reasoning_length_is_prompt_guidance_without_runtime_rejection(self):
        from tools.prompt_reasoner import SCHEMA
        for prose in ('I preserve the requested design.', 'I preserve the requested design. '*180):
            data=dict.fromkeys(('intent','rules','references'),prose)
            model=Mock(return_value=dict(text=json.dumps(dict(think=data)),finish_reason='stop'))
            with self.subTest(prose=prose[:30]),tempfile.TemporaryDirectory() as temp:
                report=PromptReasoner(model).write([], 'city', [],seed=1,output_dir=temp)
                self.assertEqual(model.call_count,1)
                self.assertTrue(all(step.endswith(prose.strip()) for step in report['steps'].values()))
        for field in SCHEMA['properties']['think']['properties'].values():
            self.assertNotIn('maxLength',field)

    def test_style_is_instructed_without_lexical_rejection_gates(self):
        data=dict(intent='My next step is to compose the requested image prompt.',
                  rules='The requested format is square.',
                  references='This request establishes a new design without a visual source.')
        model=Mock(return_value=dict(text=json.dumps(dict(think=data)),finish_reason='stop'))
        with tempfile.TemporaryDirectory() as temp:
            report=PromptReasoner(model).write([], 'city', [],seed=1,output_dir=temp)
        self.assertEqual(model.call_count,1)
        for stage,opening in zip(('intent','rules','references'),(
                "Let's analyze what the user is asking for.",
                "Now let's look at which cross-turn rules are effective.",
                "Now let's gather visual references for this generation.")):
            self.assertEqual(report['steps'][stage],opening+' '+data[stage])

    def test_invalid_structure_still_rejected(self):
        malformed=[dict(intent='',rules='rules',references='refs'),
                   dict(intent='<think>content</think>',rules='rules',references='refs'),
                   dict(intent='intent',rules='rules'),
                   dict(intent='intent',rules='rules',references=None)]
        for data in malformed:
            model=Mock(return_value=dict(text=json.dumps(dict(think=data)),finish_reason='stop'))
            with self.subTest(data=data),tempfile.TemporaryDirectory() as temp,self.assertRaises(RuntimeError):
                PromptReasoner(model).write([], 'city', [],seed=1,output_dir=temp)

    def test_malformed_response_gets_feedback(self):
        calls=[]
        def model(user,refs,**kw):
            calls.append(user)
            data=plan()
            if len(calls)==1:data['intent']=''
            return dict(text=json.dumps(dict(think=data)),finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            PromptReasoner(model).write([], 'city', [],seed=1,output_dir=temp)
        self.assertEqual(len(calls),2)
        self.assertIn('Every step needs plain prose',calls[1])

    def test_exact_prompt_and_actual_parameters_follow_closed_think(self):
        result=dict(turns=[dict(id=1,role='user',content='Make a city.'),dict(id=2,role='assistant',
            content='Here is the city. [[image_0001]]',output_media=['[[image_0001]]'])])
        parameters=dict(seed=17,width=4096,height=4096,steps=6,guidance_scale=1.,sampler='euler',scheduler='simple')
        attempts=[dict(attempt=i,prompt='EXACT prompt '+str(i),kind='image',path=str(i)+'.png',backend='h3_image',
            effective_parameters=parameters,references=[],reasoner=dict(think='\n\n'.join(plan().values())) if i==1 else {},
            critique=dict(thinking='I see the output.',decision='rejected' if i==1 else 'accepted',reply='I will retry.')) for i in (1,2)]
        with tempfile.TemporaryDirectory() as temp:
            official=export_sequence(temp,result,{'[[image_0001]]':dict(status='complete',attempts=attempts)})
            segments=official['turns'][1]['segments']
            calls=[s for s in segments if s.get('tool_calls')]
            self.assertEqual(len(calls),2)
            for i,s in enumerate(calls,1):
                self.assertEqual(s['tool_calls'][0]['name'],'minimax-h3-image')
                self.assertEqual(s['tool_calls'][0]['arguments']['prompt'],'EXACT prompt '+str(i))
                self.assertEqual(s['tool_calls'][0]['arguments']['parameters'],{k:v for k,v in parameters.items() if k!='seed'})
                self.assertNotIn('id',s['tool_calls'][0])
                self.assertEqual(s['type'],'tool_call')
                self.assertTrue(s['text'].startswith('<tool_call>'))
                self.assertNotIn('<think>',s['text'])
                before=segments[segments.index(s)-1]
                self.assertEqual(before['type'],'thinking')
                self.assertTrue(before['text'].endswith('</think>'))
                self.assertNotIn('<tool_call>',before['text'])
            self.assertIn('asking for',segments[segments.index(calls[0])-1]['text'])
            self.assertFalse(any(s['type']=='generation_prompt' for s in segments))
            self.assertEqual(official['version'],3)
            text=(Path(temp)/'dialogue.md').read_text()
            self.assertLess(text.index('EXACT prompt 1'),text.index('1.png'))

    def test_realizer_uses_expanded_budget_without_native_thinking(self):
        model=QwenRealizer.__new__(QwenRealizer)
        model._generate=Mock(return_value=(json.dumps(dict(think=plan())),'stop'))
        context=dict(visible_sequence=[dict(role='user',content='Make a city.',media=[])],latest_rules=[],output_position=1,output_count=1)
        with tempfile.TemporaryDirectory() as temp:
            model.generation_reasoning(context,'FINAL_PROMPT_SECRET','image',seed=1,output_dir=temp)
        self.assertEqual(model._generate.call_args.kwargs['stage'],'prompt_reasoner')
        self.assertEqual(model._generate.call_args.kwargs['max_tokens'],3072)
        self.assertEqual(json.loads(model._generate.call_args.args[0])['generation_prompt']['text'],'FINAL_PROMPT_SECRET')

    def test_reasoner_overrides_native_thinking_mode(self):
        model=QwenRealizer.__new__(QwenRealizer)
        model.backend='vllm';model.model_mode='thinking';model.local=Mock(return_value=dict(text='{}',finish_reason='stop'))
        model.sampling=dict(temperature=.2,top_p=.95,top_k=50)
        model.context_tokens=32768;model.metrics=dict(prompt_tokens=0,completion_tokens=0)
        model._generate_once('request',1,system='system',max_tokens=3072,stage='prompt_reasoner',response_schema={})
        self.assertFalse(model.local.call_args.kwargs['enable_thinking'])

if __name__=='__main__':unittest.main()
