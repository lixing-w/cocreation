"""Assembly keeps existing content and records the real Python operation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.prompt_reasoner import PromptReasoner
from tools.dialogue_realizer import QwenRealizer
from tools.video_editing import edit_tool_call, concatenate
from scripts.dialogue_generation.formal_sequence import export_sequence


def assembly_plan():
    return dict(intent="I will join the existing clips in the requested order.",
                rules="I will preserve their existing content and use the requested landscape 16:9 canvas.",
                references="I will use video_0002 first and video_0003 second, retaining their audio.")


class AssemblyTools(unittest.TestCase):
    def test_reasoner_receives_python_operation_and_no_generation_prompt(self):
        model=Mock(return_value=dict(text=json.dumps(dict(think=assembly_plan())),finish_reason='stop'))
        call=edit_tool_call(dict(operation='concatenate',clips=[dict(placeholder='b'),dict(placeholder='a')]),
                            dict(a='a.mp4',b='b.mp4'),'joined.mp4')
        with tempfile.TemporaryDirectory() as temp:
            report=PromptReasoner(model).write_operation([dict(role='user',content='Join B then A.',private='secret')],
                call,[dict(requirement='Use landscape 16:9.')],seed=1,output_dir=temp,
                source_formats=[dict(placeholder='b',aspect_ratio='9:16'),dict(placeholder='a',aspect_ratio='16:9')])
        payload=json.loads(model.call_args.args[0])
        self.assertEqual(payload['tool_call']['name'],'tools.video_editing.concatenate')
        self.assertEqual(payload['tool_call']['arguments']['clips'],['b.mp4','a.mp4'])
        self.assertNotIn('size',payload['tool_call']['arguments'])
        self.assertEqual(payload['source_formats'][0]['aspect_ratio'],'9:16')
        self.assertNotIn('generation_prompt',payload)
        self.assertNotIn('generation_prompt',report)
        self.assertNotIn('secret',json.dumps(payload))
        self.assertIn('no new media generation',model.call_args.kwargs['system_prompt'])
        self.assertFalse(model.call_args.kwargs['enable_thinking'])

    def test_realizer_uses_operation_reasoner_stage(self):
        model=QwenRealizer.__new__(QwenRealizer)
        model._generate=Mock(return_value=(json.dumps(dict(think=assembly_plan())),'stop'))
        with tempfile.TemporaryDirectory() as temp:
            model.operation_reasoning(dict(visible_sequence=[],latest_rules=[]),
                dict(name='tools.video_editing.concatenate',arguments={}),seed=1,output_dir=temp)
        self.assertEqual(model._generate.call_args.kwargs['stage'],'operation_reasoner')
        self.assertEqual(model._generate.call_args.kwargs['max_tokens'],3072)

    def test_first_spoken_reply_precedes_python_call_inside_think(self):
        opening="This turn you only asked me to join existing clips, so I'm not generating new media and the rule requiring a winter background for new media does not apply."
        result=dict(turns=[dict(id=1,role='user',content='Join B then A.'),dict(id=2,role='assistant',
            content='Here is the assembled video.',output_media=['[[video_0004]]'],generation_opening=opening)])
        call=dict(id='video_0004_python',**edit_tool_call(dict(operation='concatenate',
            clips=[dict(placeholder='b'),dict(placeholder='a')],output_size=[1344,768]),
            dict(a='a.mp4',b='b.mp4'),'joined.mp4'))
        with tempfile.TemporaryDirectory() as temp:
            artifact=export_sequence(temp,result,{'[[video_0004]]':dict(status='complete',kind='video',
                operation='concatenate',path='joined.mp4',tool_call=call,
                operation_reasoner=dict(think='I will join B then A.'))})
        segments=artifact['turns'][1]['segments']
        self.assertEqual(segments[0]['text'],opening)
        self.assertEqual(segments[1]['tool_calls'],[call])
        self.assertTrue(segments[1]['text'].startswith('<think>'))
        self.assertLess(segments[1]['text'].index('<tool_call>'),segments[1]['text'].index('</think>'))
        self.assertNotIn('prompt',call['arguments'])
        self.assertNotIn('minimax-h3',json.dumps(segments))

    def test_failed_python_operation_is_retained_without_publishing_media(self):
        result=dict(turns=[dict(id=1,role='user',content='Join these clips.'),dict(id=2,role='assistant',
            content='The assembled result is ready.',output_media=['[[video_0004]]'])])
        call=dict(name='tools.video_editing.concatenate',arguments=dict(clips=['a.mp4','b.mp4']))
        with tempfile.TemporaryDirectory() as temp:
            artifact=export_sequence(temp,result,{'[[video_0004]]':dict(status='failed',tool_call=call)},status='failed')
        segments=artifact['turns'][1]['segments']
        self.assertEqual(segments[0]['tool_calls'],[call])
        self.assertFalse(any(segment['type']=='media' for segment in segments))


if __name__=='__main__':unittest.main()
