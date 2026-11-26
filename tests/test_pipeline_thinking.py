"""Explicit Qwen thinking controls and the pipeline's non-thinking default."""
from pathlib import Path
import unittest
from unittest.mock import Mock
from cocreation import VLLMQwen
from tools.dialogue_realizer import QwenRealizer
from tools.dialogue_verifier import build_prompt

ROOT=Path(__file__).resolve().parents[1]

class ThinkingPipelineTests(unittest.TestCase):
    def test_vllm_wire_request_enables_thinking_with_final_output_budget(self):
        request=VLLMQwen(enable_thinking=True)('Resolve attribute conflicts.',[],max_new_tokens=4096,dry_run=True)['request']
        self.assertTrue(request['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(request['max_tokens'],4096)
        self.assertEqual(request['thinking_token_budget'],2048)

    def test_dialogue_and_generation_writer_forward_thinking(self):
        model=QwenRealizer(backend='vllm',model_mode='thinking',max_tokens=4096)
        model.local=Mock(return_value=dict(text='Final text.',finish_reason='stop'))
        model._generate('Dialogue.',1,system='rules',max_tokens=4096)
        self.assertTrue(model.local.call_args.kwargs['enable_thinking'])
        model.generation_request('Expert prompt.',system='Resolve conflicts.',seed=1,max_tokens=6144)
        self.assertTrue(model.local.call_args.kwargs['enable_thinking'])
        self.assertEqual(model.local.call_args.kwargs['max_new_tokens'],6144)
        self.assertEqual(model.metrics['stages']['prompt_rewriter']['requests'],1)

    def test_script_defaults_do_not_invent_a_format_rule(self):
        script=(ROOT/'run_pipeline.sh').read_text()
        self.assertNotIn('--aspect-ratio-rule',script)
        self.assertEqual(script.count('--no-enable-thinking'),3)
        self.assertNotIn(' --enable-thinking',script)
        self.assertIn('--max-tokens 8192',script)
        self.assertIn('--max-tokens 4096',script)
        self.assertIn('--video-steps 50',script)  # Existing original-H3 run_pipeline default.
        self.assertIn('--video-width 1344 --video-height 768 --native-video-output',script)
        self.assertNotIn('--output-width',script)
        self.assertNotIn('--output-height',script)
        self.assertNotIn('--video-two-pass',script)

    def test_live_verifier_can_evaluate_media_kind_and_explicit_exception(self):
        brief=dict(speaker='user',action='request_rule_exception',draft_content_requirements=['Make this image square for this request only.'],
                   intent_facts=dict(requested_media_kind='image'),subjects=['bag'])
        prompt=build_prompt('This image only, make it square.',brief)
        self.assertIn('Requested media kind: image.',prompt)
        self.assertIn('No further clarification',prompt)
        self.assertNotIn('ROLE_NAMES',prompt)

if __name__=='__main__':unittest.main()
