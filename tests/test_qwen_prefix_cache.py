"""Actual tiny Qwen3-VL numerical tests; no downloads or GPU required.

Run with the project's torch/transformers dependencies installed. Tiny random
weights exercise real visual encoding, mRoPE, DynamicCache and generate().
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

from cocreation.qwen_prefix_cache import generate_with_prefix


class PrefixProcessor:
    """Deterministic token boundary fixture; model and caches are real."""
    chat_template = 'test system boundary'

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def tokenizer(self, text, **kwargs):
        return {'input_ids': torch.tensor([json.loads(text)])}


@unittest.skipUnless(AVAILABLE, 'Requires torch and transformers')
class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        torch.manual_seed(123)
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             rope_scaling={'rope_type':'default','mrope_section':[2,1,1]},
                             max_position_embeddings=1024),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
                               num_heads=4, patch_size=2, temporal_patch_size=2,
                               spatial_merge_size=2, out_hidden_size=32,
                               num_position_embeddings=16, deepstack_visual_indexes=[0,1]),
            image_token_id=90, video_token_id=91,
            vision_start_token_id=92, vision_end_token_id=93)
        config._commit_hash='test-resolved-revision'
        self.model = Qwen3VLForConditionalGeneration(config).eval()
        self.model.generation_config.pad_token_id=0
        self.model.generation_config.eos_token_id=None
        self.processor = PrefixProcessor()
        self.request = dict(model_id='test-random-qwen', runtime={'revision':None}, options=dict(
            system_prompt='[1, 2, 3]', system_cache_path=str(Path(self.tmp.name)/'system.kvcache'),
            max_new_tokens=3))

    def inputs(self, kind='image', offset=0):
        prefix = json.loads(self.request['options']['system_prompt'])
        if kind == 'image':
            suffix=[4,92,90,90,90,90,93,5,6]
            media=dict(pixel_values=torch.randn(16,24)+offset, image_grid_thw=torch.tensor([[1,4,4]]))
        elif kind == 'video':
            # Official video format: temporal groups each have a timestamp text
            # and their own vision-start/end region. Spatial merger => 4 tokens/frame.
            suffix=[4,7,92,91,91,91,91,93,8,92,91,91,91,91,93,5,6]
            media=dict(pixel_values_videos=torch.randn(32,24)+offset,
                       video_grid_thw=torch.tensor([[2,4,4]]))
        elif kind == 'mixed':
            suffix=[4,92,90,90,90,90,93,7,92,91,91,91,91,93,5,6]
            media=dict(pixel_values=torch.randn(16,24)+offset,
                       image_grid_thw=torch.tensor([[1,4,4]]),
                       pixel_values_videos=torch.randn(16,24)+offset,
                       video_grid_thw=torch.tensor([[1,4,4]]))
        else:
            suffix=[4,5,6];media={}
        ids=torch.tensor([prefix+suffix])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), **media)

    def compare(self, inputs, state):
        baseline=[];cached=[]
        def capture(target):
            def hook(module, args, kwargs, output):
                target.append(output.logits[:, -1].detach().clone())
            return hook
        with torch.inference_mode():
            handle=self.model.register_forward_hook(capture(baseline),with_kwargs=True)
            torch.manual_seed(456)
            expected=self.model.generate(**inputs,max_new_tokens=3,do_sample=True,
                                         temperature=.8,top_p=.95,top_k=50,use_cache=True)
            handle.remove()
            handle=self.model.register_forward_hook(capture(cached),with_kwargs=True)
            torch.manual_seed(456)
            actual, info=generate_with_prefix(self.model,self.processor,inputs,self.request,state)
            handle.remove()
        self.assertTrue(torch.equal(expected,actual))
        # Last three forwards are exactly the three decoding decisions; previous
        # forwards build the system and/or media prefix without decoding a token.
        torch.testing.assert_close(torch.stack(baseline),torch.stack(cached[-3:]),rtol=1e-4,atol=1e-6)
        self.assertIsNone(self.model.model.rope_deltas)
        self.assertTrue(all(pair[0].shape[-2]==3 for pair in state['system_prefix_payload']['layers']))
        return info

    def test_image_video_text_reuse_and_disk_reload(self):
        state={}
        self.assertEqual(self.compare(self.inputs(),state)['status'],'built')
        original=copy.deepcopy(state['system_prefix_payload'])
        self.assertEqual(self.compare(self.inputs('image',offset=5),state)['status'],'memory')
        self.assertEqual(self.compare(self.inputs('video'),state)['status'],'memory')
        self.assertEqual(self.compare(self.inputs('text'),state)['status'],'memory')
        self.assertEqual(self.compare(self.inputs('mixed'),state)['status'],'memory')
        for old,new in zip(original['layers'],state['system_prefix_payload']['layers']):
            torch.testing.assert_close(old[0],new[0],rtol=0,atol=0)
            torch.testing.assert_close(old[1],new[1],rtol=0,atol=0)
        self.assertEqual(self.compare(self.inputs('video'),{})['status'],'disk')

    def test_reload_in_new_model_instance(self):
        inputs=self.inputs();self.compare(inputs,{})
        previous=self.model
        self.model=Qwen3VLForConditionalGeneration(copy.deepcopy(previous.config)).eval()
        self.model.load_state_dict(previous.state_dict())
        self.model.generation_config=copy.deepcopy(previous.generation_config)
        self.assertEqual(self.compare(inputs,{})['status'],'disk')

    def test_invalidation_and_corrupt_file(self):
        state={};self.compare(self.inputs(),state)
        self.request['options']['system_prompt']='[1, 2, 9]'
        self.assertEqual(self.compare(self.inputs(),state)['status'],'built')
        Path(self.request['options']['system_cache_path']).write_bytes(b'broken')
        self.assertEqual(self.compare(self.inputs(),{})['status'],'built')
        self.model.config._commit_hash='changed-weights'
        self.assertEqual(self.compare(self.inputs(),state)['status'],'built')

    def test_prefix_mismatch_and_error_cleanup(self):
        inputs=self.inputs();inputs['input_ids'][0,0]=7
        with torch.inference_mode(),self.assertRaisesRegex(ValueError,'exact prefix'):
            generate_with_prefix(self.model,self.processor,inputs,self.request,{})
        inputs=self.inputs()
        with torch.inference_mode(),patch.object(self.model,'generate',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                generate_with_prefix(self.model,self.processor,inputs,self.request,{})
        self.assertIsNone(self.model.model.rope_deltas)
