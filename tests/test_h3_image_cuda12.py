"""CUDA12 image startup cannot select CUDA13-only packaged attention kernels."""
import unittest
from types import SimpleNamespace
from cocreation.h3_image_comfy import configure_cuda12_attention


class CUDA12AttentionTests(unittest.TestCase):
    def test_only_packaged_attention_is_disabled_on_cuda12(self):
        quantize = object()
        kitchen = SimpleNamespace(int8_attention_is_available=lambda *a: True,
                                  sol_attn_is_available=lambda *a: True,
                                  quantize_int8_rowwise=quantize)
        self.assertTrue(configure_cuda12_attention('12.8', kitchen))
        self.assertFalse(kitchen.int8_attention_is_available())
        self.assertFalse(kitchen.int8_attention_is_available('cuda:0'))
        self.assertFalse(kitchen.sol_attn_is_available('cuda:0'))
        self.assertIs(kitchen.quantize_int8_rowwise, quantize)

    def test_cuda13_probes_are_preserved(self):
        probe = lambda: True
        kitchen = SimpleNamespace(int8_attention_is_available=probe, sol_attn_is_available=probe)
        self.assertFalse(configure_cuda12_attention('13.0', kitchen))
        self.assertIs(kitchen.int8_attention_is_available, probe)
        self.assertFalse(configure_cuda12_attention(None, kitchen))


if __name__ == '__main__':
    unittest.main()
