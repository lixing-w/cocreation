"""Start isolated image ComfyUI without CUDA-13-only attention on cu128."""
import logging
from pathlib import Path
import runpy
import sys


def configure_cuda12_attention(cuda_version, kitchen):
    """Checkpoint attention metadata must not bypass the selected SDPA backend.

    Kitchen's packaged attention kernels require CUDA 13, while its Triton
    quantization kernels remain usable on CUDA 12. Disable only those attention
    availability probes so Comfy falls back to --use-pytorch-cross-attention.
    """
    if cuda_version is None or int(cuda_version.split('.')[0]) >= 13:
        return False
    kitchen.int8_attention_is_available = lambda *args, **kwargs: False
    kitchen.sol_attn_is_available = lambda *args, **kwargs: False
    return True


def main():
    source = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(source.parent))
    # Comfy selects its allocator before importing torch. Preserve that order;
    # importing torch first makes its allocator disagree with the later setting.
    import comfy.options
    comfy.options.enable_args_parsing()
    import cuda_malloc  # noqa: F401
    import torch
    import comfy_kitchen
    if configure_cuda12_attention(torch.version.cuda, comfy_kitchen):
        logging.warning('H3 Image cu128 compatibility: using PyTorch SDPA instead of packaged CUDA13 attention.')
    runpy.run_path(str(source), run_name='__main__')


if __name__ == '__main__':
    main()
