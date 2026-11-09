"""Lightweight FreeVideo CLI contract. No ML imports or setup during planning."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import sys

from .storage import scratch_root

FREEVIDEO_REVISION = '7c3a03c1a93580454f71628b78ac3df9d9e1880e'
VDN_REVISION = '30b6b380c2482f3519469350810c2955d8847fd9'
COMFY_REVISION = '421a1c245c682c04d4325ba365f40c834c66f5b0'
DIFFUSERS_REVISION = '3a2f35d4efa4c059c8bfb3bc0d6c906264895c81'
TORCH_VERSION = '2.10.0+cu128'
ENCODER = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
MODEL_REPO = 'OpenVDN/vdn-minimax-h3-edge'
PATCH_VERSION = 1


@dataclass(frozen=True)
class VDNRuntime:
    root: str = field(default_factory=lambda: str(scratch_root() / 'runtime/vdn'))
    python: str = sys.executable
    auto_setup: bool = True
    local_files_only: bool = False
    accept_model_license: bool = True
    # GiB capacities are capped again by FreeVideo against detected/free memory.
    vram_gib: float = 96.0
    ram_gib: float = 64.0
    gpu_reserve_gib: float = 3.0
    ram_reserve_gib: float = 8.0
    attention: str = 'torch-flash'
    timeout: float = 7200.0

    def __post_init__(self):
        for name in ('vram_gib', 'ram_gib', 'gpu_reserve_gib', 'ram_reserve_gib', 'timeout'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        if self.gpu_reserve_gib >= self.vram_gib or self.ram_reserve_gib >= self.ram_gib:
            raise ValueError('Memory reserves must be smaller than capacities')
        if self.attention not in ('torch-flash', 'cudnn'):
            raise ValueError('cu128 adapter supports torch-flash or cudnn attention')


def media_request(request):
    return {'version': 1, 'references': [dict(path=r['path'], kind=r['kind'],
        **({'sound': False} if r['kind'] == 'video' else {})) for r in request['references']]}


def generation_command(request, job, installation):
    rt, opts = request['runtime'], request['options']
    return [installation['python'], '-m', 'freevideo_engine.cli', 'generate',
        '--prompt-file', str(job / 'prompt.txt'), '--media', str(job / 'media.json'),
        '--cache', installation['cache'], '--base', installation['base'],
        '--checkpoint', installation['checkpoint'],
        '--encoder-root', installation['comfy_root'], '--encoder-python', installation['python'],
        '--model-paths', installation['model_paths'], '--encoder', ENCODER,
        '--out', str(job / 'video.mp4'), '--width', str(opts['width']), '--height', str(opts['height']),
        '--frames', str(opts['frames']), '--seed', str(opts['seed']),
        '--two-pass' if opts['two_pass'] else '--no-two-pass',
        '--base-steps', '8', '--refine-steps', '3', '--upscale-factor', '2',
        '--attention', rt['attention'], '--vram-gib', str(rt['vram_gib']),
        '--ram-gib', str(rt['ram_gib']), '--gpu-reserve-gib', str(rt['gpu_reserve_gib']),
        '--ram-reserve-gib', str(rt['ram_reserve_gib'])]
