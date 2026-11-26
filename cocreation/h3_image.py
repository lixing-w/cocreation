"""MiniMax-H3-Image deployment contract; importing this module never loads ML."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from .storage import scratch_root

MODEL_ID = 'MiniMaxAI/MiniMax-H3-Image'
COMFY_REVISION = 'c226dc23a3692117fd2f087d26ce5dc5eee8b138'
WEIGHT_REPO = 'Comfy-Org/MiniMax-H3-Image'
WEIGHT_REVISION = '0bb149f51e3156ed4b90dd45d5620346bfe1df8f'
TRANSFORMER = 'minimax_h3_image_pruned_w4a8.safetensors'
ENCODER = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
VAE = 'minimax_h3_image_vae_fp16.safetensors'
SIZES = {'1:1': (4096, 4096), '4:3': (4736, 3552), '3:4': (3552, 4736),
         '3:2': (5024, 3360), '2:3': (3360, 5024), '16:9': (5472, 3072),
         '9:16': (3072, 5472), '21:9': (6272, 2688)}


def size_for(ratio='1:1'):
    if ratio in SIZES:
        return SIZES[ratio]
    width, height = (int(part) for part in ratio.split(':'))
    if min(width, height) <= 0 or not 1/3 <= width/height <= 3:
        raise ValueError('H3 image aspect ratio must be between 1:3 and 3:1')
    scale = math.sqrt(4096**2 / (width*height))
    return tuple(max(32, round(value*scale/32)*32) for value in (width, height))


@dataclass(frozen=True)
class H3ImageRuntime:
    root: str = field(default_factory=lambda: str(scratch_root()/'runtime/minimax-h3-image'))
    python: str = field(default_factory=lambda: str(scratch_root()/'runtime/minimax-h3-image/venv/bin/python'))
    timeout: float = 7200
    startup_timeout: float = 300

    def __post_init__(self):
        if min(self.timeout, self.startup_timeout) <= 0:
            raise ValueError('H3 image requires positive timeouts')


def installation(root=None):
    root = Path(root or H3ImageRuntime().root)
    manifest = root/'model.json'
    if not manifest.is_file():
        raise RuntimeError('Prepare H3 Image inside a GPU allocation: python scripts/setup/setup_h3_image.py')
    info = json.loads(manifest.read_text())
    if info['model_id'] != MODEL_ID or not (root/'venv/bin/python').is_file():
        raise RuntimeError('Incomplete H3 Image runtime: '+str(root))
    if info.get('comfy_revision') != COMFY_REVISION or info.get('revision') != WEIGHT_REVISION:
        raise RuntimeError('H3 Image runtime has an unexpected revision')
    for name in info['files']:
        path = root/'models'/name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Incomplete H3 Image checkpoint: '+str(path))
    return info


def build_graph(request, media):
    """Official Comfy image graph with its checkpoint-owned PDD sigma schedule."""
    graph = {}
    def add(key, cls, **inputs):
        graph[key] = dict(class_type=cls, inputs=inputs)
        return [key, 0]
    model = add('model', 'UNETLoader', unet_name=TRANSFORMER, weight_dtype='default')
    clip = add('encoder', 'CLIPLoader', clip_name=ENCODER, type='minimax', device='default')
    vae = add('vae', 'VAELoader', vae_name=VAE)
    options = request['options']
    inputs = dict(clip=clip, vae=vae, prompt=request['prompt'], width=options['width'], height=options['height'])
    for i, name in enumerate(media, 1):
        inputs[f'images.image_{i}'] = add(f'image_{i}', 'LoadImage', image=name)
    positive = add('condition', 'MiniMaxH3ImageConditioning', **inputs)
    negative = add('negative', 'ConditioningZeroOut', conditioning=positive)
    samples = add('sample', 'KSampler', model=model, positive=positive, negative=negative,
                  latent_image=['condition', 1], seed=options['seed'], steps=6, cfg=1.0,
                  sampler_name='euler', scheduler='simple', denoise=1.0)
    image = add('decode', 'VAEDecode', samples=samples, vae=vae)
    add('save', 'SaveImage', images=image, filename_prefix='image')
    return graph
