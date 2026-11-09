"""Isolated, pinned CUDA 12.8 installation; never run FreeVideo's launcher setup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .vdn import (FREEVIDEO_REVISION, VDN_REVISION, COMFY_REVISION,
                  DIFFUSERS_REVISION, TORCH_VERSION, PATCH_VERSION, ENCODER)


def run(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, **kwargs)


def checkout(url, revision, path):
    if not path.exists():
        run(['git', 'clone', '--no-checkout', url, path])
        run(['git', '-C', path, 'checkout', '--detach', revision])
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'Expected {path} at {revision}; found {actual}. Use a new runtime root.')


def replace_once(path, before, after):
    content = path.read_text()
    if after in content:
        return
    if content.count(before) != 1:
        raise RuntimeError(f'Compatibility patch no longer matches {path}')
    path.write_text(content.replace(before, after))


def patch_sources(freevideo, comfy):
    # Keep cu130 CUDA kernels DISABLED on cu128. Do not spoof the CUDA version.
    # Kitchen's eager dequantization remains available; native FP4 GEMM is not required.
    quant = comfy / 'comfy/quant_ops.py'
    before = 'logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.\\nWARNING WARNING WARNING\\nIf you are on nvidia 20 series and above it is required that you update your pytorch to cu130 or higher.\\n")'
    after = 'logging.info("cocreation cu128: comfy-kitchen native CUDA kernels disabled; using compatible fallback dequantization.")'
    replace_once(quant, before, after)
    encoder = freevideo / 'freevideo_engine/encode_worker.py'
    replace_once(encoder, "'--disable-dynamic-vram', '--disable-comfy-compiler', '--disable-cuda-graphs']",
        "'--disable-dynamic-vram', '--disable-comfy-compiler', '--disable-cuda-graphs',\n"
        "                '--disable-triton-backend', '--use-pytorch-cross-attention']")
    # Force the upstream installed-backend inventory to the tested-by-probe torch routes.
    # Do not select locally installed cu130 extension wheels through auto-planning.
    hardware = freevideo / 'freevideo_engine/hardware.py'
    text = hardware.read_text()
    marker = '\n# cocreation-cu128 backend inventory\n'
    if marker not in text:
        text += marker + "def installed_backends():\n    return {'cudnn', 'torch-flash'}\n"
        hardware.write_text(text)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for root, p in ((freevideo, encoder), (freevideo, hardware), (comfy, quant))}


def environment(root, env=None):
    env = dict(os.environ if env is None else env)
    env.update(FREEVIDEO_HOME=str(root), FREEVIDEO_VDN_ROOT=str(root / 'vendor/vdn'),
               FREEVIDEO_COMFY_ROOT=str(root / 'vendor/ComfyUI'),
               FREEVIDEO_MODEL_ROOT=str(root / 'models'), FREEVIDEO_KEEP_MODELS='off')
    # A dedicated venv, no inherited external resident session or import shadowing.
    env.pop('FREEVIDEO_RESIDENT_SESSION', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = str(root / 'vendor/FreeVideo')
    env['PATH'] = str(root / 'venv/bin') + os.pathsep + env.get('PATH', '')
    env['PIP_CONSTRAINT'] = str(root / 'cu128-constraints.txt')
    return env


def prepare_runtime(rt):
    root = Path(rt['root']).expanduser().resolve()
    python = root / 'venv/bin/python'
    receipt = root / 'runtime-ready.json'
    expected = dict(freevideo=FREEVIDEO_REVISION, vdn=VDN_REVISION, comfy=COMFY_REVISION,
                    diffusers=DIFFUSERS_REVISION, torch=TORCH_VERSION, patch=PATCH_VERSION)
    if receipt.exists() and python.exists():
        actual = json.loads(receipt.read_text())
        if actual.get('versions') != expected:
            raise RuntimeError('VDN runtime pins changed; select a new VDNRuntime(root=...)')
        # Validate the interpreter even on reused installations; never trust a stale marker.
        for name, revision in (('FreeVideo', FREEVIDEO_REVISION), ('vdn', VDN_REVISION), ('ComfyUI', COMFY_REVISION)):
            if not (root / 'vendor' / name / '.git').is_dir():
                raise RuntimeError('Missing pinned runtime source: ' + name)
            checkout('', revision, root / 'vendor' / name)
        for name, digest in actual.get('patches', {}).items():
            parent = 'ComfyUI' if name.startswith('comfy/') else 'FreeVideo'
            if hashlib.sha256((root / 'vendor' / parent / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('VDN compatibility patch changed: ' + name)
        check_torch(python, environment(root))
        return root, python
    if not rt['auto_setup'] or rt['local_files_only']:
        raise RuntimeError('VDN runtime not prepared. Run python scripts/setup_vdn.py on the server first.')
    root.mkdir(parents=True, exist_ok=True)
    vendor = root / 'vendor'
    vendor.mkdir(exist_ok=True)
    fv, vdn, comfy = (vendor / name for name in ('FreeVideo', 'vdn', 'ComfyUI'))
    checkout('https://github.com/FlashML-org/FreeVideo.git', FREEVIDEO_REVISION, fv)
    checkout('https://github.com/OpenVDN/vdn-minimax-h3.git', VDN_REVISION, vdn)
    checkout('https://github.com/Comfy-Org/ComfyUI.git', COMFY_REVISION, comfy)
    diffusers = vdn / 'diffusers'
    checkout('https://github.com/huggingface/diffusers.git', DIFFUSERS_REVISION, diffusers)
    # Apply the pinned VDN patches without creating commits or requiring git identity.
    for patch in sorted((vdn / 'diffusers_patches').glob('*.patch')):
        reverse = subprocess.run(['git', '-C', str(diffusers), 'apply', '--reverse', '--check', str(patch)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if reverse.returncode:
            run(['git', '-C', diffusers, 'apply', '--check', patch])
            run(['git', '-C', diffusers, 'apply', patch])
    patches = patch_sources(fv, comfy)
    # FreeVideo requires Python 3.12. Let the caller select its bootstrap interpreter.
    version = subprocess.check_output([rt['python'], '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
    if version != '3.12':
        raise RuntimeError('VDN needs Python 3.12. Set VDNRuntime(python="/path/to/python3.12") or setup_vdn.py --python.')
    if not python.exists():
        run([rt['python'], '-m', 'venv', root / 'venv'])
    constraints = root / 'cu128-constraints.txt'
    constraints.write_text('torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\ntorchaudio==2.10.0+cu128\ntriton==3.6.0\n')
    env = environment(root)
    run([python, '-m', 'pip', 'install', '--upgrade', 'pip'], env=env)
    run([python, '-m', 'pip', 'install', 'torch==2.10.0+cu128', 'torchvision==0.25.0+cu128',
         'torchaudio==2.10.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'], env=env)
    # Encoder-library dependencies only. No frontend, manager, launcher, SageAttention or FA4 wheels.
    excluded = ('torch', 'torchaudio', 'torchvision', 'comfyui-', 'comfy-angle', 'PyOpenGL')
    requirements = [line for line in (comfy / 'requirements.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#') and not any(
                        line == x or line.startswith(x + '=') or line.startswith(x + '>') or
                        (x.endswith('-') and line.startswith(x)) for x in excluded)]
    requirements += ['transformers==5.15.0', 'accelerate==1.14.0', 'peft==0.20.0',
                     'safetensors==0.8.0', 'huggingface_hub==1.30.0', 'omegaconf==2.3.1',
                     'einops==0.8.2', 'numpy>=2,<3', 'httpx[socks]>=0.28', 'requests[socks]>=2.32',
                     'importlib_metadata>=8']
    deps = root / 'encoder-engine-requirements.txt'
    deps.write_text('\n'.join(requirements) + '\n')
    run([python, '-m', 'pip', 'install', '-r', deps, '-e', diffusers, '-e', fv], env=env)
    run([python, '-m', 'pip', 'check'], env=env)
    check_torch(python, env)
    # Basic import/CLI checks do not load weights or claim GPU correctness.
    run([python, '-m', 'freevideo_engine.cli', 'generate', '--help'], env=env)
    with (root / 'installed-packages.txt').open('w') as output:
        run([python, '-m', 'pip', 'freeze'], env=env, stdout=output)
    receipt.write_text(json.dumps(dict(versions=expected, patches=patches, gpu_validated=False), indent=2))
    return root, python


def check_torch(python, env):
    run([python, '-c', 'import torch; assert torch.__version__ == "2.10.0+cu128", torch.__version__; '
         'assert torch.version.cuda == "12.8", torch.version.cuda'], env=env)
