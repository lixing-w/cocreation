"""Install W4A8 H3 Image with NVFP4 text encoding in an isolated CUDA 12.8 runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation.storage import runtime_environment, scratch_root
from cocreation.h3_image import (MODEL_ID, COMFY_REVISION, WEIGHT_REPO, WEIGHT_REVISION,
                                 TRANSFORMER, ENCODER, VAE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-install', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Run inside an allocated GPU session, never on a login node')
    env = runtime_environment()
    root = scratch_root()/'runtime/minimax-h3-image'
    root.mkdir(parents=True, exist_ok=True)
    python, source = root/'venv/bin/python', root/'ComfyUI'
    def run(*command, **kwargs):
        return subprocess.run(list(map(str, command)), check=True, env=env, **kwargs)
    if not source.exists():
        run('git', 'clone', 'https://github.com/Comfy-Org/ComfyUI.git', source)
        run('git', '-C', source, 'checkout', '--detach', COMFY_REVISION)
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != COMFY_REVISION:
        raise RuntimeError('Image ComfyUI must be pinned to '+COMFY_REVISION)
    if not args.skip_install:
        if not python.is_file():
            run(sys.executable, '-m', 'venv', root/'venv')
        run(python, '-m', 'pip', 'install', 'torch==2.10.0+cu128', 'torchvision==0.25.0+cu128',
            '--index-url', 'https://download.pytorch.org/whl/cu128')
        constraints = root/'constraints.txt'
        constraints.write_text('torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\n')
        run(python, '-m', 'pip', 'install', '-c', constraints, '-r', source/'requirements.txt',
            'huggingface-hub>=0.34,<2')
        run(python, '-m', 'pip', 'check')
    files = [f'diffusion_models/{TRANSFORMER}', f'text_encoders/{ENCODER}', f'vae/{VAE}']
    if not args.skip_download:
        code = '''from huggingface_hub import hf_hub_download
from pathlib import Path
import json,sys
repo, revision, root, files = json.loads(sys.argv[1])
for name in files:
 target=Path(root)/'models'/name
 target.parent.mkdir(parents=True,exist_ok=True)
 shared=Path(root).parent/'minimax-h3/models'/name
 cached=shared.resolve() if name.startswith('text_encoders/') and shared.is_file() else Path(hf_hub_download(repo,name,revision=revision)).resolve()
 if target.is_symlink() and target.resolve()==cached: continue
 if target.exists() or target.is_symlink(): raise RuntimeError('Unmanaged image weight: '+str(target))
 target.symlink_to(cached)
'''
        run(python, '-c', code, json.dumps([WEIGHT_REPO, WEIGHT_REVISION, str(root), files]))
    with (root/'packages.txt').open('w') as stream:
        run(python, '-m', 'pip', 'freeze', stdout=stream)
    (root/'model.json').write_text(json.dumps(dict(model_id=MODEL_ID, weight_repo=WEIGHT_REPO,
        revision=WEIGHT_REVISION, comfy_revision=COMFY_REVISION, files=files,
        transformer_precision='W4A8', encoder_precision='NVFP4-AWQ', torch='2.10.0+cu128'),indent=2)+'\n')
    print('H3 Image ready:', root, flush=True)


if __name__ == '__main__':
    main()
