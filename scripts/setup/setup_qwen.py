"""Install the Qwen3.6 NVFP4 deployment in an isolated CUDA 12.9 environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation.storage import runtime_environment, scratch_root, REPO_ROOT
from cocreation.qwen_client import MODEL_ID

VLLM_VERSION = '0.28.0'
VLLM_WHEEL = ('https://github.com/vllm-project/vllm/releases/download/v0.28.0/'
              'vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-vllm', action='store_true')
    p.add_argument('--skip-download', action='store_true')
    a = p.parse_args()
    if not sys.platform.startswith('linux'):
        p.error('Run this setup on the Linux NVIDIA server')
    env = runtime_environment()
    root = scratch_root()/'runtime/qwen3.6-35b-a3b'
    root.mkdir(parents=True, exist_ok=True)
    if not a.skip_download:
        os.environ.update(env)
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(MODEL_ID).sha
        checkpoint = snapshot_download(MODEL_ID, revision=revision, max_workers=4)
        (root/'model.json').write_text(json.dumps(dict(model_id=MODEL_ID,
            revision=revision, path=checkpoint), indent=2)+'\n')
    if a.install_vllm:
        venv = root/'vllm'
        if not (venv/'bin/python').exists():
            subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True, env=env)
        python = str(venv/'bin/python')
        subprocess.run([python, '-m', 'pip', 'install', '-U', 'pip'], check=True, env=env)
        subprocess.run([python, '-m', 'pip', 'install', VLLM_WHEEL,
            '--extra-index-url', 'https://download.pytorch.org/whl/cu129',
            'qwen-vl-utils>=0.0.14', 'av>=14,<17',
            'transformers==5.7.0', 'huggingface-hub==1.27.0', 'tokenizers==0.22.2'], check=True, env=env)
        # PyPI TorchCodec currently links CUDA 13 even when Torch uses cu129.
        # CPU video decoding keeps the inference environment compatible with driver 575.
        subprocess.run([python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
            'torchcodec==0.17.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True, env=env)
        subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '-e', str(REPO_ROOT)], check=True, env=env)
        with (root/'vllm-packages.txt').open('w') as stream:
            subprocess.run([python, '-m', 'pip', 'freeze'], check=True, env=env, stdout=stream)


if __name__ == '__main__':
    main()
