"""One storage root for datasets, outputs, model weights, and runtime caches."""
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def scratch_root():
    return Path(os.environ.get("COCREATION_SCRATCH", REPO_ROOT / "scratch")).expanduser().resolve()


def ensure_scratch():
    link = REPO_ROOT / "scratch"
    if sys.platform.startswith("linux") and not os.environ.get("COCREATION_SCRATCH") and not link.is_symlink():
        raise RuntimeError("Configure scratch first: python scripts/setup_scratch.py --target /users/lwang223/scratch/cocreation")
    root = scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_environment():
    root = ensure_scratch()
    cache = root / "cache"
    env = os.environ.copy()
    paths = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub",
        "TRANSFORMERS_CACHE": cache / "huggingface/hub",
        "HF_XET_CACHE": cache / "huggingface/xet",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "XDG_CACHE_HOME": cache,
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "PIP_CACHE_DIR": cache / "pip",
        "UV_CACHE_DIR": cache / "uv",
        "CONDA_PKGS_DIRS": cache / "conda/pkgs",
        "MAMBA_ROOT_PREFIX": root / "runtime/mamba",
        "TMPDIR": root / "tmp",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env
