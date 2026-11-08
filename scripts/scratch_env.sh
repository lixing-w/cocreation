# Source this file after activating conda, before pip installs or running jobs.
# The Python controllers also set these paths for their inference children.
_cocreation_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${COCREATION_SCRATCH:-}" ] && [ ! -L "$_cocreation_repo/scratch" ]; then
    echo "Run python scripts/setup_scratch.py first." >&2
    unset _cocreation_repo
    return 1
fi
export COCREATION_SCRATCH="${COCREATION_SCRATCH:-$_cocreation_repo/scratch}"
export XDG_CACHE_HOME="$COCREATION_SCRATCH/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda/pkgs"
export MAMBA_ROOT_PREFIX="$COCREATION_SCRATCH/runtime/mamba"
export TMPDIR="$COCREATION_SCRATCH/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
unset _cocreation_repo
