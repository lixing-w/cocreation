#!/usr/bin/env bash
# Run on the same GPU node as start.sh. No CUDA module or conda activation needed.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="${COCREATION_SCRATCH:-$repo_dir/scratch}/runtime/qwen3.6-35b-a3b"
python_bin="$runtime_dir/vllm/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "Qwen vLLM environment is missing: $python_bin" >&2
    exit 1
fi

cd -- "$repo_dir"
exec "$python_bin" "$repo_dir/scripts/qwen/serve.py" --stop
