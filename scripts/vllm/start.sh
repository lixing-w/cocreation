#!/usr/bin/env bash
# Run inside a GPU interactive session: bash -l scripts/vllm/start.sh
set -eo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="${COCREATION_SCRATCH:-$repo_dir/scratch}/runtime/qwen3.6-35b-a3b"
python_bin="$runtime_dir/vllm/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "Qwen vLLM environment is missing: $python_bin" >&2
    echo "Prepare it with scripts/setup/setup_qwen.py --install-vllm." >&2
    exit 1
fi
if ! type module >/dev/null 2>&1; then
    echo "Initialize Oscar modules with: bash -l $repo_dir/scripts/vllm/start.sh" >&2
    exit 1
fi

module load cuda/12.9.0-cinr
cd -- "$repo_dir"
echo "Starting Qwen3.6 vLLM on $(hostname) in the foreground."
echo "Wait for 'Application startup complete' before running the data pipeline."
exec "$python_bin" -u "$repo_dir/scripts/qwen/serve.py" "$@"
