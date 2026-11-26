#!/bin/bash
#SBATCH --job-name=dialogue-sonnet-medium48-c12
#SBATCH --account=ssrinath-gcondo
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output=/users/lwang223/cocreation/slurm_logs/%x-%j.out
#SBATCH --error=/users/lwang223/cocreation/slurm_logs/%x-%j.err

# 在 Oscar 主 repo 运行 sbatch run_native_dialogue_batch.sh 即可提交默认批次。
PROJECT_DIR="/users/lwang223/cocreation"
INPUT_MANIFEST="$PROJECT_DIR/data/inputs/dialogue-native-48-20261124/manifest.json"
# Slurm 每次提交会分配一个作业 ID，用它给新输出目录命名。
RUN_DIR="$PROJECT_DIR/data/runs/dialogue-native-sonnet-medium-48-c12-${SLURM_JOB_ID:-manual}"
BACKEND="sonnet"
THINK_MODE="medium"
# 同时处理的 sequence 数。GPU 数量在文件开头配置。
CONCURRENCY=12
H3_VAE="mt"

set -eo pipefail

# B200 实验启用媒体模型常驻，可用 --no-resident-media-models 覆盖。
python_options=(--backend "$BACKEND" --think-mode "$THINK_MODE" --concurrency "$CONCURRENCY" --resident-media-models --h3-vae "$H3_VAE")

# 手写 manifest 时，由传入参数和批跑程序的默认值决定后端等设置。
# sbatch run_native_dialogue_batch.sh MANIFEST [RUN_DIR] [OPTIONS]
if [[ $# -gt 0 && "$1" != --* ]]; then
    INPUT_MANIFEST="$1"
    [[ "$INPUT_MANIFEST" == /* ]] || INPUT_MANIFEST="$PROJECT_DIR/$INPUT_MANIFEST"
    RUN_DIR="$PROJECT_DIR/data/runs/dialogue-native-${SLURM_JOB_ID:-manual}"
    python_options=()
    shift
    if [[ $# -gt 0 && "$1" != --* ]]; then
        RUN_DIR="$1"
        [[ "$RUN_DIR" == /* ]] || RUN_DIR="$PROJECT_DIR/$RUN_DIR"
        shift
    fi
fi

# 加载与原作业相同的软件环境。
module load miniforge3/25.3.0-3
module load cuda/12.9.0-cinr
source "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate cocreation

# 正式作业始终运行主 repo 的代码。
export COCREATION_REPO="$PROJECT_DIR"
cd "$COCREATION_REPO"
export COCREATION_SCRATCH="${COCREATION_SCRATCH:-$(realpath scratch)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

printf 'Code directory: %s\nInput manifest: %s\nRun directory: %s\n' "$PWD" "$INPUT_MANIFEST" "$RUN_DIR"
exec python -u scripts/dialogue_generation/native_batch.py \
    "$INPUT_MANIFEST" "$RUN_DIR" "${python_options[@]}" "$@"
