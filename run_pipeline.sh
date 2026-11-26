#!/bin/bash
#SBATCH --nodes=1
# SBATCH --nodelist=gpu3202
#SBATCH --account=ssrinath-gcondo
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=6:30:00
#SBATCH --mem=64G
#SBATCH --job-name=cocreation
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --constraint=b200

set -e
module load miniforge3/25.3.0-3
module load cuda/12.9.0-cinr
source "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate cocreation
cd "$HOME/cocreation"

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
RUN_DIR="scratch/runs/sequence-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
echo "Run directory: $PWD/$RUN_DIR"

# Start Qwen and wait until its API is ready.
setsid scratch/runtime/qwen3.6-35b-a3b/vllm/bin/python -u scripts/qwen/serve.py \
    --port 8000 --no-enable-thinking > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill -- "-$VLLM_PID" 2>/dev/null || true' EXIT
for attempt in {1..240}; do
    curl --max-time 2 -fsS http://127.0.0.1:8000/health > /dev/null 2>&1 && break
    kill -0 "$VLLM_PID"
    sleep 5
done
curl --max-time 2 -fsS http://127.0.0.1:8000/health > /dev/null

# Observe the source video.
python -u tools/video_observer.py scratch/data/samples/story_example.mp4 \
    --output-dir "$RUN_DIR/observation" \
    --backend vllm --base-url http://127.0.0.1:8000 \
    --no-enable-thinking --max-tokens 8192 --timeout 600

# Plan the dialogue and write the H3-Image / H3 prompts.
python -u scripts/dialogue_generation/generate.py "$RUN_DIR/observation/observation.json" \
    --output-dir "$RUN_DIR/dialogue" --seed 0 --reworks 2 \
    --backend vllm --base-url http://127.0.0.1:8000 \
    --no-enable-thinking --max-tokens 4096 --timeout 600

# Free Qwen's GPU memory before generating images and videos.
kill -- "-$VLLM_PID"
wait "$VLLM_PID" || true
trap - EXIT

# H3-Image images; original H3 Base-Ref2VA videos without LoRA.
# For Turbo v4-600, add --video-turbo-lora and change --video-steps to 6.
python -u scripts/dialogue_generation/materialize.py "$RUN_DIR/dialogue" --resume \
    --video-width 1344 --video-height 768 --native-video-output \
    --video-steps 50

echo "Finished: $PWD/$RUN_DIR/dialogue/dialogue.md"
