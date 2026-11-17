#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -A ssrinath-gcondo
#SBATCH -p gpu-he --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=6       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 2:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=64GB           # CPU RAM
#SBATCH --job-name='cocreation'
#SBATCH --output=slurm_logs/R-%x.%j/log.out
#SBATCH --error=slurm_logs/R-%x.%j/log.err
#SBATCH --constraint=blackwell

set -Eeo pipefail

echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""
echo "GPU Information (from host):"
nvidia-smi
echo ""
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
# source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
# conda init
conda activate cocreation
# check pytorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Commands follow
# Submit from the repo root. The nested Slurm output directory must exist
# before release: sbatch --hold --parsable run_pipeline.sh [video] [new-run-dir],
# mkdir -p slurm_logs/R-cocreation.JOB_ID; scontrol release JOB_ID.
# Defaults run the full pipeline on story_example.mp4 in a new scratch run.
set -u
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this script from the repo root}"
cd -- "$REPO_DIR"
INPUT_VIDEO="${1:-$REPO_DIR/scratch/data/samples/story_example.mp4}"
RUN_DIR="${2:-$REPO_DIR/scratch/runs/sequence-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)}"
PLAN_SEED="${PLAN_SEED:-0}"
REWORKS="${REWORKS:-2}"
VLLM_PORT="${VLLM_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
BASE_URL="http://127.0.0.1:$VLLM_PORT"
VLLM_PID=""
[[ -f "$INPUT_VIDEO" ]] || { echo "Input video does not exist: $INPUT_VIDEO" >&2; exit 1; }
mkdir -p -- "$REPO_DIR/scratch/runs"
mkdir -- "$RUN_DIR"  # Never overwrite or resume another run accidentally.
RUN_DIR="$(realpath -- "$RUN_DIR")"
INPUT_VIDEO="$(realpath -- "$INPUT_VIDEO")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" PYTHONUNBUFFERED=1
export COCREATION_PIPELINE_RUN_DIR="$RUN_DIR" COCREATION_PIPELINE_INPUT_VIDEO="$INPUT_VIDEO"
export COCREATION_PIPELINE_SEED="$PLAN_SEED" COCREATION_PIPELINE_REWORKS="$REWORKS" COCREATION_PIPELINE_BASE_URL="$BASE_URL"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2> >(tee -a "$RUN_DIR/pipeline.log" >&2)
printf '%s\n' "$RUN_DIR" > "$REPO_DIR/scratch/runs/latest-story-generation.txt"
printf '%s\n' "$RUN_DIR" > "$REPO_DIR/scratch/runs/latest-pipeline.txt"
echo "Run directory: $RUN_DIR"
echo "Input video: $INPUT_VIDEO"
echo "Video: 1280x736, single-pass 16 steps, 4 seconds per generated shot; thinking=false"

mark_state() {
    python - "$1" "$2" "${3:-0}" <<'PY_STATE'
from datetime import datetime,timezone
from pathlib import Path
import json,os,socket,sys
root=Path(os.environ['COCREATION_PIPELINE_RUN_DIR'])
path=root/'status.json'
state=json.loads(path.read_text()) if path.exists() else dict(
    job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),run_dir=str(root),
    input_video=os.environ['COCREATION_PIPELINE_INPUT_VIDEO'],history=[],
    settings=dict(seed=int(os.environ['COCREATION_PIPELINE_SEED']),reworks=int(os.environ['COCREATION_PIPELINE_REWORKS']),
                  thinking=False,video_width=1280,video_height=736,video_steps=16,video_seconds=4,
                  base_url=os.environ['COCREATION_PIPELINE_BASE_URL']))
now=datetime.now(timezone.utc)
old=state.get('stage')
if old and old!=sys.argv[2] and state.get('stage_started_at'):
    state['history'].append(dict(stage=old,finished_at=now.isoformat(),
        elapsed_seconds=round((now-datetime.fromisoformat(state['stage_started_at'])).total_seconds(),3)))
if old!=sys.argv[2]: state['stage_started_at']=now.isoformat()
state.update(status=sys.argv[1],stage=sys.argv[2],updated_at=now.isoformat(),exit_code=int(sys.argv[3]))
state.setdefault('started_at',now.isoformat())
if sys.argv[1] in ('complete','failed'): state['finished_at']=now.isoformat()
tmp=path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
print(now.isoformat(),sys.argv[1]+':',sys.argv[2],flush=True)
PY_STATE
}

stop_vllm() {
    [[ -n "$VLLM_PID" ]] || return 0
    echo "Stopping this job's vLLM process group: $VLLM_PID"
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        for ((attempt=0; attempt<60; attempt++)); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 2
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        fi
    fi
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 8
}

STAGE="initializing"
on_exit() {
    local code=$?
    trap - EXIT INT TERM
    set +e
    stop_vllm
    if ((code==0)); then
        mark_state complete complete 0
        echo "Finished: $RUN_DIR/dialogue/dialogue.md"
    else
        mark_state failed "$STAGE" "$code"
        echo "Pipeline failed in $STAGE (exit $code). See $RUN_DIR and the Slurm logs." >&2
    fi
    exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

STAGE="starting_vllm"
mark_state running "$STAGE"
setsid bash -l scripts/vllm/start.sh --port "$VLLM_PORT" --no-enable-thinking > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
python - "$BASE_URL" "$VLLM_PID" <<'PY_WAIT'
import os,sys,time,urllib.request
url=sys.argv[1]+'/health';pid=int(sys.argv[2]);deadline=time.monotonic()+1200
while True:
    try:
        with urllib.request.urlopen(url,timeout=3) as response:
            if response.status==200: break
    except Exception: pass
    try: os.kill(pid,0)
    except ProcessLookupError: raise SystemExit('vLLM exited before becoming ready; see vllm.log')
    if time.monotonic()>deadline: raise SystemExit('vLLM startup timed out; see vllm.log')
    time.sleep(3)
print('vLLM is ready at '+url,flush=True)
PY_WAIT

STAGE="observation"
mark_state running "$STAGE"
python -u tools/video_observer.py "$INPUT_VIDEO" \
    --output-dir "$RUN_DIR/observation" --backend vllm --base-url "$BASE_URL" \
    --no-enable-thinking --timeout 600 2>&1 | tee "$RUN_DIR/observation.log"

STAGE="dialogue_and_prompts"
mark_state running "$STAGE"
python -u scripts/dialogue_generation/generate.py "$RUN_DIR/observation/observation.json" \
    --output-dir "$RUN_DIR/dialogue" --seed "$PLAN_SEED" --reworks "$REWORKS" \
    --backend vllm --base-url "$BASE_URL" --max-tokens 512 --timeout 600 \
    --no-enable-thinking 2>&1 | tee "$RUN_DIR/dialogue.log"

STAGE="stopping_vllm"
mark_state running "$STAGE"
stop_vllm

STAGE="materialize"
mark_state running "$STAGE"
python -u scripts/dialogue_generation/materialize.py "$RUN_DIR/dialogue" --resume \
    --video-width 1280 --video-height 736 --output-width 1280 --output-height 736 \
    --video-steps 16 2>&1 | tee "$RUN_DIR/materialize.log"
