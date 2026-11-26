#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=ssrinath-gcondo
#SBATCH --partition=gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=09:00:00
#SBATCH --mem=64G
#SBATCH --job-name=actual-prompts
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Submit from the server repo: mkdir -p slurm_logs && sbatch run_actual_prompt_pipeline.sh
# Resume an interrupted batch (same options/code):
#   sbatch --export=ALL,RUN_DIR=/absolute/path/to/original/run run_actual_prompt_pipeline.sh
# Inputs are unchanged dataset records, bundled with provenance in DATA_DIR.
# New runs sample 0-1 reserved asset reworks per case, reproducibly by id/seed.
# REAL_PROMPT_REWORKS_POLICY=manifest reproduces source-manifest counts; 0/1/2 overrides.
# Resumes always retain the recorded cases.tsv choices.
# VLM_CONCURRENCY=8 by default: concurrent clients share ONE vLLM server.
# ALL VLM clients/preflights must succeed before serial materialization starts.
set -eo pipefail
module load miniforge3/25.3.0-3
module load cuda/12.9.0-cinr
source "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate cocreation
set -u
REPO="$HOME/cocreation"
# A direct shell invocation can use a test or debug checkout.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    REPO="${COCREATION_REPO:-$REPO}"
fi
export COCREATION_REPO="$REPO"
cd "$REPO"
SCRATCH_DIR="${COCREATION_SCRATCH:-$REPO/scratch}"
if [[ ! -d "$SCRATCH_DIR" ]]; then
    echo "Set COCREATION_SCRATCH to the existing scratch directory." >&2
    exit 2
fi
export COCREATION_SCRATCH="$(realpath "$SCRATCH_DIR")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1
DATA_DIR="${DATA_DIR:-$COCREATION_SCRATCH/data/prompts/actual-prompts-mixed-20261119}"
RUN_DIR="${RUN_DIR:-$COCREATION_SCRATCH/runs/actual-prompts-${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"
REAL_PROMPT_REWORKS_POLICY="${REAL_PROMPT_REWORKS_POLICY:-random_0_1}"
MATCH_VIDEO_BRIGHTNESS="${MATCH_VIDEO_BRIGHTNESS:-1}"
case "$MATCH_VIDEO_BRIGHTNESS" in
    1) TONE_ARGS=(--match-video-brightness) ;;
    0) TONE_ARGS=(--no-match-video-brightness) ;;
    *) echo "MATCH_VIDEO_BRIGHTNESS must be 0 or 1" >&2; exit 2 ;;
esac
EXPECTED_CASES="${EXPECTED_CASES:-8}"
VLM_CONCURRENCY="${VLM_CONCURRENCY:-8}"
if [[ ! "$VLM_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    echo "VLM_CONCURRENCY must be a positive integer" >&2
    exit 2
fi
H3_VIDEO_STEPS="${H3_VIDEO_STEPS:-50}"
EDIT_VIDEO_SAMPLER="${EDIT_VIDEO_SAMPLER:-res_multistep}"
VIDEO_ARGS=(--video-steps "$H3_VIDEO_STEPS" --edit-video-sampler "$EDIT_VIDEO_SAMPLER")
export RUN_DIR DATA_DIR REAL_PROMPT_REWORKS_POLICY MATCH_VIDEO_BRIGHTNESS
export EXPECTED_CASES VLM_CONCURRENCY H3_VIDEO_STEPS EDIT_VIDEO_SAMPLER
VLM_KV_CACHE_GIB="${VLM_KV_CACHE_GIB:-16}"
export VLM_KV_CACHE_GIB
QWEN_PORT="${QWEN_PORT:-8017}"
BASE_URL="http://127.0.0.1:$QWEN_PORT"
QWEN_PYTHON="$COCREATION_SCRATCH/runtime/qwen3.6-35b-a3b/vllm/bin/python"
VLLM_PID=""
VLM_POOL_PID=""
PHASE=initializing
CASE_ID=all

state() {
    PHASE="$1"
    CASE_ID="${2:-all}"
    python - "$PHASE" "$CASE_ID" <<'PY'
import datetime, json, os, socket, sys
from pathlib import Path
p=Path(os.environ['RUN_DIR'])/'pipeline_state.json'
v=dict(phase=sys.argv[1],case=sys.argv[2],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       job_id=os.environ.get('SLURM_JOB_ID'),node=socket.gethostname())
p.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v),flush=True)
PY
}
stop_vlm_clients() {
    if [[ -n "$VLM_POOL_PID" ]]; then
        # The pool handles TERM by terminating and reaping every active client group.
        kill -TERM "$VLM_POOL_PID" 2>/dev/null || true
        wait "$VLM_POOL_PID" 2>/dev/null || true
        VLM_POOL_PID=""
    fi
}
stop_qwen() {
    if [[ -n "$VLLM_PID" ]]; then
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        for ((attempt=0; attempt<60; attempt++)); do
            kill -0 -- "-$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 -- "-$VLLM_PID" 2>/dev/null; then
            kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        fi
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
cleanup() {
    status=$?
    trap - EXIT TERM INT
    stop_vlm_clients
    stop_qwen
    if [[ "$status" -ne 0 ]]; then
        printf 'Pipeline stopped in phase=%s case=%s exit=%s\n' "$PHASE" "$CASE_ID" "$status" >&2
        state failed "$CASE_ID" || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

printf 'Run directory: %s\nInput directory: %s\n' "$RUN_DIR" "$DATA_DIR"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$RUN_DIR/gpu.csv"
# Store provenance and validate the actual GPU plus all immutable source records.
python - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
from cocreation.real_prompt_policy import real_prompt_cases
from cocreation.qwen_inputs import DEFAULT_VIDEO_FPS, DEFAULT_VIDEO_MAX_FRAMES
run=Path(os.environ['RUN_DIR']); data=Path(os.environ['DATA_DIR'])
names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
if 'B200' not in names:
    raise RuntimeError('This batch requires an actual NVIDIA B200: '+names)
m=json.loads((data/'manifest.json').read_text())
expected=int(os.environ['EXPECTED_CASES'])
if expected<1 or len(m['cases'])!=expected or len({c['id'] for c in m['cases']})!=expected:
    raise ValueError(f'Expected exactly {expected} distinct inputs')
for c in m['cases']:
    raw=(data/c['input_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=c['input_sha256']:
        raise ValueError('Changed source record: '+c['id'])
(run/'inference_policy.json').write_text(json.dumps(dict(new_video=dict(turbo=False,steps=int(os.environ['H3_VIDEO_STEPS'])),video_edit=dict(turbo=False,steps=int(os.environ['H3_VIDEO_STEPS']),sampler=os.environ['EDIT_VIDEO_SAMPLER']),brightness_matching=os.environ['MATCH_VIDEO_BRIGHTNESS']=='1'),indent=2)+'\n')
(run/'vlm_execution.json').write_text(json.dumps(dict(concurrency=int(os.environ['VLM_CONCURRENCY']),shared_vllm=True,persistent_through_materialization=True,kv_cache_memory_gib=float(os.environ['VLM_KV_CACHE_GIB']),materialization='serial_after_all_vlm_success',video_sampling=dict(fps=DEFAULT_VIDEO_FPS,max_frames=DEFAULT_VIDEO_MAX_FRAMES,coverage='uniform_across_full_clip_including_endpoints')),indent=2)+'\n')
(run/'inputs.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
case_file=run/'cases.tsv'
rows, rework_policy=real_prompt_cases(m['cases'],
    policy=os.environ['REAL_PROMPT_REWORKS_POLICY'],
    previous_tsv=case_file.read_text() if case_file.exists() else None)
case_file.write_text(rows)
policy_file=run/'rework_policy.json'
if not policy_file.exists():
    policy_file.write_text(json.dumps(rework_policy,indent=2)+'\n')
print('Reserved rework choices: '+json.dumps(rework_policy),flush=True)
files=['tools/prompt_observer.py','tools/prompt_observer_prompts.py','scripts/dialogue_generation/generate.py',
       'tools/turn_media_review.py','scripts/dialogue_generation/reviewed_turn.py',
       'scripts/dialogue_generation/asset_variations.py','scripts/dialogue_generation/materialize.py',
       'scripts/dialogue_generation/parallel_vlm.py','scripts/dialogue_generation/formal_sequence.py',
       'scripts/dialogue_generation/planner_rules.py','scripts/dialogue_generation/generation.py','scripts/dialogue_generation/catalog.py','tools/video_editing.py','tools/media_review.py','tools/prompt_fixer.py','tools/prompt_reasoner.py','tools/placeholder_prompter.py','tools/dialogue_realizer.py','tools/dialogue_verifier.py','scripts/dialogue_generation/rewrite_prompts.py',
       'scripts/qwen/serve.py','cocreation/qwen_inputs.py','cocreation/qwen_client.py','cocreation/real_prompt_policy.py','cocreation/video_tone.py','cocreation/h3_policy.py','cocreation/h3.py','cocreation/models.py','cocreation/comfy_worker.py','run_actual_prompt_pipeline.sh']
(run/'code_hashes.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2)+'\n')
PY
cp run_actual_prompt_pipeline.sh "$RUN_DIR/submitted_pipeline.sh"

# Phase 1: a single Qwen deployment serves all selected observations and dialogues.
state starting_qwen
# Refuse to share a live service whose GPU ownership is unknown.
if curl --max-time 2 -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    echo "Port $QWEN_PORT already serves a model; choose a different QWEN_PORT." >&2
    exit 1
fi
setsid "$QWEN_PYTHON" -u scripts/qwen/serve.py \
    --port "$QWEN_PORT" --no-enable-thinking --kv-cache-memory-gib "$VLM_KV_CACHE_GIB" > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
printf '%s\n' "$VLLM_PID" > "$RUN_DIR/vllm.pid"
for ((attempt=0; attempt<240; attempt++)); do
    curl --max-time 2 -fsS "$BASE_URL/health" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID"
    sleep 5
done
curl --max-time 2 -fsS "$BASE_URL/health" >/dev/null
cp "$COCREATION_SCRATCH/runtime/qwen3.6-35b-a3b/server.json" "$RUN_DIR/qwen_server.json"

# Each case runs observer -> dialogue/prompts -> dry-run in dependency order.
# The pool bounds concurrent cases; vLLM batches their requests on the same GPU.
state vlm_parallel
python -u scripts/dialogue_generation/parallel_vlm.py \
    --repo "$REPO" --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" --base-url "$BASE_URL" \
    --concurrency "$VLM_CONCURRENCY" "${VIDEO_ARGS[@]}" \
    "${TONE_ARGS[@]}" &
VLM_POOL_PID=$!
# Hard barrier: a failed/cancelled pool exits through cleanup; no media is generated.
wait "$VLM_POOL_PID"
VLM_POOL_PID=""

state shared_qwen_ready
# Keep this owned service resident for critiques, repairs and video localization.
# EXIT cleanup stops it after all sequences, including failed/cancelled batches.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RUN_DIR/gpu-after-text.csv"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/text_phase_complete.txt"

# Phase 2: materialize every sequence after ALL text-model work has finished.
# All H3 videos: original model, 50 steps by default; Turbo disabled.
# New videos use Euler; edits retain res_multistep/simple.
# H3-Image retains the repository's default resolution.
# Generated-video localization reuses this persistent VLM after its source exists.
while IFS=$'\t' read -r case_id source_format input_file seed reworks width height; do
    case_dir="$RUN_DIR/$case_id"
    state materializing "$case_id"
    MATERIALIZE_RESUME_ARGS=()
    if [[ -f "$case_dir/dialogue/media_manifest.json" ]]; then
        MATERIALIZE_RESUME_ARGS=(--resume)
    fi
    if python -u scripts/dialogue_generation/materialize.py "$case_dir/dialogue" "${MATERIALIZE_RESUME_ARGS[@]}" \
        --seed "$seed" --video-width "$width" --video-height "$height" \
        --native-video-output --verify-media --shared-vlm --review-base-url "$BASE_URL" "${VIDEO_ARGS[@]}" "${TONE_ARGS[@]}" \
        2>&1 | tee -a "$case_dir/materialize.log"; then
        printf 'Sequence completed: %s\n' "$case_id"
    else
        case_status=$?
        printf 'Sequence terminated: %s exit=%s; continuing remaining cases\n' "$case_id" "$case_status"
    fi
done < "$RUN_DIR/cases.tsv"

python - <<'PY'
import json, os
from pathlib import Path
run=Path(os.environ['RUN_DIR']); results=[]
for c in json.loads((run/'inputs.json').read_text())['cases']:
    d=run/c['id']; obs=json.loads((d/'observation/observation.json').read_text())
    dialogue=json.loads((d/'dialogue/dialogue.json').read_text())
    media=json.loads((d/'dialogue/media_manifest.json').read_text())
    # Failed/terminated sequences are retained as useful reviewed traces.
    results.append(dict(id=c['id'],assets=len(obs['assets']),shots=len(obs['shots']),
        turns=len(dialogue['turns']),media_count=len(media['media']),materialization=media['status'],
        dialogue=str(d/'dialogue/dialogue.md'),sequence=str(d/'dialogue/sequence.json'),error=media.get('error')))
(run/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
(run/'batch_outcome.txt').write_text(('complete' if all(r['materialization']=='complete' for r in results) else 'complete_with_failures')+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
PY
read -r BATCH_OUTCOME < "$RUN_DIR/batch_outcome.txt"
state "$BATCH_OUTCOME"
printf 'Finished %s sequences: %s/results.json\n' "$EXPECTED_CASES" "$RUN_DIR"
