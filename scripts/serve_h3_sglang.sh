#!/usr/bin/env bash
# Serve the MiniMax-H3 video and image checkpoints on Oscar.
set -euo pipefail

usage() {
  printf '%s\n' "Usage: $0 {video|text-video|image} [--h3-vae {native|mt}] [--print-config] [SGLang server options...]"
}

kind=${1:-}
case "$kind" in
  video|text-video|image) shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

h3_vae=native
if [[ ${1:-} == --h3-vae ]]; then
  h3_vae=${2:-}
  if [[ $kind == image || ( $h3_vae != native && $h3_vae != mt ) ]]; then
    printf '%s\n' '--h3-vae requires video or text-video and a value of native or mt' >&2
    exit 2
  fi
  shift 2
fi

scratch=${COCREATION_SCRATCH:-$(realpath "$HOME/cocreation/scratch")}
runtime=${COCREATION_SGLANG_RUNTIME:-"$scratch/runtime/sglang-h3-benchmark"}
venv="$runtime/venv-cu130"
video_models="$scratch/runtime/minimax-h3/models"
image_models="$scratch/runtime/minimax-h3-image/models"

args=(
  serve --model-type diffusion --host 127.0.0.1 --num-gpus 1
  --warmup-mode off --performance-mode speed
)
if [[ $kind != image ]]; then
  variant=ref2va
  transformer=${H3_VIDEO_TRANSFORMER:-"$runtime/weights/minimax_h3_ref2va_pruned_int8_convrot.safetensors"}
  if [[ $kind == text-video ]]; then
    variant=fl2va
    transformer=${H3_TEXT_VIDEO_TRANSFORMER:-"$runtime/weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors"}
  fi
  video_vae="$video_models/vae/minimax_h3_video_vae_fp16.safetensors"
  audio_vae="$video_models/vae/minimax_h3_audio_vae_fp32.safetensors"
  if [[ $h3_vae == mt ]]; then
    video_vae="$video_models/vae/minimax_h3_video_vae_mt_fp16.safetensors"
    audio_vae="$video_models/vae/minimax_h3_audio_vae_mt_fp32.safetensors"
    for checkpoint in "$video_vae" "$audio_vae"; do
      if [[ ! -r $checkpoint ]]; then
        printf 'H3-MT-VAE checkpoint is unreadable: %s\n' "$checkpoint" >&2
        exit 1
      fi
    done
  fi
  args+=(
    --model-path "$runtime/native-models/MiniMax-H3"
    --model-variant "$variant"
    --pipeline-class-name MiniMaxH3Pipeline
    --served-model-name MiniMaxAI/MiniMax-H3
    --component-weights-paths.text_encoder "$video_models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    --component-weights-paths.video_vae "$video_vae"
    --component-weights-paths.audio_vae "$audio_vae"
  )
else
  image_snapshot="$scratch/cache/huggingface/hub/models--Comfy-Org--MiniMax-H3-Image/snapshots/89c635085f539d03ad35485ed542c5a286e3c6d6"
  transformer=${H3_IMAGE_TRANSFORMER:-"$image_snapshot/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"}
  args+=(
    --model-path "$runtime/native-models/MiniMax-H3-Image"
    --pipeline-class-name MiniMaxH3ImagePipeline
    --served-model-name MiniMaxAI/MiniMax-H3-Image
    --component-paths.text_encoder "$runtime/native-models/MiniMax-H3-Image/text_encoder"
    --component-weights-paths.text_encoder "$image_models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    --component-weights-paths.video_vae "$image_models/vae/minimax_h3_image_vae_fp16.safetensors"
  )
fi
args+=(--component-weights-paths.transformer "$transformer")
if [[ ! -r $transformer ]]; then
  printf 'Transformer checkpoint is unreadable: %s\n' "$transformer" >&2
  exit 1
fi
if [[ ${1:-} == --print-config ]]; then
  shift
  printf '%q ' "$venv/bin/sglang" "${args[@]}" "$@"
  printf '\n'
  exit 0
fi

# The batch may load a different CUDA module for other services. Its CPATH would
# override the explicit CUDA 13 system headers in FlashInfer's JIT compiler.
inherited_cuda=${CUDA_HOME:-${CUDA_PATH:-}}
if [[ -n $inherited_cuda && $inherited_cuda != "$runtime/cuda-13.0" && -n ${CPATH:-} ]]; then
  IFS=: read -r -a include_dirs <<< "$CPATH"
  own_includes=()
  for directory in "${include_dirs[@]}"; do
    if [[ $directory != "$inherited_cuda/include" && $directory != "$inherited_cuda"/targets/*/include ]]; then
      own_includes+=("$directory")
    fi
  done
  export CPATH="$(IFS=:; printf '%s' "${own_includes[*]}")"
fi
export CUDA_HOME="$runtime/cuda-13.0" CUDA_PATH="$runtime/cuda-13.0"
compat="$runtime/cuda-compat-13-0/usr/local/cuda-13.0/compat"
export PATH="$venv/bin:$CUDA_HOME/bin:$HOME/.conda/envs/cocreation/bin:$PATH"
export LD_LIBRARY_PATH="$compat:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export XDG_CACHE_HOME="$runtime/cache"
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"$runtime/cache/flashinfer-cu130"}
export HF_HOME="$scratch/cache/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}" SGLANG_USE_RUNAI_MODEL_STREAMER=0

exec "$venv/bin/sglang" "${args[@]}" "$@"
