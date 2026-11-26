# Qwen3.6-35B-A3B on Oscar

The CUDA pipeline uses `nvidia/Qwen3.6-35B-A3B-NVFP4` through one shared vLLM service. This is NVIDIA's mixed-precision checkpoint of Qwen3.6, with NVFP4 MoE weights, 16-bit MoE activations (W4A16), FP8 components and a retained vision encoder for images/video. NVFP4 is floating-point 4-bit quantization, not GGUF Q4_K_M. vLLM uses the checkpoint-compatible Marlin W4A16 path, FP8 kernels where configured, FlashInfer attention, CUDA graphs and continuous batching. W4A16 does not imply native FP4 activation arithmetic. No performance benchmark is required or run by setup/validation.

Linux observation/dialogue CLIs default to vLLM. Mac Ollama defaults remain unchanged. Use `--model qwen3.6-35b-a3b` or the exact NVIDIA model ID for the server. Old Qwen3-VL aliases are accepted only by Ollama.

## Install and serve

Use the existing GPU allocation. The 575 driver supports CUDA 12.9, so setup selects the official vLLM 0.28.0 **cu129** wheel and its matching Torch dependencies, in an isolated scratch venv. Do not install the default CUDA 13 wheel or alter the Comfy/H3/VDN environments. Transformers is pinned to 5.7.0 with compatible Hub/tokenizers versions: newer processor validation rejects vLLM 0.28's split video/text preprocessing. Setup uses the CPU TorchCodec wheel because its default CUDA wheel links CUDA 13 libraries. The launcher also pins both Triton PTXAS paths to the loaded CUDA 12.9 toolkit, avoiding kernels assembled by a newer bundled toolchain on driver 575.

```bash
cd /users/lwang223/cocreation
conda activate cocreation
python scripts/setup/setup_qwen.py --install-vllm
module load cuda/12.9.0-cinr
scratch/runtime/qwen3.6-35b-a3b/vllm/bin/python scripts/qwen/serve.py
```

Runtime, pinned model revision, package versions and server settings are under `scratch/runtime/qwen3.6-35b-a3b/`. Download/compilation caches and outputs stay under scratch. Defaults: localhost:8000, one GPU, BF16 nonquantized components, checkpoint-defined NVFP4/FP8 weights and 16-bit MoE activations, 32K context, eight scheduled sequences and 50% GPU memory. The service lifetime is limited by the Slurm allocation. `serve.py --stop` verifies the recorded process identity before stopping it; run it on the serving node.

## Thinking and caching

Thinking defaults to **false** at both the server and Python transport. Override per call with `VLLMQwen(...).chat(messages, enable_thinking=True)`; the transport sends `chat_template_kwargs.enable_thinking` and returns separate `reasoning` when provided by vLLM. `--enable-thinking` / `--no-enable-thinking` are available on inference, observation and batch CLIs. Dialogue generation also supports `--model-mode thinking` (default is instruct/nonthinking). No artificial `<think>` prefill is sent to vLLM.

vLLM owns prefix/KV caching and request isolation. Qwen3.6's hybrid attention uses `--enable-prefix-caching --mamba-cache-mode align`; this mode is experimental upstream. The custom `qwen_prefix_cache.py` is **not used** on this path. It is retained for the Mac/native Qwen3-VL tools that still use it; Qwen3.6 native calls reject this legacy cache, and tool wrappers omit it for newer models. Structured requests always use the normal engine generation path.

`Qwen(runtime=...)` remains an optional native Transformers adapter; its Linux default model is the unquantized `Qwen/Qwen3.6-35B-A3B`, distinct from the deployed NVFP4 service. Prefer `VLLMQwen` for CUDA work; setup downloads only the deployed NVIDIA checkpoint.

## Two-stage generation and batching

```bash
python tools/video_observer.py scratch/data/samples/story_example.mp4 \
  --backend vllm --output-dir scratch/runs/observation
python scripts/qwen/observe_batch.py video_a.mp4 video_b.mp4 \
  --output-dir scratch/runs/observations --workers 2
python scripts/dialogue_generation/generate.py \
  scripts/dialogue_generation/examples/tokyo_vlog.json \
  --backend vllm --output-dir scratch/runs/dialogue --seed 1
python scripts/dialogue_generation/batch.py \
  scripts/dialogue_generation/examples/tokyo_vlog.json \
  scripts/dialogue_generation/examples/alpine_kayak_vlog.json \
  --output-dir scratch/runs/dialogues --workers 2
```

Observation samples video frames, inventories subjects/shots, extracts and reviews attributes, repairs recoverable JSON and validates the final observation contract. Dialogue generation takes real or virtual observation JSON, plans in Python, writes/verifies spoken dialogue, then rewrites every executable placeholder's generation prompt. `dialogue.json`, `dialogue.md`, `placeholders.json` and `generation_plan.json` contain the results. Batch reports preserve input/output association and fail the command if any input fails.

Raw ordered multimodal messages and mixed batches are also supported:

```python
from cocreation import VLLMQwen
with VLLMQwen() as model:
    answer = model.chat(messages)  # thinking is false
    batch = model.generate_batch([messages, other_messages], concurrency=2)
```

Media file paths refer to the serving host and must be inside the scratch root. Raw video sampling defaults to 8 FPS, at most 64 frames, uniformly spanning the full clip including both endpoints. The server, serving transport and optional native adapter share these defaults from `cocreation/qwen_inputs.py`. Server media I/O sets both `num_frames` (generic loader) and `max_frames` (Qwen3-VL loader) so the limit applies to either backend. The processor preserves the already sampled frames instead of sampling them a second time. A 4.46-second clip supplies about 35 distinct frames instead of four; clips reaching the frame limit still cover their full duration. Observation samples its own frame images independently (24 by default). The transport records sampling settings alongside the wire request and response, and real-prompt batches record them in `vlm_execution.json`. Keep explicit client and server overrides synchronized.

The fixed Qwen-based text encoders bundled with H3/VDN are video-generation dependencies, not replaceable Qwen inference checkpoints; they are retained.


`run_pipeline.sh` explicitly disables thinking on the server and each observation/dialogue/prompt-writing request for the next experiment (review of sequence 9242687). Optional thinking support and its hard token budget remain available.
Its default media layout is square images and landscape videos. Dialogue rules come from the observation or the planner.

Thinking requests now send vLLM's `thinking_token_budget=min(2048, max_new_tokens//2)` by default. The server forces the end of the reasoning block at that limit so generation can continue with the final answer. `max_new_tokens` still covers both reasoning and the answer. Override the reasoning cap per request with `thinking_token_budget=1024`, for example; keep it below the total token budget. Non-thinking requests omit this option.

Observation preserves response metadata and reports truncated or empty final responses before JSON parsing. A reasoning-only response that exhausts the token limit is retried with an explicit completion instruction, retaining all raw responses and errors. This fixes the misleading `Expecting value` failure in job 9238547.
