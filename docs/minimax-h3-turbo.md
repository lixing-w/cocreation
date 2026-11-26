# MiniMax-H3 Turbo on CUDA 12.8

The sequence pipeline uses original H3-Base-Ref2VA through ComfyUI, with the Comfy-Org AdaLN-pruned FP8 conversion. It uses neither a VDN model nor int8_convrot/CUDA 13.0.

- DiT: `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video/audio VAEs: existing FP16 video VAE and FP32 audio VAE
- LoRA: `minimax_h3_turbo_v4_step600_ema.safetensors`
- Sampling: 7 steps, strength 1.0, low_vram=False, CFG 1.0, simple scheduler, MiniMaxH3TurboSampler with SamplerCustomAdvanced
- Torch: 2.10.0+cu128; driver 575.57.08

Revisions are pinned in `cocreation/h3.py`. The Turbo extension alone is whitelisted; other custom nodes remain disabled. Model downloads and preflight use the same scratch Hugging Face cache as the resident worker.

The official generation range is **4–15 seconds**, with video/audio references **2–15 seconds** each. ComfyUI aligns durations upward to its 17k+5 frame grid at 24 FPS; a 4-second request generates 107 frames (~4.458 seconds), and materialization keeps the requested 4 seconds. Short video references are held to 2 seconds without losing source frames. Native generated stereo audio is retained except for visual video edits that explicitly preserve their source soundtrack.

`materialize.py` uses stable topological scheduling, drains ready CPU operations, and exhausts all available tasks for the resident model before switching. It evaluates the starting model to avoid an unnecessary initial/final phase. Identical implicit and explicit model IDs share a worker. Dialogue/catalog order and clip assembly order are unchanged. Dependencies from generated video to later image references still cause necessary switches.

On the previous saved sequence `sequence-9249692-20261117T231043Z`, the 20-asset plan requires 1 model switch instead of the previous DFS order's 5. Unit coverage compares small two-model DAGs with exhaustive optimal schedules, and covers forced video-to-image dependencies, intermediate CPU operations, and the Turbo graph.

Validation logs: `scratch/runs/h3-turbo-validation-9256419/`.

Interactive allocation:

```bash
interact -n 6 -t 6:30:00 -a ssrinath-gcondo -q gpu-he -m 64g -g 1
```

Job 9256419 on gpu4105 uses one 96GB RTX PRO 6000 Blackwell. The story_example.mp4 rerun is under `scratch/runs/sequence-h3turbo-9256419-20261118T024905Z/`; its `status.json` and `pipeline.log` describe live progress.

Sources: [original model specifications](https://huggingface.co/MiniMaxAI/MiniMax-H3), [Comfy-Org weights](https://huggingface.co/Comfy-Org/MiniMax-H3), [Turbo v4-600 usage](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
