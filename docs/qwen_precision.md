# Qwen3.6 precision

The deployed checkpoint is [NVIDIA Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4). Its model configuration combines FP8 attention/linear components and NVFP4 MoE weights with 16-bit MoE activations (W4A16) with unquantized components. It retains image/video support. NVFP4 is a hardware-oriented 4-bit floating-point format; it is different from integer AWQ/GPTQ and GGUF Q4_K_M.

On the allocated B200, vLLM uses Marlin for W4A16 MoE, configured FP8 components, FlashInfer attention, CUDA graphs and continuous batching. This checkpoint does not quantize MoE activations to FP4; we retain its calibrated configuration. Setup pins the CUDA 12.9 wheel for the installed 575 driver. Actual speed depends on inputs, video preprocessing, context and concurrency. This migration runs functional acceptance only; no speed or quality benchmark was run and no throughput claim is made.

The model revision and installed package versions are recorded in `scratch/runtime/qwen3.6-35b-a3b/model.json` and `vllm-packages.txt`.
