# MiniMax-H3-Image on Oscar

New dialogue plans use `h3_image` for still images. Flux, its prompt writer and
cached checkpoints remain available, and existing Flux catalogs still execute.
Video inference continues to use the existing H3 Turbo installation.

The image stack is isolated under `scratch/runtime/minimax-h3-image/`:

- Comfy-Org `minimax_h3_image_pruned_w4a8.safetensors`: 4-bit weights, 8-bit activations, about 12.55 GB.
- Existing `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`: reused through a symlink.
- Image-specific `minimax_h3_image_vae_fp16.safetensors`: about 1.17 GB.
- PyTorch **2.10.0+cu128**, torchvision **0.25.0+cu128**.
- Independent ComfyUI checkout, pinned in `cocreation/h3_image.py`; no custom nodes.

The checkpoint owns the native six-step PDD schedule: **6 steps, CFG 1,
Euler / simple**. CPU-seeded initial noise follows the upstream recommendation.
Recommended canvases are 4096×4096 square, 5472×3072 landscape and 3072×5472
portrait. Materialization preserves the native PNG dimensions. Explicit image
overrides must have sides divisible by 32 and 4–20 MP area. Older Flux catalogs
keep their existing defaults.

Run setup and every test **inside an allocated GPU session**, never a login node:

```bash
cd /users/lwang223/cocreation
module load miniforge3/25.3.0-3 cuda/12.9.0-cinr
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate cocreation
python scripts/setup/setup_h3_image.py
python -m unittest tests.test_h3_image
```

The installer pins CUDA 12.8 wheels and records package versions, revisions and
weight provenance. CUDA 13 is not installed. Inference is offline and rejects
execution without an allocated Slurm GPU. Release interactive jobs with
`scancel JOB_ID` immediately after finishing.

```python
from cocreation import MiniMaxH3Image, Reference

with MiniMaxH3Image() as image:
    first = image("A studio product photograph of a blue ceramic vessel on a neutral backdrop.", seed=42)
    edited = image("The vessel in <Picture 1> is <Subject 1>. The target image has an amber vessel. Preserve its shape, material and all other details.",
                   [Reference("vessel", first["path"], "image")], seed=43)
```

`tools/h3_image_prompter.py` accepts selected text and ordered sources, like the
Flux writer. On Linux it uses the shared Qwen3.6 vLLM service. The prompt writer
receives text only. Video sources become stills before generation, then
`h3_image_generation_prompt` maps source labels to ordered `<Picture N>` slots in
one pass. Placeholder rewriting already uses final generation slots.

The official skill is vendored unchanged with revision and SHA-256 provenance
under `skills/h3-image-prompt-writing/`. Qwen receives the relevant mode's rules
without worked examples or unrelated source captions. Local edits send new
values and current presentation; old values and planning history are omitted.
Current structured asset attributes replace stale prose.

Upstream references:

- [MiniMax parameters and prompt skill](https://github.com/MiniMax-AI/MiniMax-H3-Image)
- [Comfy-Org quantized weights](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
- [Official ComfyUI workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json)
