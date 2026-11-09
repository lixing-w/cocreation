# VDN-H3 through the FreeVideo CLI

The `VDNH3` wrapper supports ordered image, video and audio references. It runs
FreeVideo's native CLI, not the ComfyUI web server. ComfyUI remains an encoder-library
dependency. Existing Qwen, Flux and official MiniMaxH3 backends are unchanged.

Defaults: FP8 per-tensor VDN-H3 edge cache, two-pass **8 + 3** with 2x latent
upscaling, NVFP4-AWQ Qwen3-VL encoder with FreeVideo's BF16 compute patch. VAE loading,
FP16 decode autocast, FP32 normalization and audio VAE follow FreeVideo unchanged.
Do not cast the whole VAE to half precision. No optional approximate step cache.

## Server setup

Requirements: Linux NVIDIA, an existing controller Python, git, ffmpeg/ffprobe, a working C compiler,
compatible NVIDIA driver, disk for Hugging Face assets plus runtime caches. Use fast
local storage. Download volume is substantial; FP8 avoids downloading the original
BF16 transformer. Cross-filesystem assets are copied instead of hard-linked, so keep
the HF cache and runtime root on the same filesystem to avoid duplicates.

The controller may use Python 3.11. By default setup installs uv in
`scratch/runtime/vdn/bootstrap`, downloads Python 3.12 into
`scratch/runtime/vdn/python`, and creates `scratch/runtime/vdn/venv/bin/python3.12`.
No system Python 3.12 or global uv installation is needed. The venv references the
managed base interpreter, so keep both directories. Subsequent calls reuse them.
Optionally use `setup_vdn.py --python /path/to/python3.12` or
`VDNRuntime(bootstrap_python="/path/to/python3.12")` to use an existing interpreter.
Configure the repository's scratch directory first as described in the main README.
Model download/use requires accepting the MiniMax H3 model license; this project
retains its existing `accept_model_license` runtime convention.

```bash
python scripts/setup_vdn.py --probe
python examples/plan.py --model vdn_h3 --dry-run
python examples/plan.py --model vdn_h3 --run
```

`--runtime-only` installs code and dependencies without downloading models.
`--offline` requires an already installed runtime and locally cached model files.
The kernel probe does not validate a full generation; run the example afterwards.
For a shorter first GPU test, use the Python call below with a video reference no
longer than 5 seconds, or omit the video and supply two images.

```python
from cocreation import VDNH3, VDNRuntime, Reference

model = VDNH3(runtime=VDNRuntime())
result = model(
    "Use the subject in {{artifact:person}} and setting in {{artifact:place}}. "
    "Follow the camera motion in {{artifact:motion}}.",
    [Reference("person", "/data/person.png", "image"),
     Reference("place", "/data/place.png", "image"),
     Reference("motion", "/data/motion.mp4", "video")],
    seconds=5, width=768, height=512, seed=42,
)
print(result["path"])
```

Output duration is aligned to the H3 17*n+5 frame grid at 24 fps (5 seconds becomes
124 frames). Two-pass dimensions must be multiples of 64 and at least 512 pixels
per side. Up to 32 references are accepted by the contract, not guaranteed to fit
in memory. Default budgets are 80 GiB VRAM / 64 GiB RAM, with 3 / 8 GiB reserves;
FreeVideo also considers actual detected/free capacity. Adjust these runtime fields
if the server's quota differs. More/longer references increase memory and latency.

Video soundtracks are enabled (`sound: true`); standalone audio references are also supported.
Qwen3-VL continues to read video frames only. Non-image references
must be 1–15 seconds and no longer than the aligned target clip; the wrapper rejects
longer clips instead of letting upstream silently trim them. Frames still undergo
FreeVideo's normal 24-fps and H3 grid normalization. This is reference-conditioned
VDN generation, not a guarantee of frame-exact video editing.

## CUDA 12.8 compatibility changes

The installer pins FreeVideo `7c3a03c1a93580454f71628b78ac3df9d9e1880e`, its VDN
`30b6b380c2482f3519469350810c2955d8847fd9`, ComfyUI
`421a1c245c682c04d4325ba365f40c834c66f5b0`, and the VDN-patched Diffusers base
`3a2f35d4efa4c059c8bfb3bc0d6c906264895c81`.

- Bypass `setup.sh`, the launcher installer, and `[runtime]` extra. They select
  newer Torch/CUDA/Triton versions. Install into `scratch/runtime/vdn/venv` only.
- Constrain torch **2.10.0+cu128**, torchvision **0.25.0+cu128**, torchaudio
  **2.10.0+cu128**, Triton **3.6.0** on every pip invocation. Never upgrade the
  caller's environment. A conflicting dependency fails installation, not the pin.
- Keep Comfy's native comfy-kitchen CUDA backend disabled on cu128; replace the
  misleading upgrade warning with an accurate fallback message. Do not fake a CUDA
  version or enable a cu130 binary. Keep its eager dequantization path available.
- Disable Comfy's optional Triton backend, dynamic VRAM, CUDA graphs and compiler
  in the encoder child. Use PyTorch attention there. FreeVideo's own Triton kernels
  remain necessary and use Torch's paired Triton version.
- Limit FreeVideo's backend discovery to `torch-flash` and `cudnn`; default to
  `torch-flash`. Do not install FA4/SageAttention CUDA extensions. Set
  `VDNRuntime(attention="cudnn")` explicitly if server probes justify it.
- Force the pinned FP8 `per_tensor` prepared model, verify artifact SHA256/size,
  and reject INT8 or incompatible manifests. Include the learned latent upscaler
  and 8/3-step sampling tables, including audio-reference tables.

Patches are exact-source and idempotent. Runtime receipts record patch hashes and
installed package versions. This is a compatibility adaptation, **not a GPU-validated
port**. Torch/Triton kernel compilation, NVFP4 eager dequantization, full mixed-reference
generation and peak memory still need testing on the server. Installation also needs
a real Linux dependency resolution run; Mac contract tests cannot validate CUDA wheels.
If installation fails, do not remove the torch constraint: retain the error so the
specific dependency can be patched or pinned.

## Lifecycle and diagnostics

Each invocation creates a fresh CLI worker tree and terminates its process group
on completion, timeout or failure. A file lock serializes this backend across local
processes. It does not coordinate external Qwen/Flux services. There is no persistent
service or sleep/wake promise in this first version; runtime preparation and CLI
execution are separated for a future worker implementation.

Each job keeps `request.json`, `prompt.txt`, `media.json`, `command.json`, `runtime.log`,
`result.json` or `error.json`, plus FreeVideo's `video.request.json`, engine logs and
any `video.debug.json` it produces. `generation_seconds` excludes initial setup.
For server debugging send the failing log and error report; `runtime-ready.json`,
`installed-packages.txt`, and `gpu-probe.json` describe the installation.

Upstream: https://github.com/FlashML-org/FreeVideo and https://github.com/OpenVDN/vdn-minimax-h3.
Sources are fetched at setup into scratch; no third-party source or model weights
are vendored into this repository. Upstream license/notice files remain in those checkouts.
