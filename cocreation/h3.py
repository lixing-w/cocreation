"""Lightweight H3 contract and pinned Comfy API graph; no ML imports."""
from dataclasses import dataclass, field
import math
import re
from pathlib import Path
import sys

from .storage import scratch_root

COMFY_REVISION = "0decb39ee5b5e17c200334589fc317dbbe98310e"
MODEL_REPO = "Comfy-Org/MiniMax-H3"
MODEL_REVISION = "54472221d27c695fca9621b0c3ed52f06c4af707"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
TURBO_REVISION = "43a74557ac3f6539db8e0f2a959d03feb7a81480"
TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
TURBO_NODE_REVISION = "4274783a23afcfdbea3b4876cb79effd6c510785"
TURBO_NODE_FOLDER = "ComfyUI-MiniMax-H3-Turbo"


@dataclass(frozen=True)
class H3Runtime:
    root: str = field(default_factory=lambda: str(scratch_root() / "runtime/minimax-h3"))
    python: str = sys.executable
    auto_setup: bool = True
    local_files_only: bool = False
    accept_model_license: bool = True
    turbo: bool = False
    turbo_low_vram: bool = False
    timeout: float = 7200
    startup_timeout: float = 300

    def __post_init__(self):
        if type(self.turbo) is not bool or type(self.turbo_low_vram) is not bool:
            raise ValueError("H3 turbo switches must be booleans")
        if self.timeout <= 0 or self.startup_timeout <= 0:
            raise ValueError("H3 timeouts must be positive")


def model_files(variant):
    if variant not in ("fl2va", "ref2va"):
        raise ValueError("Unknown H3 variant")
    return [f"diffusion_models/minimax_h3_{variant}_pruned_fp8_scaled.safetensors",
            f"text_encoders/{ENCODER}", f"vae/{VIDEO_VAE}", f"vae/{AUDIO_VAE}"]


def frame_count(seconds):
    return math.ceil((seconds * 24 - 5) / 17) * 17 + 5


def default_steps(turbo=False):
    """Original H3 baseline; Turbo v4-600 uses its own few-step schedule."""
    return 6 if turbo else 50


def build_graph(request, media, media_info=None):
    """media: staged Comfy input filenames, in request reference order."""
    opts = request["options"]
    if len(media) != len(request["references"]):
        raise ValueError("Staged media count differs from references")
    graph = {}
    def add(key, cls, **inputs):
        graph[key] = {"class_type": cls, "inputs": inputs}
        return [key, 0]
    model = add("model", "UNETLoader", unet_name=Path(model_files(opts["variant"])[0]).name,
                weight_dtype="default")
    turbo = request.get("runtime", {}).get("turbo", False)
    if turbo:
        model = add("turbo_lora", "MiniMaxH3TurboLoRA", model=model,
                    lora_name=TURBO_LORA, strength=1.0,
                    low_vram=request.get("runtime", {}).get("turbo_low_vram", False))
    clip = add("encoder", "CLIPLoader", clip_name=ENCODER, type="minimax", device="default")
    vae = add("video_vae", "VAELoader", vae_name=VIDEO_VAE)
    audio_vae = add("audio_vae", "VAELoader", vae_name=AUDIO_VAE)
    prompt = request["prompt"]
    # Comfy numbers video soundtracks before standalone audio references.
    soundtrack_count = sum(bool(item.get("video_audio_used")) for item in (media_info or []))
    if soundtrack_count:
        prompt = re.sub(r"<Audio (\d+)>", lambda m: f"<Audio {int(m[1]) + soundtrack_count}>", prompt)
    inputs = dict(clip=clip, vae=vae, prompt=prompt, width=opts["width"],
                  height=opts["height"], length=opts["frames"])
    if opts["variant"] == "ref2va":
        inputs.update(audio_vae=audio_vae, ref_image_size="match")
        counts = dict(image=0, video=0, audio=0)
        for i, (ref, filename) in enumerate(zip(request["references"], media)):
            kind = ref["kind"]
            index = counts[kind]
            counts[kind] += 1
            key = f"media_{i}"
            if kind == "image":
                value = add(key, "LoadImage", image=filename)
                inputs[f"ref_images.ref_image_{index}"] = value
            elif kind == "video":
                video = add(key, "LoadVideo", file=filename)
                value = add(key + "_components", "GetVideoComponents", video=video)
                inputs[f"ref_videos.ref_video_{index}"] = value
                inputs[f"ref_video_audios.ref_video_audio_{index}"] = [key + "_components", 1]
            else:
                value = add(key, "LoadAudio", audio=filename)
                inputs[f"ref_audios.ref_audio_{index}"] = value
        conditioning = add("condition", "MiniMaxH3ReferenceToVideo", **inputs)
    else:
        conditioning = add("condition", "MiniMaxH3ImageToVideo", **inputs)
    shifted = add("shift", "MiniMaxH3SigmaShift", model=model, shift_video=12.0, shift_audio=3.0)
    negative = add("negative", "ConditioningZeroOut", conditioning=conditioning)
    if turbo:
        noise = add("noise", "RandomNoise", noise_seed=opts["seed"])
        guider = add("guider", "CFGGuider", model=shifted, positive=conditioning,
                     negative=negative, cfg=1.0)
        sampler = add("turbo_sampler", "MiniMaxH3TurboSampler")
        sigmas = add("sigmas", "BasicScheduler", model=shifted, scheduler="simple",
                     steps=opts["steps"], denoise=1.0)
        samples = add("sample", "SamplerCustomAdvanced", noise=noise, guider=guider,
                      sampler=sampler, sigmas=sigmas, latent_image=["condition", 1])
    else:
        samples = add("sample", "KSampler", model=shifted, positive=conditioning, negative=negative,
                      latent_image=["condition", 1], seed=opts["seed"], steps=opts["steps"],
                      cfg=1.0, sampler_name=opts.get("sampler", "euler"), scheduler="simple", denoise=1.0)
    split = add("split", "LTXVSeparateAVLatent", av_latent=samples)
    images = add("decode_video", "VAEDecodeTiled", samples=split, vae=vae,
                 tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    audio = add("decode_audio", "VAEDecodeAudio", samples=["split", 1], vae=audio_vae)
    video = add("video", "CreateVideo", images=images, fps=24.0, audio=audio)
    add("save", "SaveVideo", video=video, filename_prefix="video", format="mp4", **{"format.codec": "h264"})
    return graph
