"""Lightweight H3 contract and pinned Comfy API graph; no ML imports."""
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

from .storage import scratch_root

COMFY_REVISION = "0decb39ee5b5e17c200334589fc317dbbe98310e"
MODEL_REPO = "Comfy-Org/MiniMax-H3"
MODEL_REVISION = "54472221d27c695fca9621b0c3ed52f06c4af707"
ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


@dataclass(frozen=True)
class H3Runtime:
    root: str = field(default_factory=lambda: str(scratch_root() / "runtime/minimax-h3"))
    python: str = sys.executable
    auto_setup: bool = True
    local_files_only: bool = False
    accept_model_license: bool = False
    timeout: float = 7200
    startup_timeout: float = 300

    def __post_init__(self):
        if self.timeout <= 0 or self.startup_timeout <= 0:
            raise ValueError("H3 timeouts must be positive")


def model_files(variant):
    if variant not in ("fl2va", "ref2va"):
        raise ValueError("Unknown H3 variant")
    return [f"diffusion_models/minimax_h3_{variant}_pruned_fp8_scaled.safetensors",
            f"text_encoders/{ENCODER}", f"vae/{VIDEO_VAE}", f"vae/{AUDIO_VAE}"]


def frame_count(seconds):
    return math.ceil((seconds * 24 - 5) / 17) * 17 + 5


def build_graph(request, media):
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
    clip = add("encoder", "CLIPLoader", clip_name=ENCODER, type="minimax", device="default")
    vae = add("video_vae", "VAELoader", vae_name=VIDEO_VAE)
    audio_vae = add("audio_vae", "VAELoader", vae_name=AUDIO_VAE)
    inputs = dict(clip=clip, vae=vae, prompt=request["prompt"], width=opts["width"],
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
                # Visual motion only, matching the public Reference contract.
            else:
                value = add(key, "LoadAudio", audio=filename)
                inputs[f"ref_audios.ref_audio_{index}"] = value
        conditioning = add("condition", "MiniMaxH3ReferenceToVideo", **inputs)
    else:
        conditioning = add("condition", "MiniMaxH3ImageToVideo", **inputs)
    shifted = add("shift", "MiniMaxH3SigmaShift", model=model, shift_video=12.0, shift_audio=3.0)
    negative = add("negative", "ConditioningZeroOut", conditioning=conditioning)
    samples = add("sample", "KSampler", model=shifted, positive=conditioning, negative=negative,
                  latent_image=["condition", 1], seed=opts["seed"], steps=opts["steps"],
                  cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    split = add("split", "LTXVSeparateAVLatent", av_latent=samples)
    images = add("decode_video", "VAEDecodeTiled", samples=split, vae=vae,
                 tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    audio = add("decode_audio", "VAEDecodeAudio", samples=["split", 1], vae=audio_vae)
    video = add("video", "CreateVideo", images=images, fps=24.0, audio=audio)
    add("save", "SaveVideo", video=video, filename_prefix="video", format="mp4", **{"format.codec": "h264"})
    return graph
