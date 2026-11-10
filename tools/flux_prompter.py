"""Write a FLUX.2-klein prompt from selected attributes and image/video sources."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen
from tools._prompter import run_prompter
from tools.flux_prompter_prompts import flux_system_prompt, flux_user_prompt


def flux_prompter(attributes, references=None, user_prompt='', *, qwen=None,
                  output_dir=None, use_system_cache=True, max_new_tokens=1024,
                  video_fps=2.0, video_max_frames=32, max_pixels=512*512):
    """Return expert prompt str, or None for unfulfillable/invalid model output.

    Args:
        attributes: List of global id/category/text records (e.g. video_0_bear); artifact_ids lists sources,
            or is omitted/empty for new desired properties.
        references: Ordered image/video References or artifact_id/path/kind mappings.
            Every source must belong to at least one selected attribute.
        user_prompt: Optional creative brief; attributes remain explicit constraints.
        qwen: Optional reusable instance; only internally created instances are closed.
        output_dir: Qwen run-log root when this tool owns the instance.
        use_system_cache: Enable this tool's dedicated fixed-system KV cache.
        max_new_tokens: Qwen response budget (JSON envelope plus expert text).
        video_fps: Sampling rate for Qwen's original-video observation.
        video_max_frames: Maximum sampled frames per source video.
        max_pixels: Observation image/frame pixel budget.

    Rejected model outputs print a precise diagnostic and raw text to stderr;
    output_dir also receives diagnostics.jsonl. Input/operational errors raise. No generation or frame extraction happens here.
    Output <Image N> addresses reference slot N across BOTH source kinds: replace
    each video with one matching image before calling Flux, preserving all slots.
    """
    return run_prompter('flux', attributes, references, user_prompt, qwen=qwen,
        factory=Qwen, system_builder=flux_system_prompt, user_builder=flux_user_prompt,
        output_dir=output_dir, use_system_cache=use_system_cache,
        max_new_tokens=max_new_tokens, video_fps=video_fps,
        video_max_frames=video_max_frames, max_pixels=max_pixels)


if __name__ == '__main__':
    from tools._prompter_cli import main
    main(flux_prompter, 'flux')
