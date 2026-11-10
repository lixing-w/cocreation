"""Write a VDN-H3 prompt from selected attributes and image/video sources."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen
from tools._prompter import run_prompter
from tools.h3_prompter_prompts import h3_system_prompt, h3_user_prompt


def h3_prompter(attributes, references=None, user_prompt='', *, seconds=15.0,
                qwen=None, output_dir=None, use_system_cache=True, max_new_tokens=4096,
                video_fps=2.0, video_max_frames=32, max_pixels=512*512):
    """Return H3 expert prompt str, or None for unfulfillable/invalid model output.

    Args:
        attributes: Natural-language description, one element per line; retained
            source elements use exact (from <Image N>) / (from <Video N>) markers.
        references: Ordered image/video References. Every source must be used.
        user_prompt: Optional creative brief for the target video.
        seconds: Target VDN-H3 video duration, 5-15 seconds.
        qwen: Optional shared instance; never closed by this tool.
        output_dir: Run-log root for an internally created Qwen instance.
        use_system_cache: Reuse the dedicated H3 writing-system prefix.
        max_new_tokens: Qwen output budget for the complete H3 prompt.
        video_fps: Qwen observation sampling rate, independent of target frame rate.
        video_max_frames: Maximum observed frames per source video.
        max_pixels: Qwen image/frame pixel budget.

    Pass the resulting text and SAME ordered references directly to VDNH3, using
    the same seconds. This tool observes visual content only, not video audio.
    Rejected model outputs print diagnostics/raw text to stderr and, when output_dir
    is set, diagnostics.jsonl. Input/inference failures raise; an owned instance is closed on every exit.
    """
    return run_prompter('h3', attributes, references, user_prompt, qwen=qwen,
        factory=Qwen, system_builder=h3_system_prompt, user_builder=h3_user_prompt,
        output_dir=output_dir, use_system_cache=use_system_cache,
        max_new_tokens=max_new_tokens, video_fps=video_fps,
        video_max_frames=video_max_frames, max_pixels=max_pixels, seconds=seconds)


if __name__ == '__main__':
    from tools._prompter_cli import main
    main(h3_prompter, 'h3')
