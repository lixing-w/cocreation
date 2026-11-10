"""Write a FLUX.2-klein prompt from selected attributes and image/video sources."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen
from tools._prompter import run_prompter
from tools._attribute_protocol import media_catalog, MEDIA_LABEL
from tools._attribute_vlm import reference
from tools.flux_prompter_prompts import flux_system_prompt, flux_user_prompt


def flux_prompter(attributes, references=None, user_prompt='', *, qwen=None,
                  output_dir=None, use_system_cache=True, max_new_tokens=1024,
                  video_fps=2.0, video_max_frames=32, max_pixels=512*512):
    """Return expert prompt str, or None for unfulfillable/invalid model output.

    Args:
        attributes: Natural-language description, one element per line; retained
            source elements use exact (from <Image N>) / (from <Video N>) markers.
        references: Ordered image/video References. Every source must be used.
        user_prompt: Optional creative brief; attributes remain explicit constraints.
        qwen: Optional reusable instance; only internally created instances are closed.
        output_dir: Qwen run-log root when this tool owns the instance.
        use_system_cache: Enable this tool's dedicated fixed-system KV cache.
        max_new_tokens: Qwen response budget (direct expert text).
        video_fps: Sampling rate for Qwen's original-video observation.
        video_max_frames: Maximum sampled frames per source video.
        max_pixels: Observation image/frame pixel budget.

    Rejected model outputs print a precise diagnostic and raw text to stderr;
    output_dir also receives diagnostics.jsonl. Input/operational errors raise. No generation or frame extraction happens here.
    Output preserves original <Video N> and <Image N> observation labels.
    After replacing each video with a matching still, call flux_generation_prompt
    to map those labels to Flux's image slots in Python, immediately before Flux.
    """
    return run_prompter('flux', attributes, references, user_prompt, qwen=qwen,
        factory=Qwen, system_builder=flux_system_prompt, user_builder=flux_user_prompt,
        output_dir=output_dir, use_system_cache=use_system_cache,
        max_new_tokens=max_new_tokens, video_fps=video_fps,
        video_max_frames=video_max_frames, max_pixels=max_pixels)


def flux_generation_prompt(prompt, references, generation_references):
    """Translate original labels simultaneously, after frame selection, without a VLM.

    references is the original prompter input; generation_references is the same
    ordered artifact list with video paths replaced by selected images. Never
    derive source labels from that all-image list: mixed-kind numbering differs.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Expected nonempty Flux prompter text')
    refs = [reference(r) for r in references or []]
    images = [reference(r) for r in generation_references or []]
    if ([r.artifact_id for r in refs] != [r.artifact_id for r in images]
            or any(r.kind != 'image' for r in images)):
        raise ValueError('Flux images must preserve source order and contain no videos')
    labels = media_catalog(refs)
    mapping = {labels[r.artifact_id]: f'<Image {i}>' for i, r in enumerate(refs, 1)}
    found = set(MEDIA_LABEL.findall(prompt))
    if found != set(mapping):
        raise ValueError('Prompt must use exactly the original source labels')
    # One pass prevents <Video 1> -> <Image 1> -> <Image 2> cascading replacements.
    return MEDIA_LABEL.sub(lambda match: mapping[match.group()], prompt)


if __name__ == '__main__':
    from tools._prompter_cli import main
    main(flux_prompter, 'flux')
