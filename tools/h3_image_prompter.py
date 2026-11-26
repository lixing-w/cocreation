"""Write MiniMax-H3-Image prompts using the official skill and selected text only."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen, VLLMQwen
from tools._prompter import run_prompter, LABEL
from tools._attribute_vlm import reference
from tools.h3_image_prompter_prompts import h3_image_system_prompt, h3_image_user_prompt


def h3_image_prompter(attributes, references=None, user_prompt='', *, qwen=None,
                      output_dir=None, use_system_cache=False, max_new_tokens=2048,
                      video_fps=2.0, video_max_frames=32, max_pixels=512*512):
    """Return direct generation prose; no media is sent to the prompt-writing Qwen."""
    return run_prompter('h3_image', attributes, references, user_prompt, qwen=qwen,
        factory=VLLMQwen if sys.platform.startswith('linux') else Qwen,
        system_builder=lambda: h3_image_system_prompt(bool(references)),
        user_builder=h3_image_user_prompt, output_dir=output_dir,
        use_system_cache=use_system_cache, max_new_tokens=max_new_tokens,
        video_fps=video_fps, video_max_frames=video_max_frames, max_pixels=max_pixels)


def h3_image_generation_prompt(prompt, references, generation_references):
    """Map prompter source labels to <Picture N> after videos become stills."""
    refs = [reference(r) for r in references or []]
    images = [reference(r) for r in generation_references or []]
    if [r.artifact_id for r in refs] != [r.artifact_id for r in images] or any(r.kind != 'image' for r in images):
        raise ValueError('Generation images must retain source identity and order')
    from tools._attribute_protocol import media_catalog
    labels = media_catalog(refs)
    mapping = {(labels[r.artifact_id].replace('<Image ', '<Picture ')): f'<Picture {i}>'
               for i, r in enumerate(refs, 1)}
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Expected nonempty H3 Image prompt')
    found = {m.group() for m in LABEL.finditer(prompt) if not m.group().startswith('<Subject ')}
    if found != set(mapping):
        raise ValueError('Prompt must use exactly the supplied source labels')
    return LABEL.sub(lambda m: mapping.get(m.group(), m.group()), prompt)


if __name__ == '__main__':
    from tools._prompter_cli import main
    main(h3_image_prompter, 'h3_image')
