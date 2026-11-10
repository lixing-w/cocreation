"""Observe one artifact as a selective, line-separated natural-language description."""
from cocreation.storage import scratch_root
from tools._attribute_vlm import ask_text, reference, observation_result
from tools.extract_attribute_nodes_prompts import extraction_system_prompt


def extractor(artifact, *, qwen=None, output_dir=None, use_system_cache=True,
              max_new_tokens=4096, video_fps=2.0, video_max_frames=32,
              max_pixels=512*512):
    ref = reference(artifact)
    return observation_result(ask_text(extraction_system_prompt(),
        'Describe the essential visible attributes of this ' + ref.kind + '.', [ref],
        qwen=qwen, output_dir=output_dir if output_dir is not None else scratch_root()/'runs/extractor',
        cache_name='extractor', use_system_cache=use_system_cache,
        max_new_tokens=max_new_tokens, video_fps=video_fps,
        video_max_frames=video_max_frames, max_pixels=max_pixels))
