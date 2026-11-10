"""Creative additions conditioned on retained attributes and source media."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._attribute_protocol import repair_additions, assemble
from tools._attribute_vlm import ask_json, reference

from tools.imaginer_prompts import SYSTEM, imaginer_payload


def imaginer(selected_attributes, add_count, references=None, *, kind, qwen=None,
             output_dir=None, seed=0, max_new_tokens=2048, video_fps=2.0,
             video_max_frames=32, max_pixels=512*512):
    """Return list[{category,text}], [] for V=0, or None on null/malformed output.

    Input is atomize() records after dependency-closed sampling. All source refs
    are current carriers, not earliest provenance. Shared Qwen is never closed.
    Invalid entries are dropped; fewer than V (including zero) is valid.
    Operational failures raise. There are no internal retries or tool decisions.
    """
    assemble(selected_attributes, 'selection', kind)
    if type(add_count) is not int or add_count < 0:
        raise ValueError('Addition count must be a nonnegative integer')
    if not add_count:
        return []
    refs = [reference(r) for r in (references or [])]
    if {r.artifact_id for r in refs} != {a['carrier_artifact_id'] for a in selected_attributes}:
        raise ValueError('Attach exactly the selected source media')
    result = ask_json(SYSTEM, imaginer_payload(selected_attributes, add_count, refs, kind, seed), refs,
        qwen=qwen, output_dir=output_dir, seed=seed, max_new_tokens=max_new_tokens,
        video_fps=video_fps, video_max_frames=video_max_frames, max_pixels=max_pixels)
    if result is None:
        return None
    try:
        if not isinstance(result, dict) or 'additions' not in result:
            raise ValueError('Expected additions object')
        return repair_additions(selected_attributes, result['additions'], add_count, kind)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        print(f'imaginer: invalid response: {error}', file=sys.stderr)
        return None
