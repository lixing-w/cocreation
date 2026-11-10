"""Creative text-only patches to selected attribute atoms, grounded in source media."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._attribute_protocol import apply_variations, assemble
from tools._attribute_vlm import ask_json, reference

from tools.imaginer_prompts import SYSTEM, imaginer_payload


def imaginer(selected_attributes, vary_ids, references=None, *, kind, qwen=None,
             output_dir=None, seed=0, max_new_tokens=2048, video_fps=2.0,
             video_max_frames=32, max_pixels=512*512):
    """Return list[{id,text}], [] for V=0, or None on null/malformed output.

    Input is atomize() records after dependency-closed sampling. All source refs
    are current carriers, not earliest provenance. Shared Qwen is never closed.
    Operational failures raise. There are no internal retries or tool decisions.
    """
    assemble(selected_attributes, 'selection', kind)
    ids = [r['id'] for r in selected_attributes]
    if len(ids) != len(set(ids)) or len(vary_ids) != len(set(vary_ids)) or not set(vary_ids) <= set(ids):
        raise ValueError('Invalid selected/variation IDs')
    if not vary_ids:
        return []
    refs = [reference(r) for r in (references or [])]
    if {r.artifact_id for r in refs} != {a['carrier_artifact_id'] for a in selected_attributes}:
        raise ValueError('Attach exactly the selected source media')
    result = ask_json(SYSTEM, imaginer_payload(selected_attributes, vary_ids, refs, kind, seed), refs,
        qwen=qwen, output_dir=output_dir, seed=seed, max_new_tokens=max_new_tokens,
        video_fps=video_fps, video_max_frames=video_max_frames, max_pixels=max_pixels)
    if result is None:
        return None
    try:
        if not isinstance(result, dict) or set(result) != {'edits'}:
            raise ValueError('Expected edits object')
        revised = apply_variations(selected_attributes, result['edits'], vary_ids)
        assemble(revised, 'selection', kind)
        return result['edits']
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        print(f'imaginer: invalid response: {error}', file=sys.stderr)
        return None
