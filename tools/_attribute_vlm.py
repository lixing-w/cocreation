"""Small shared JSON transport for imaginer/verifier; inference errors propagate."""
import json
from pathlib import Path
import sys

from cocreation import Qwen, Reference
from tools.extract_attribute_nodes import _unique_object


def reference(value):
    ref = Reference(value['artifact_id'], value['path'], value['kind']) if isinstance(value, dict) else value
    if not isinstance(ref, Reference) or ref.kind not in ('image', 'video'):
        raise ValueError('Expected image/video Reference')
    if not Path(ref.path).expanduser().is_file():
        raise FileNotFoundError(ref.path)
    return ref


def ask_json(system, payload, references, *, qwen=None, output_dir=None,
             max_new_tokens=4096, seed=0, video_fps=2.0, video_max_frames=32,
             max_pixels=512*512):
    refs = [reference(r) for r in references]
    if len({r.artifact_id for r in refs}) != len(refs):
        raise ValueError('Duplicate reference artifact IDs')
    media = '\n'.join('{{artifact:'+r.artifact_id+'}}' for r in refs)
    # Literal placeholder syntax in captions is data, not a new media binding.
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('{{artifact:', '\\u007b\\u007bartifact:')
    prompt = 'Media in order:\n'+(media or 'None')+'\nTask data:\n'+data
    owned = qwen is None
    model = Qwen(output_dir=output_dir) if owned else qwen
    try:
        response = model(prompt, refs, system_prompt=system, output_dir=output_dir,
                         max_new_tokens=max_new_tokens, seed=seed, video_fps=video_fps,
                         video_max_frames=video_max_frames, max_pixels=max_pixels)
    finally:
        if owned:
            model.close()
    try:
        text = response.get('text') if isinstance(response, dict) else None
        if not isinstance(text, str):
            raise ValueError('Missing response text')
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        def reject_constant(value):
            raise ValueError('Non-finite JSON constant: '+value)
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError) as error:
        print(f'attribute tool: invalid JSON: {error}', file=sys.stderr)
        return None
