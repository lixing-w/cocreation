"""Plain-text transport for VLM tools; JSON is never an instruction/output contract."""
from pathlib import Path
import sys
from cocreation import Qwen, Reference
from cocreation.storage import scratch_root
from tools._attribute_protocol import description_text


def reference(value):
    ref = Reference(value['artifact_id'], value['path'], value['kind']) if isinstance(value, dict) else value
    if not isinstance(ref, Reference) or ref.kind not in ('image', 'video'):
        raise ValueError('Expected image/video Reference')
    if not Path(ref.path).expanduser().is_file():
        raise FileNotFoundError(ref.path)
    return ref


def ask_text(system, task, references, *, qwen=None, output_dir=None,
             max_new_tokens=4096, seed=None, video_fps=2.0, video_max_frames=32,
             max_pixels=512*512, cache_name=None, use_system_cache=True):
    if not isinstance(task, str) or '{{artifact:' in task:
        raise ValueError('Task must be prose without artifact placeholders')
    refs = [reference(r) for r in references]
    if len({r.artifact_id for r in refs}) != len(refs):
        raise ValueError('Duplicate reference artifact IDs')
    media = '\n'.join('{{artifact:' + r.artifact_id + '}}' for r in refs)
    prompt = 'Observation media in order:\n' + (media or 'No reference media.') + '\nTask:\n' + task
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'input.txt').write_text(prompt, encoding='utf-8')
    owned = qwen is None
    model = Qwen(output_dir=output_dir) if owned else qwen
    try:
        response = model(prompt, refs, system_prompt=system, output_dir=output_dir,
            system_cache_path=(scratch_root()/f'kvcache/{cache_name}/system_prompt.kvcache'
                               if cache_name and use_system_cache else None),
            max_new_tokens=max_new_tokens, seed=seed, video_fps=video_fps,
            video_max_frames=video_max_frames, max_pixels=max_pixels)
    finally:
        if owned:
            model.close()
    text = response.get('text') if isinstance(response, dict) else None
    if output_dir is not None and isinstance(text, str):
        (Path(output_dir) / 'response.txt').write_text(text, encoding='utf-8')
    if not isinstance(text, str) or not text.strip():
        print('VLM tool: missing response text', file=sys.stderr)
        return None
    return text.strip()


def observation_result(text):
    if text is None:
        return None
    try:
        return description_text(text)
    except ValueError as error:
        print(f'Observation rejected: {error}', file=sys.stderr)
        return None
