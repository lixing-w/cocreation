"""Materialize one dialogue run from its executable placeholders.json catalog."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from contextlib import ExitStack, closing, contextmanager, nullcontext
from copy import deepcopy
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dialogue_generation.generation import construction_order, FLUX_MODEL_ID, H3_IMAGE_MODEL_ID
from scripts.dialogue_generation.generation_retry import attempt_placeholder, clip_placeholder, retry_edit_task, retry_source_path
from scripts.dialogue_generation.media_queue import AssetEvent, AssetRejected, ReviewResult, generation_time, review_call
from tools.video_editing import EXPORT_VIDEO_CRF, prepare_inputs, execute_edit, refine_shots, probe
from cocreation.qwen_inputs import DEFAULT_VIDEO_FPS
from cocreation.video_tone import POLICY as TONE_POLICY, brightness_anchor, match_video

TOKEN = re.compile(r'\[\[(image|video)_\d{4}\]\]')
DETERMINISTIC = {'source', 'copy', 'extract_frame', 'extract_shot', 'concatenate'}
GENERATIVE = {'text_generation', 'asset_reconstruction', 'local_edit', 'reference_generation', 'video_edit'}


def execution_key(task):
    """Deterministic media work does not evict the resident generator."""
    request = task.get('clip_edits', [{}])[0].get('generation', task) if task.get('clip_edits') else task
    if request.get('operation') not in GENERATIVE:
        return None
    backend = request['backend']
    model_id = request.get('model_id') or {
        'flux': FLUX_MODEL_ID, 'h3_image': H3_IMAGE_MODEL_ID, 'minimax_h3': 'Comfy-Org/MiniMax-H3',
        'vdn_h3': 'OpenVDN/vdn-minimax-h3-edge'}[backend]
    return backend, model_id


def materialization_units(tasks, recipes=None, groups=(), *, separate_text_video=False):
    """Contract jointly reviewed deliveries into single dependency nodes."""
    baseline = construction_order(tasks)
    group_for = {token: members[0] for members in groups for token in members}
    units = {}
    for token in baseline:
        leader = group_for.get(token, token)
        unit = units.setdefault(leader, dict(members=[], dependencies=[], model=None))
        unit['members'].append(token)
        for parent in tasks[token]['dependencies']:
            parent = group_for.get(parent, parent)
            if parent != leader and parent not in unit['dependencies']:
                unit['dependencies'].append(parent)
        if unit['model'] is None and (key := execution_key(tasks[token])) is not None:
            if key[0] == 'minimax_h3' and recipes is not None:
                task = tasks[token]
                request = task['clip_edits'][0]['generation'] if task.get('clip_edits') else task
                turbo = recipes[id(request)]['turbo']
                key = (*key, turbo)
                if separate_text_video and not turbo and not request.get('references'):
                    key = (*key, 't2va')
            unit['model'] = key
    return units


def model_order(units, *, current_model=None):
    """Schedule ready nodes together by model, leaving CPU-only work neutral."""
    baseline = construction_order(units)

    def schedule(first):
        pending, done, order = dict.fromkeys(baseline), set(), []
        current = None
        while pending:
            ready = [node for node in pending if set(units[node]['dependencies']) <= done]
            selected = [node for node in ready if units[node]['model'] is None]
            if not selected:
                models = [units[node]['model'] for node in ready]
                if current not in models:
                    current = first if current is None and first in models else models[0]
                selected = [node for node in ready if units[node]['model'] == current]
            order.extend(selected)
            done.update(selected)
            for node in selected:
                del pending[node]
        return order

    rank = {node: index for index, node in enumerate(baseline)}

    def cost(order):
        models = [units[node]['model'] for node in order if units[node]['model'] is not None]
        switches = sum(a != b for a, b in zip(models, models[1:]))
        if models and current_model is not None:
            switches += models[0] != current_model
        return switches, [rank[node] for node in order]

    models = dict.fromkeys(units[node]['model'] for node in baseline if units[node]['model'] is not None)
    return min((schedule(model) for model in models), key=cost, default=baseline)


def queued_units(tasks, recipes, groups, *, separate_text_video):
    """Expand deliveries into individual ready assets while retaining their gates."""
    from scripts.dialogue_generation.reviewed_turn import delivery_order

    units = materialization_units(tasks, recipes, separate_text_video=separate_text_video)
    delivery_end = {}
    for members in groups:
        ordered = delivery_order(members, tasks)
        delivery_end.update({token: ordered[-1] for token in ordered})
        for before, after in zip(ordered, ordered[1:]):
            if before not in units[after]['dependencies']:
                units[after]['dependencies'].append(before)
    for token, unit in units.items():
        unit['dependencies'] = list(dict.fromkeys(
            delivery_end.get(parent, parent) if delivery_end.get(token) != delivery_end.get(parent) else parent
            for parent in unit['dependencies']))
    for position, token in enumerate(construction_order(tasks), 1):
        units[token]['plan_position'] = position
    return units


def materialization_order(tasks, *, recipes=None, groups=None):
    units = materialization_units(tasks, recipes, groups or ())
    return [token for leader in model_order(units) for token in units[leader]['members']]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def local_source(value, directory):
    if not isinstance(value, str) or '://' in value:
        raise ValueError('source_video must be a real local file; virtual examples require --source-video')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory/path).resolve(strict=True)


def duration(path):
    info = probe(path)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    seconds = float(video.get('duration') or info['format']['duration'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid video duration: '+str(path))
    return seconds


def validate_media(path, kind):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Missing or empty media: '+str(path))
    if kind == 'image':
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    else:
        info = probe(path)
        if not any(s['codec_type'] == 'video' for s in info['streams']):
            raise ValueError('Output has no video stream: '+str(path))
        duration(path)


def load_contract(directory, source_video=None):
    catalog = read_json(directory/'placeholders.json')
    if not isinstance(catalog, list) or not catalog:
        raise ValueError('placeholders.json must be a nonempty executable catalog')
    tasks = {}
    for original in catalog:
        row = deepcopy(original)
        token = row.get('placeholder', '')
        match = TOKEN.fullmatch(token)
        if not match or row.get('kind') != match[1] or token in tasks:
            raise ValueError('Invalid or duplicate placeholder: '+str(token))
        op = row.get('operation')
        if row.get('status', 'ready') != 'ready' or op not in DETERMINISTIC | GENERATIVE:
            raise ValueError('Finish generation prompt rewriting first: '+token)
        deps = row.get('dependencies', row.get('reference_materials', []))
        if not isinstance(deps, list) or any(not isinstance(p, str) for p in deps):
            raise ValueError('Invalid dependencies: '+token)
        row['dependencies'] = list(dict.fromkeys(deps))
        if op in GENERATIVE:
            backend = row.get('backend')
            if backend not in ('flux', 'h3_image', 'vdn_h3', 'minimax_h3') or not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('Missing model backend/prompt: '+token)
            if (backend in ('flux', 'h3_image')) != (row['kind'] == 'image'):
                raise ValueError('Backend and output kind disagree: '+token)
            counts = Counter()
            for ref in row.get('references', []):
                kind = ref.get('kind')
                if kind not in ('image', 'video', 'audio') or (backend in ('flux', 'h3_image') and kind != 'image'):
                    raise ValueError('Unsupported expert reference kind: '+token)
                counts[kind] += 1
                label_kind = 'Image' if backend == 'flux' else {'image':'Picture', 'video':'Video', 'audio':'Audio'}[kind]
                expected = f'<{label_kind} {counts[kind]}>'
                if ref.get('label', expected) != expected:
                    raise ValueError('Reference labels must preserve expert slot order: '+token)
                if ref.get('placeholder') and ref['placeholder'] not in row['dependencies']:
                    raise ValueError('Reference missing from dependencies: '+token)
        for clip in row.get('clips', []):
            if clip.get('placeholder') not in row['dependencies']:
                raise ValueError('Assembly clip missing from dependencies: '+token)
        if op in ('copy', 'extract_frame', 'extract_shot') and len(row['dependencies']) != 1:
            raise ValueError('Operation requires exactly one source: '+token)
        if op == 'concatenate' and not row.get('clips'):
            raise ValueError('Assembly requires ordered clips: '+token)
        for part in row.get('clip_edits', []):
            child=part.get('generation', {})
            if part.get('placeholder') not in row['dependencies'] or child.get('backend') not in ('vdn_h3', 'minimax_h3') or child.get('operation')!='video_edit' or not child.get('prompt') or child.get('status')!='ready':
                raise ValueError('Finish rewriting the independent clip edit: '+token)
            if child.get('dependencies') != [part['placeholder']] or len(child.get('references', []))!=1 or child['references'][0].get('placeholder')!=part['placeholder']:
                raise ValueError('An assembly edit must reference exactly one individual source clip: '+token)
        for request in [row]+[part['generation'] for part in row.get('clip_edits',[])]:
            if request.get('backend') in ('minimax_h3','vdn_h3') and len(re.findall(r'\[Shot \d+\]',request.get('prompt','')))>1:
                raise ValueError('Each generation task must produce one single-shot clip: '+token)
        tasks[token] = row
    for token, row in tasks.items():
        for ref in row.get('references', []):
            parent = ref.get('placeholder')
            if parent is not None and parent not in tasks:
                raise ValueError('Unknown reference: '+str(parent))
            if parent is not None and tasks[parent]['kind'] != ref['kind']:
                if (tasks[parent]['kind'], ref['kind']) != ('video','image') or not ('frame_seconds' in ref or 'crop_query' in ref):
                    raise ValueError('Video-to-image references require an explicit frame_seconds: '+token)
            if 'frame_seconds' in ref:
                value = ref['frame_seconds']
                if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid reference frame_seconds: '+token)
    order = materialization_order(tasks)
    roots = [p for p in order if tasks[p]['operation'] == 'source']
    observation_file = directory/'observation.json'
    observation = read_json(observation_file) if observation_file.exists() else {}
    # Archived reverse runs retain their explicit source root. New forward runs
    # have independent text-generation roots and never open the guide as input.
    root, source = None, None
    if roots:
        if len(roots) != 1 or tasks[roots[0]]['kind'] != 'video' or tasks[roots[0]]['dependencies']:
            raise ValueError('Invalid legacy source root')
        root = roots[0]
        source = local_source(str(source_video) if source_video is not None else tasks[root].get('source_video'), directory)
        tasks[root]['source_video'] = str(source)
    for token in order:
        if not tasks[token]['dependencies'] and tasks[token]['operation'] not in ('source', 'text_generation', 'reference_generation'):
            raise ValueError('Input-free roots must be generation operations: '+token)
    needs_shots = any(row.get('operation') == 'extract_shot' or any(ref.get('shot_id') for ref in row.get('references', []))
                      for row in tasks.values())
    if needs_shots and not observation.get('shots'):
        raise ValueError('Shot extraction requires observation.json in the run directory')
    markdown_file = directory/'dialogue.placeholders.md'
    if not markdown_file.exists():
        markdown_file = directory/'dialogue.md'
    original_markdown = markdown_file.read_text(encoding='utf-8')
    for match in TOKEN.finditer(original_markdown):
        if match[0] not in tasks:
            raise ValueError('Unknown placeholder in dialogue.md: '+match[0])
    return tasks, order, root, source, observation, original_markdown


def render_markdown(original, tasks, records):
    def replace(match):
        token = match[0]
        if token not in records:
            raise ValueError('Unresolved dialogue placeholder: '+token)
        url = html.escape(quote(records[token]['path'], safe='/'), quote=True)
        name = html.escape(token[2:-2], quote=True)
        if tasks[token]['kind'] == 'image':
            return f'<img src="{url}" alt="{name}" style="max-width:100%;height:auto;" loading="lazy">'
        return (f'<video src="{url}" controls preload="metadata" style="max-width:100%;height:auto;" '
                f'aria-label="{name}"><a href="{url}">{name}</a></video>')
    return TOKEN.sub(replace, original)


def cached_flux_snapshot(model_id):
    """Resolve and validate an offline Diffusers snapshot before producing any media."""
    from huggingface_hub import snapshot_download
    from cocreation import Runtime
    try:
        snapshot = Path(snapshot_download(model_id, cache_dir=Runtime().cache_dir,
                                          local_files_only=True))
    except Exception as exc:
        raise RuntimeError(f'FLUX checkpoint is not cached: {model_id}. '
                           'Prepare this exact checkpoint or change the run contract model_id '
                           'to an installed model before materializing.') from exc
    required = ['model_index.json']
    for component in ('scheduler', 'tokenizer', 'text_encoder', 'transformer', 'vae'):
        folder = snapshot/component
        if not folder.is_dir():
            raise RuntimeError(f'Incomplete FLUX cache: {folder}')
        if component in ('text_encoder', 'transformer', 'vae'):
            indexes = list(folder.glob('*.safetensors.index.json'))
            if indexes:
                required += [str((folder/name).relative_to(snapshot)) for index in indexes
                             for name in set(read_json(index)['weight_map'].values())]
            else:
                weights = list(folder.glob('*.safetensors'))
                if not weights:
                    raise RuntimeError(f'Incomplete FLUX cache: no weights in {folder}')
                required += [str(p.relative_to(snapshot)) for p in weights]
    for name in required:
        path = snapshot/name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Incomplete FLUX cache: {path}')
    return snapshot


def preflight_models(tasks, *, video_turbo_lora=False):
    """Check offline model installations without starting workers or allocating VRAM."""
    report = {}
    requests=[request for task in tasks.values() for request in [task]+[part['generation'] for part in task.get('clip_edits',[])]]
    for task in requests:
        if task['operation'] not in GENERATIVE:
            continue
        backend = task['backend']
        model_id = task.get('model_id') or {'flux': FLUX_MODEL_ID, 'h3_image': H3_IMAGE_MODEL_ID, 'vdn_h3': 'OpenVDN/vdn-minimax-h3-edge', 'minimax_h3': 'Comfy-Org/MiniMax-H3'}[backend]
        key = backend+':'+model_id
        if key in report:
            continue
        if backend == 'flux':
            snapshot = cached_flux_snapshot(model_id)
            report[key] = dict(snapshot=str(snapshot), revision=snapshot.name)
        elif backend == 'h3_image':
            from cocreation.h3_image import installation
            report[key] = installation()
        elif backend == 'minimax_h3':
            from dataclasses import asdict
            from cocreation import H3Runtime
            from cocreation.comfy_worker import prepare_runtime, prepare_turbo, download_models
            from cocreation.h3 import MODEL_REVISION, TURBO_REVISION, TURBO_LORA
            rt = asdict(H3Runtime(auto_setup=False, local_files_only=True, turbo=video_turbo_lora))
            root = Path(rt['root'])
            _, python = prepare_runtime(rt)
            files = download_models(python, root, 'ref2va', True)
            report[key] = dict(root=str(root), revision=MODEL_REVISION, weights=files,
                               lora=None)
            if video_turbo_lora:
                prepare_turbo(python, root, rt)
                report[key].update(lora=TURBO_LORA, lora_revision=TURBO_REVISION)
        else:
            from cocreation import VDNRuntime
            from cocreation.vdn_assets import verify_fp8
            root = Path(VDNRuntime().root)
            for filename in ('runtime-ready.json', 'assets-ready.json'):
                if not (root/filename).is_file():
                    raise RuntimeError('VDN is not prepared: '+str(root/filename))
            installation = read_json(root/'assets-ready.json')
            for field in ('python', 'cache', 'base', 'checkpoint', 'comfy_root', 'model_paths'):
                if not Path(installation[field]).exists():
                    raise RuntimeError('Incomplete VDN installation: '+installation[field])
            verify_fp8(installation['cache'])
            report[key] = dict(root=str(root), revision=installation['model_revision'])
    return report


def default_model(backend, model_id, output_dir, *, video_turbo_lora=False):
    from cocreation import Flux, VDNH3, Runtime, VDNRuntime, MiniMaxH3, H3Runtime
    if backend == 'flux':
        model_id = model_id or FLUX_MODEL_ID
        snapshot = cached_flux_snapshot(model_id)
        return Flux(model_id=model_id,
                    runtime=Runtime(local_files_only=True, revision=snapshot.name), output_dir=output_dir)
    if backend == 'h3_image':
        from cocreation import MiniMaxH3Image
        if model_id not in (None, H3_IMAGE_MODEL_ID):
            raise ValueError('Unsupported H3 Image checkpoint: '+model_id)
        return MiniMaxH3Image(output_dir=output_dir)
    if backend == 'minimax_h3':
        return MiniMaxH3(runtime=H3Runtime(auto_setup=False, local_files_only=True,
                                         turbo=video_turbo_lora), output_dir=output_dir)
    return VDNH3(runtime=VDNRuntime(auto_setup=False, local_files_only=True), output_dir=output_dir)


def pad_short_references(references, directory, *, minimum=1.):
    """Hold short clips without dropping frames: Base H3 requires >=2s, VDN >=1s."""
    from dataclasses import replace
    result = []
    for index, ref in enumerate(references):
        if ref.kind == 'video' and duration(ref.path) < minimum:
            path = directory/f'padded-reference-{index}.mp4'
            info = probe(ref.path)
            audio = any(s['codec_type']=='audio' for s in info['streams'])
            command = ['ffmpeg','-nostdin','-v','error','-i',str(ref.path),
                       '-vf',f'tpad=stop_mode=clone:stop_duration={minimum}','-t',str(minimum),
                       '-c:v','libx264','-crf','18','-pix_fmt','yuv420p']
            if audio:
                command += ['-af','apad','-c:a','aac']
            subprocess.run(command+[str(path)],check=True)
            ref = replace(ref,path=path)
        result.append(ref)
    return result


def finish_video(source, destination, *, seconds, size, audio_source=None):
    """Normalize generated clips for exact-order assembly; retain edit-source audio."""
    width, height = size
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source)]
    if audio_source is not None:
        command += ['-i', str(audio_source)]
    command += ['-map', '0:v:0', '-map', '1:a:0?' if audio_source is not None else '0:a:0?',
                '-t', str(seconds), '-vf',
                f'scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,'
                f'crop={width}:{height},setsar=1',
                '-c:v', 'libx264', '-crf', EXPORT_VIDEO_CRF, '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)


def materialize(run_dir, **options):
    """Drain one materialization session in model-aware order."""
    with closing(materialization_session(run_dir, **options)) as session:
        try:
            units = next(session)
            discarded = set()
            for leader in model_order(units):
                if discarded.isdisjoint(units[leader]['members']):
                    event = session.send(leader)
                    if event is not None:
                        discarded.update(event.discarded)
        except StopIteration as finished:
            return finished.value


def materialization_session(run_dir, *, resume=False, source_video=None, seed=0,
                image_width=None, image_height=None, video_width=None, video_height=None,
                video_steps=None, video_turbo_lora=False, video_two_pass=False, output_width=None, output_height=None, native_video_output=False, dry_run=False, model_factory=None, progress=True, match_video_brightness=True, new_video_turbo_lora=False,
                new_video_steps=None, edit_video_steps=None, edit_video_sampler='euler',
                verify_media=False, reviewer_factory=None, review_base_url='http://127.0.0.1:8019', review_timeout=600, shared_vlm=False, shared_generator=None, timings=None,
                review_model=None, defer_retries=False, async_reviews=False):
    if async_reviews and not (shared_vlm and defer_retries):
        raise ValueError('Async media review requires a shared VLM and deferred retries')

    @contextmanager
    def measure(stage):
        start = time.monotonic()
        try:
            yield
        finally:
            if timings is not None:
                timings[stage] += time.monotonic() - start

    directory = Path(run_dir).expanduser().resolve(strict=True)
    tasks, order, root, source, observation, original = load_contract(directory, source_video)
    dialogue_file=directory/'dialogue.json'
    dialogue=read_json(dialogue_file) if dialogue_file.exists() else {}
    groups=[list(turn['output_media']) for turn in dialogue.get('turns',[])
            if turn['role']=='assistant' and len(turn.get('output_media',[]))>1
            and not any(tasks[token].get('clip_edits') for token in turn['output_media'])
            and any(tasks[token].get('backend') for token in turn['output_media'])] if verify_media else []
    group_for={token:members for members in groups for token in members}
    if type(match_video_brightness) is not bool:
        raise ValueError('match_video_brightness must be a boolean')
    if type(shared_vlm) is not bool:
        raise ValueError('shared_vlm must be a boolean')
    # Validate even in dry runs; never guess an edit source from multiple videos.
    tone_anchors = {}
    for token, task in tasks.items():
        requests = [task] + [part['generation'] for part in task.get('clip_edits', [])]
        tone_anchors[token] = [brightness_anchor(request, tasks) for request in requests]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if (image_width is None) != (image_height is None):
        raise ValueError('Supply both image dimensions')
    explicit_image_size = image_width is not None
    image_width, image_height = (image_width, image_height) if explicit_image_size else (1024, 1024)
    if any(type(v) is not int or v < 64 or v % 16 for v in (image_width, image_height)):
        raise ValueError('FLUX image dimensions must be multiples of 16 and >=64')
    if (video_width is None) != (video_height is None):
        raise ValueError('Supply both --video-width and --video-height')
    for binary in ('ffmpeg', 'ffprobe'):
        if shutil.which(binary) is None:
            raise RuntimeError('Install ffmpeg/ffprobe or add the cocreation conda bin directory to PATH')
    source_hash = None
    if source is not None:
        validate_media(source, 'video')
        source_hash = file_hash(source)
        apply_frame_selections(directory, tasks, root, source_hash, observation)
        info = next(s for s in probe(source)['streams'] if s['codec_type'] == 'video')
        output_size = (info['width'], info['height'])
    else:
        # The original video is a guide; it does not dictate generated geometry.
        output_size = (1280, 720)
    if type(native_video_output) is not bool:
        raise ValueError('native_video_output must be a boolean')
    if native_video_output and (output_width is not None or output_height is not None):
        raise ValueError('Native video output cannot be combined with output size overrides')
    if (output_width is None) != (output_height is None):
        raise ValueError('Supply both output dimensions')
    if output_width is not None:
        output_size = (output_width, output_height)
    if any(type(v) is not int or v < 32 for v in output_size):
        raise ValueError('Output dimensions must be positive even integers >=32')
    if any(v % 2 for v in output_size):
        raise ValueError('Target video dimensions must be even for H.264 assembly')
    if video_width is None:
        # Render at the output's native scale, rounding up only for latent geometry.
        grid = 64 if video_two_pass else 32
        video_width, video_height = (max(512, math.ceil(v/grid)*grid) for v in output_size)
    if type(video_turbo_lora) is not bool or type(video_two_pass) is not bool:
        raise ValueError('Video mode switches must be booleans')
    requests = [r for task in tasks.values() for r in [task]+[p['generation'] for p in task.get('clip_edits', [])]]
    video_backends = {r['backend'] for r in requests if r['operation'] in GENERATIVE
                      and r['backend'] in ('minimax_h3', 'vdn_h3')}
    from cocreation.h3 import default_steps
    from cocreation.h3_policy import inference_recipe
    recipe_options = dict(steps=video_steps, turbo=video_turbo_lora, new_turbo=new_video_turbo_lora,
        new_steps=new_video_steps, edit_steps=edit_video_steps, edit_sampler=edit_video_sampler)
    def recipe_for(operation, *, has_references=None):
        if shared_generator is not None:
            if has_references is not None and hasattr(shared_generator, 'video_recipe'):
                return shared_generator.video_recipe(operation, has_references=has_references)
            return shared_generator.recipe(operation)
        return inference_recipe(operation, **recipe_options)

    if shared_generator is not None:
        unsupported = {r['backend'] for r in requests if r['operation'] in GENERATIVE
                       and r['backend'] not in ('minimax_h3', 'h3_image')}
        if unsupported:
            raise ValueError('SGLang batch requires H3 generation backends: ' + ', '.join(sorted(unsupported)))
    # Validate CLI recipes before allocating any generation workers.
    for operation in ('reference_generation', 'video_edit'):
        recipe_for(operation)
    h3_requests = [r for r in requests if r.get('backend') == 'minimax_h3']
    recipes = {id(r): recipe_for(r['operation'], has_references=bool(r.get('references')))
               for r in h3_requests}
    units = (queued_units(tasks, recipes, groups, separate_text_video=shared_generator is not None)
             if defer_retries else
             materialization_units(tasks, recipes, groups, separate_text_video=shared_generator is not None))
    order = [token for leader in model_order(units) for token in units[leader]['members']]
    steps_by_backend = {backend: video_steps if video_steps is not None else
                        default_steps(video_turbo_lora) if backend == 'minimax_h3' else 7
                        for backend in video_backends}
    if video_steps is not None and (type(video_steps) is not int or not 1 <= video_steps <= 10000):
        raise ValueError('Video steps must be an integer in 1–10000')
    if video_two_pass and 'minimax_h3' in video_backends:
        raise ValueError('MiniMax-H3 uses a single pass; remove --video-two-pass')
    if ('vdn_h3' in video_backends and steps_by_backend['vdn_h3'] > 32) or (video_two_pass and video_steps != 8):
        raise ValueError('VDN steps must be 1–32; --video-two-pass requires --video-steps 8')
    if any(type(v) is not int or v < (512 if video_two_pass else 256) or v % 32 for v in (video_width, video_height)):
        raise ValueError('VDN dimensions must be multiples of 32 (>=256 single-pass, >=512 two-pass)')
    if native_video_output:
        output_size = (video_width, video_height)
    config = dict(seed=seed, native_video_output=native_video_output, image_size=[image_width,image_height] if explicit_image_size else 'backend_recommended',
                  video_size=[video_width,video_height], video_steps=video_steps,
                  video_steps_by_backend=steps_by_backend, video_turbo_lora=video_turbo_lora,
                  video_two_pass=video_two_pass, output_size=list(output_size))
    if shared_generator is not None or new_video_turbo_lora or new_video_steps is not None or edit_video_steps is not None or edit_video_sampler != 'euler':
        new_video_recipe = next((recipes[id(r)] for r in h3_requests if r['operation'] != 'video_edit'),
                                recipe_for('reference_generation'))
        video_edit_recipe = next((recipes[id(r)] for r in h3_requests if r['operation'] == 'video_edit'),
                                 recipe_for('video_edit'))
        config['h3_inference_policy'] = dict(new_video=new_video_recipe, video_edit=video_edit_recipe)
    if shared_generator is not None:
        config['media_serving'] = 'sglang'

    from scripts.dialogue_generation.output_geometry import geometry
    h3_images = {token for token, task in tasks.items() if task.get('backend') == 'h3_image'}
    for token in order:
        task = tasks[token]
        if task['kind'] == 'image' and task['operation'] == 'copy' and task['dependencies'][0] in h3_images:
            h3_images.add(token)
        native, final = geometry(task.get('render_state', {}), task['kind'],
            image=(image_width, image_height), video=(video_width, video_height), output=output_size, native_video_output=native_video_output)
        if token in h3_images:
            from cocreation.h3_image import size_for
            from scripts.dialogue_generation.cross_turn_rule import output_aspect
            native = final = ((image_width, image_height) if explicit_image_size else
                size_for(output_aspect(task.get('render_state', {}).get('aspect_ratio') or 'square_assets_landscape_video', 'image')))
            if any(v % 32 for v in native) or not 2048**2 <= native[0]*native[1] <= 20*1024**2:
                raise ValueError('H3 Image canvas must be 4–20 MP with sides divisible by 32')
        task['native_size'], task['output_size'] = list(native), list(final)
        for part in task.get('clip_edits', []):
            child = part['generation']
            child_native, child_final = geometry(child.get('render_state', {}), 'video',
                video=(video_width, video_height), output=output_size, native_video_output=native_video_output)
            child['native_size'], child['output_size'] = list(child_native), list(child_final)
    summary = dict(run_dir=str(directory), construction_order=order,
                   operations=dict(Counter(row['operation'] for row in tasks.values())), settings=config,
                   output_sizes={token: task['output_size'] for token, task in tasks.items()})
    summary['video_brightness_matching'] = dict(enabled=match_video_brightness, policy=TONE_POLICY,
        anchors={token: refs for token, refs in tone_anchors.items() if any(refs)})
    summary['h3_request_recipes'] = {token: [recipes[id(r)] for r in [task]+[p['generation'] for p in task.get('clip_edits', [])] if id(r) in recipes]
                                     for token, task in tasks.items() if any(id(r) in recipes for r in [task]+[p['generation'] for p in task.get('clip_edits', [])])}
    if verify_media:
        config['media_review'] = dict(enabled=True,max_generations=3,thinking=False,analysis_field='think',base_url=review_base_url,shared_vlm=shared_vlm)
        if review_model is not None:
            config['media_review'].update(model=review_model.model_id, effort=review_model.effort)
    if dry_run:
        return dict(status='planned', **summary)
    if model_factory is None and shared_generator is None:
        summary['models'] = preflight_models(tasks, video_turbo_lora=any(r['turbo'] for r in recipes.values()))
    work_root = directory/'_work/materialize'
    work_root.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.materialize.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another materializer is using this run directory') from None
    factory = model_factory
    model, model_key = None, None
    manifest_path = directory/'media_manifest.json'
    manifest = {}
    manifest_active = False
    token = None
    started = time.monotonic()
    succeeded = False
    try:
        if manifest_path.exists():
            if not resume:
                raise FileExistsError('media_manifest.json exists; use --resume')
            manifest = read_json(manifest_path)
            if manifest.get('version') != 1:
                raise ValueError('Unsupported media manifest version')
        backup = directory/'dialogue.placeholders.md'
        if not backup.exists():
            atomic_text(backup,original)
        records = manifest.get('media', {})
        unknown = set(records)-set(tasks)
        if unknown:
            raise ValueError('Existing manifest belongs to a different catalog')
        manifest.update(version=1, status='running', **summary, media=records)
        manifest.pop('error', None)
        save(manifest_path, manifest)
        manifest_active = True
        media_dir = directory/'media'
        media_dir.mkdir(exist_ok=True)
        resolved = {}
        mapping = None
        dialogue_file = directory/'dialogue.json'
        dialogue = read_json(dialogue_file) if dialogue_file.exists() else {}
        if verify_media and not dialogue.get('turns'):
            raise ValueError('Media review requires a realized dialogue')
        from scripts.dialogue_generation.formal_sequence import (
            dialogue_without_failed_media, export_sequence, generation_context, review_context,
        )
        active_dialogue = dialogue
        pending = set(units)
        cursors = {}
        exhausted, omitted, removed_turns = {}, set(), set()

        def export(status='running'):
            if dialogue.get('turns'):
                export_sequence(directory, active_dialogue,
                                {key: row for key, row in records.items() if key not in omitted},
                                status=status)

        def discard_rejected(error):
            nonlocal active_dialogue, omitted, removed_turns
            event = error.event
            exhausted[event.token] = str(error)
            active_dialogue, discarded, removed_turns = dialogue_without_failed_media(dialogue, tasks, exhausted)
            for member in discarded - omitted:
                row = records.setdefault(member, dict(kind=tasks[member]['kind'], operation=tasks[member]['operation']))
                if member in exhausted:
                    row.update(status='failed', review_status='rejected', error=exhausted[member])
                else:
                    row['status'] = 'omitted'
                resolved.pop(member, None)
                cursor = cursors.pop(group_for.get(member, [member])[0], None)
                if cursor is not None:
                    cursor.close()
            pending.difference_update(key for key, unit in units.items() if discarded.intersection(unit['members']))
            omitted = discarded
            manifest['pruning'] = dict(exhausted=exhausted, omitted_media=sorted(omitted - exhausted.keys()),
                                       removed_turn_ids=sorted(removed_turns))
            save(manifest_path, manifest)
            export()
            return replace(event, status='exhausted', discarded=tuple(sorted(omitted)))
        def generate_media(task, job, staged, task_seed, record):
            nonlocal model, model_key
            # A video-to-image dependency uses the shared VLM when configured;
            # standalone localization releases the generation worker first.
            def crop_reference(path, query, output):
                nonlocal model, model_key
                cached, valid = generated_crop_cache(directory,path,query)
                if valid:
                    return str(cached)
                if not shared_vlm and model is not None:
                    model.close()
                    model, model_key = None, None
                cached.unlink(missing_ok=True)
                crop_options=dict(qwen=review_model) if review_model is not None else (
                    dict(base_url=review_base_url,require_running=True) if shared_vlm else {})
                with measure('review'):
                    cropped = crop_generated_reference(path, query, cached,**crop_options)
                save(cached.with_suffix('.json'),dict(sha256=file_hash(cropped)))
                return cropped
            is_retry = task.get('retry_edit', False)
            if is_retry:
                from cocreation import Reference
                references = [Reference(f'reference_{index+1}', directory / slot['path'], slot['kind'])
                              for index, slot in enumerate(task['references'])]
                retry_source_path(directory, task)
                prompt = task['prompt']
            else:
                references, prompt = prepare_inputs(task, resolved, observation, job/'inputs',
                    shot_map=mapping, crop_resolver=crop_reference)
            if not is_retry and task['backend'] in ('vdn_h3', 'minimax_h3'):
                references = pad_short_references(references, job/'inputs', minimum=2. if task['backend']=='minimax_h3' else 1.)
                from dataclasses import replace
                short_refs=[]
                for ref_index, ref in enumerate(references):
                    if task['backend']=='vdn_h3' and ref.kind=='video' and duration(ref.path)>4.001:
                        trimmed=job/'inputs'/f'four-second-reference-{ref_index}.mp4'
                        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(ref.path),'-t','4','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac',str(trimmed)],check=True)
                        record.setdefault('reference_trims',[]).append(dict(reference=ref_index,start_seconds=0,seconds=4,source_seconds=duration(ref.path)))
                        ref=replace(ref,path=trimmed)
                    short_refs.append(ref)
                references=short_refs
            record['actual_prompt'] = prompt
            record['kind'] = task['kind']
            record['backend'] = task['backend']
            record['executed_operation'] = task['operation']
            record['references'] = [
                dict(kind=ref.kind, path=str(Path(ref.path).relative_to(directory)),
                     **{key:slot[key] for key in ('label','placeholder','role','subjects','attempt',
                                                  'frame_seconds','shot_id','crop_query') if key in slot})
                for ref, slot in zip(references,task.get('references', []))]
            for ref in record['references']:
                if ref.get('placeholder') in task.get('reference_versions', {}):
                    ref['placeholder'] = task['reference_versions'][ref['placeholder']]
            key = execution_key(task)
            recipe = (recipe_for(task['operation'], has_references=bool(references))
                      if key[0] == 'minimax_h3' else None)
            resident_key = (*key, recipe['turbo']) if recipe is not None else key
            if recipe is not None:
                record['inference_recipe'] = recipe
            if shared_generator is None and model_key != resident_key:
                if model is not None:
                    model.close()
                    model = None
                # Runtime configuration is immutable in ComfySession. Close
                # the previous worker before changing LoRA state; never fuse
                # the adapter permanently into the base checkpoint.
                model = (factory(*key, job/'runtime') if factory is not None else
                         default_model(*key, job/'runtime', video_turbo_lora=bool(recipe and recipe['turbo'])))
                # NFS keeps deleted-but-open logs as .nfs files. Resident logs
                # therefore live outside the disposable per-asset request tree.
                model.session_dir = work_root/'_sessions'/'__'.join(job.relative_to(work_root).parts)
                model_key = resident_key
            options = dict(seed=task_seed)
            audio_source = None
            if key[0] in ('flux', 'h3_image'):
                options.update(width=task['native_size'][0], height=task['native_size'][1])
                if key[0] == 'h3_image':
                    options.update(steps=6,guidance_scale=1.0)
            else:
                output_seconds = min(4., float(task.get('output_seconds', task.get('seconds', 4))))
                if task.get('shot_id') and mapping is not None:
                    shot = mapping['shots'][task['shot_id']]
                    output_seconds = min(4., shot['end_seconds']-shot['start_seconds'])
                if task.get('preprocess') == 'concatenate':
                    raise ValueError('Whole-cut video inference is disabled; rewrite the contract into per-clip edits')
                input_seconds = [duration(r.path) for r in references if r.kind == 'video']
                is_h3 = key[0] == 'minimax_h3'
                seconds = task.get('retry_seconds') if is_retry else None
                if seconds is None:
                    seconds = max(4. if is_h3 else 1.625, output_seconds, *input_seconds)
                if not math.isfinite(seconds) or seconds > (15.001 if is_h3 else 4.001) or output_seconds <= 0:
                    raise ValueError('Video duration exceeds the selected backend limit: '+token)
                options.update(seconds=seconds, width=task['native_size'][0], height=task['native_size'][1],
                               steps=recipe['steps'] if is_h3 else steps_by_backend[key[0]])
                if is_h3 and shared_generator is None and recipe['sampler'] != 'turbo_custom':
                    options['sampler'] = recipe['sampler']
                if not is_h3:
                    options['two_pass'] = video_two_pass
                if task['operation'] == 'video_edit':
                    audio_source = references[-1].path if is_retry else next(
                        (r.path for r in references if r.kind == 'video'), None)
            if hasattr(model, 'output_dir'):
                model.output_dir = job/'runtime'
            call_started = time.monotonic()
            try:
                if shared_generator is None:
                    result = model(prompt, references, **options)
                else:
                    category = ('image' if task['kind'] == 'image'
                                else 'video_edit' if task['operation'] == 'video_edit' else 'video_generation')
                    with measure(category):
                        result = shared_generator.generate(key[0], prompt, references, recipe, job/'runtime', options)
            finally:
                record['generation_seconds'] = round(time.monotonic()-call_started, 6)
            if result.get('status') != 'ok' or not result.get('path'):
                raise RuntimeError('Generator did not return a successful media path: '+token)
            generated = Path(result['path'])
            validate_media(generated, task['kind'])
            if key[0] in ('flux', 'h3_image'):
                from PIL import Image, ImageOps
                with Image.open(generated) as image:
                    if key[0] == 'h3_image' and image.size != tuple(task['native_size']):
                        raise RuntimeError('H3 Image returned unexpected native image dimensions: '+token)
                    if key[0] == 'flux' and task.get('render_state', {}).get('aspect_ratio'):
                        image = ImageOps.fit(image, tuple(task['output_size']))
                    image.save(staged, format='PNG')
            elif native_video_output:
                stream = next(s for s in probe(generated)['streams'] if s['codec_type'] == 'video')
                if [stream['width'], stream['height']] != task['native_size']:
                    raise RuntimeError('Generator returned unexpected native video dimensions: '+token)
                # Preserve native geometry, audio and frame-grid duration.
                # Video edits can subsequently receive a luminance-only match.
                shutil.copy2(generated, staged)
                record['video_output_mode'] = 'native'
                record['output_seconds'] = duration(generated)
            else:
                finish_video(generated, staged, seconds=output_seconds,
                             size=task['output_size'], audio_source=audio_source)
                if abs(duration(staged)-output_seconds) > max(.15, output_seconds*.02):
                    raise RuntimeError('Generated video is shorter than its required output duration: '+token)
            if task['operation'] == 'video_edit':
                if is_retry:
                    source = task['references'][-1]
                    anchor = {key: source[key] for key in ('kind', 'placeholder', 'attempt', 'role')}
                    anchor_path, anchor_hash = references[-1].path, source['sha256']
                else:
                    anchor = brightness_anchor(task, tasks)
                    anchor_path, anchor_hash = None, records[anchor['placeholder']]['sha256']
                tone_record = dict(enabled=match_video_brightness, policy=TONE_POLICY, anchor=anchor,
                    stage='post_generation_before_publish', anchor_sha256=anchor_hash)
                if match_video_brightness:
                    if anchor_path is None:
                        anchor_refs, _ = prepare_inputs(dict(references=[anchor], prompt=''), resolved,
                            observation, job/'tone-anchor', shot_map=mapping)
                        anchor_path = anchor_refs[0].path
                    matched = job/'tone-matched.mp4'
                    tone_record.update(match_video(staged, anchor_path, matched))
                    if tone_record['status'] == 'matched':
                        matched.replace(staged)
                else:
                    tone_record.update(status='disabled')
                record['brightness_matching'] = tone_record
            record['model_id'] = result.get('model_id') or task.get('model_id') or getattr(model, 'model_id', None)
            receipt = {k:v for k,v in result.items() if k not in ('path','job_dir','request_path','resident')}
            report_path = generated.with_suffix('.request.json')
            if report_path.is_file():
                receipt['upstream_report'] = read_json(report_path)
            record_dir = directory/'generation_records'
            record_dir.mkdir(exist_ok=True)
            receipt_path = record_dir/('__'.join(job.relative_to(work_root).parts)+'.json')
            save(receipt_path, dict(prompt=prompt, operation=task['operation'], references=record['references'],
                                    options=options, result=receipt, generation_seconds=record['generation_seconds']))
            record['generation_record'] = str(receipt_path.relative_to(directory))
            record['runtime_cleaned'] = False
            record['options'] = options
            # Record effective graph settings as well as explicit API arguments.
            # The public generation tool call displays a subset of these execution settings.
            record['effective_parameters'] = dict(options)
            if shared_generator is not None:
                record['effective_parameters'].update(result['effective_parameters'])
                record['model_reload_seconds'] = result.get('model_reload_seconds', 0.0)
            elif key[0] == 'h3_image':
                record['effective_parameters'].update(sampler='euler',scheduler='simple',denoise=1.0)
            elif key[0] == 'minimax_h3':
                from cocreation.h3 import frame_count
                record['effective_parameters'].update(recipe,cfg=1.0,denoise=1.0,fps=24.0,
                                                     frames=frame_count(options['seconds']))

        def generate_reviewed(task, job, staged, task_seed, record, *, attempt_token=None):
            if not verify_media:
                return generate_media(task,job,staged,task_seed,record)
            attempt_token = attempt_token or token
            context = task.get('sequence_context') or generation_context(active_dialogue, token)
            if omitted:
                context = {**context, **generation_context(active_dialogue, token, operation=task['operation'])}
            if 'latest_rules' in task:
                context=dict(context,latest_rules=task['latest_rules'])
            context = review_context(active_dialogue, token, context=context)
            current = dict(task,sequence_context=context)
            record.setdefault('attempts', [])
            archive = directory/'generation_attempts'/token.strip('[]')
            if job.name.startswith('clip-'):
                archive = archive/job.name
            if archive.exists() and not record['attempts']:
                import uuid
                archive = archive/'sessions'/uuid.uuid4().hex
            archive.mkdir(parents=True,exist_ok=True)
            for number in range(1,4):
                attempt_dir = archive/f'attempt-{number:02d}'
                attempt_dir.mkdir(exist_ok=True)
                generated_now = len(record['attempts']) < number
                if not generated_now:
                    one = record['attempts'][number-1]
                    saved_media = directory/one['path']
                    attempt_dir = saved_media.parent
                    if not saved_media.is_file() or file_hash(saved_media)!=one['sha256']:
                        raise RuntimeError('Archived generation attempt is missing or corrupt; refusing to overwrite it')
                    shutil.copy2(saved_media,staged)
                else:
                    one = {}
                    attempt_job = job/f'attempt-{number}'
                    attempt_job.mkdir()
                    generate_media(current,attempt_job,staged,(task_seed+number-1)%2**63,one)
                    saved_media = attempt_dir/('output.png' if task['kind']=='image' else 'output.mp4')
                    shutil.copy2(staged,saved_media)
                    for ref_index, ref in enumerate(one.get('references',[])):
                        source_ref = directory/ref['path']
                        archived_ref = attempt_dir/'inputs'/(str(ref_index)+source_ref.suffix)
                        archived_ref.parent.mkdir(exist_ok=True)
                        shutil.copy2(source_ref,archived_ref)
                        ref['path'] = str(archived_ref.relative_to(directory))
                    one.update(attempt=number,placeholder=attempt_placeholder(attempt_token, number),
                               kind=task['kind'],prompt=one['actual_prompt'],
                               path=str(saved_media.relative_to(directory)),sha256=file_hash(saved_media))
                    if number==1 and current.get('reasoner'):
                        if current['reasoner']['generation_prompt']!=one['prompt']:
                            raise RuntimeError('First prompt changed after reasoner; refusing mismatched formal trace')
                        one['reasoner']=current['reasoner']
                    record['attempts'].append(one)
                    save(attempt_dir/'attempt.json',one)
                    save(manifest_path,manifest)
                    export()
                from tools.media_review import cached_review
                previous_review = cached_review(one)
                rejected = AssetEvent(token, one.get('executed_operation', task['operation']), one['path'],
                                      generation_time(one) if generated_now else 0.0, 'rejected')
                if previous_review and previous_review['decision']=='rejected':
                    if number==3:
                        record['review_status']='rejected'
                        raise AssetRejected(rejected, 'Media rejected by third critique: '+token)
                    if len(record['attempts']) > number:
                        continue
                    if one.get('retry_edit_prompt'):
                        edit = retry_edit_task(task, attempt_token, record['attempts'][0], one)
                        current = dict(edit, prompt=one['retry_edit_prompt'])
                        continue
                if previous_review and previous_review['decision']=='accepted':
                    record.update({k:v for k,v in one.items() if k not in ('path','sha256','attempt','placeholder','critique','next_prompt','retry_edit_prompt')})
                    record['accepted_placeholder'] = attempt_placeholder(attempt_token, number)
                    record['review_status']='accepted'
                    record['generation_seconds']=sum(a['generation_seconds'] for a in record['attempts'])
                    return
                from cocreation import Reference
                refs = [Reference('source_'+str(i),directory/r['path'],r['kind']) for i,r in enumerate(one.get('references',[]))]
                with ExitStack() as review_scope:
                    reviewer = None

                    def invoke(method, session_dir, *args, **kwargs):
                        nonlocal reviewer
                        if async_reviews:
                            return (yield from review_call(turn_session, token, session_dir, method,
                                                          *args, deferred=True, **kwargs))
                        if reviewer is None:
                            reviewer = review_scope.enter_context(turn_session(session_dir))
                        return getattr(reviewer, method)(*args, **kwargs)

                    critique_dir = attempt_dir/'critique'
                    report = previous_review or (yield from invoke('verify', critique_dir,
                        dict(context, reference_slots=one.get('references', []),
                             reviewed_placeholder=attempt_placeholder(attempt_token, number)), one['prompt'], saved_media,
                        task['kind'], refs, retry_available=number<3,
                        seed=(task_seed+number*100)%2**63, output_dir=critique_dir))
                    one['critique'] = report
                    if report['decision']=='accepted' and len(record['attempts'])>number:
                        # Later retries followed a superseded rejection. Keep
                        # their files and audit, outside the accepted sequence.
                        record.setdefault('superseded_attempts',[]).extend(record['attempts'][number:])
                        del record['attempts'][number:]
                    save(attempt_dir/'attempt.json',one)
                    save(manifest_path,manifest)
                    export()
                    if report['decision']=='accepted':
                        record.update({k:v for k,v in one.items() if k not in ('path','sha256','attempt','placeholder','critique')})
                        record['accepted_placeholder'] = attempt_placeholder(attempt_token, number)
                        record['review_status']='accepted'
                        record['generation_seconds']=sum(a['generation_seconds'] for a in record['attempts'])
                        return
                    if number==3:
                        record['review_status']='rejected'
                        raise AssetRejected(rejected, 'Media rejected by third critique: '+token)
                    if defer_retries and previous_review is None:
                        review_scope.close()
                        reviewer = None
                        yield rejected
                    fixer_dir = attempt_dir/'fixer'
                    edit = retry_edit_task(task, attempt_token, record['attempts'][0], one)
                    edit_prompt = yield from invoke('fix', fixer_dir, edit, context, one['prompt'], report['reason'],
                        media=retry_source_path(directory, edit), seed=(task_seed+number*1000)%2**63,
                        output_dir=fixer_dir)
                    one['retry_edit_prompt'] = edit_prompt
                    save(attempt_dir/'attempt.json', one)
                    save(manifest_path, manifest)
                    current = dict(edit, prompt=edit_prompt)


        def request_fingerprint(member):
            task=tasks[member]
            fingerprint_data = dict(task=task, settings={k:v for k,v in config.items() if k != 'h3_inference_policy'}, source_hash=source_hash,
                observation=observation, dependencies={p:records[p]['sha256'] for p in task['dependencies']})
            # Only edits gain a new fingerprint field. Existing generation roots
            # remain reusable when matching is enabled on resume.
            if match_video_brightness and any(tone_anchors[member]):
                fingerprint_data['brightness_matching'] = dict(policy=TONE_POLICY,
                    anchors=[dict(reference=ref, sha256=records[ref['placeholder']]['sha256'])
                             for ref in tone_anchors[member] if ref])
            if 'h3_inference_policy' in config:
                local_requests = [task]+[p['generation'] for p in task.get('clip_edits', [])]
                local_recipes = [recipes[id(r)] for r in local_requests if id(r) in recipes]
                if local_recipes:
                    fingerprint_data['h3_inference_recipes'] = local_recipes
            return digest(fingerprint_data)

        def generate_for_turn(member,task,job,staged,task_seed,record):
            nonlocal token
            token=member
            return generate_media(task,job,staged,task_seed,record)

        def execute_for_turn(member,task,job,staged,record):
            nonlocal token,mapping
            token=member
            if task.get('clip_edits'):
                raise ValueError('Independent clip edits must be explicit generation tasks before turn review')
            if task['operation']=='source':shutil.copy2(source,staged)
            else:
                if task['operation']=='extract_shot' and mapping is None:
                    mapping=refine_shots(resolved[root],observation,output_path=work_root/'shot_map.json')
                execute_edit(task,resolved,observation,staged,shot_map=mapping)
            if task['operation'] in ('concatenate','copy'):
                from tools.video_editing import edit_tool_call
                record['tool_call']=dict(id=member.strip('[]')+'_python',**edit_tool_call(task,resolved,staged))
                if task.get('operation_reasoner'):record['operation_reasoner']=task['operation_reasoner']
            if task['kind']=='video' and not native_video_output:
                normalized=job/'normalized.mp4'
                finish_video(staged,normalized,seconds=duration(staged),size=task['output_size']);normalized.replace(staged)
            elif task['kind']=='image' and task.get('render_state',{}).get('aspect_ratio'):
                from PIL import Image,ImageOps
                with Image.open(staged) as image:ImageOps.fit(image,tuple(task['output_size'])).save(job/'normalized.png')
                (job/'normalized.png').replace(staged)
            validate_media(staged,task['kind'])

        @contextmanager
        def turn_session(output_dir):
            nonlocal model,model_key
            if not shared_vlm and model is not None:
                model.close();model,model_key=None,None
            from tools.media_review import reviewer_session
            session=(reviewer_factory(output_dir) if reviewer_factory else reviewer_session(output_dir,
                base_url=review_base_url,timeout=review_timeout,require_running=shared_vlm,shared_model=review_model))
            with (nullcontext() if async_reviews else measure('review')), session as reviewer:
                yield reviewer

        def execute_unit(token, index):
            nonlocal mapping
            task = tasks[token]
            op = task['operation']
            task_seed = (seed + int(hashlib.sha256(token.encode()).hexdigest()[:15],16)) % 2**63
            fingerprint = request_fingerprint(token)
            previous = records.get(token, {})
            stem = token[2:-2]
            suffix = ((source.suffix or '.mp4') if op == 'source' else
                      Path(resolved[task['dependencies'][0]]).suffix if op == 'copy' else
                      '.png' if task['kind'] == 'image' else '.mp4')
            destination = media_dir/(stem+suffix.lower())
            if resume and previous.get('status') == 'complete' and previous.get('fingerprint') == fingerprint:
                cached = directory/previous['path']
                if cached.resolve() != destination.resolve():
                    raise ValueError('Unexpected recorded output path: '+token)
                if cached.is_file() and file_hash(cached) == previous.get('sha256'):
                    validate_media(cached, task['kind'])
                    resolved[token] = cached
                    if progress:
                        print(f'[{index+1}/{len(units)}] {token}: cached', flush=True)
                    return
            if destination.exists() and previous.get('path') != str(destination.relative_to(directory)):
                raise FileExistsError('Refusing to overwrite an untracked media file: '+str(destination))
            asset_started = time.monotonic()
            record = dict(status='running', operation=op, kind=task['kind'], backend=task.get('backend'),
                          path=str(destination.relative_to(directory)), fingerprint=fingerprint, seed=task_seed)
            if resume and previous.get('fingerprint')==fingerprint:
                for history_key in ('attempts','clip_edits','superseded_attempts'):
                    if previous.get(history_key):
                        record[history_key]=deepcopy(previous[history_key])
            records[token] = record
            save(manifest_path, manifest)
            job = Path(tempfile.mkdtemp(prefix=stem+'-', dir=work_root))
            staged = job/('output'+suffix.lower())
            if op in ('concatenate','copy') and not task.get('clip_edits'):
                from tools.video_editing import edit_tool_call
                record['tool_call']=dict(id=stem+'_python',**edit_tool_call(task,resolved,staged))
                if task.get('operation_reasoner'):
                    record['operation_reasoner']=task['operation_reasoner']
                save(manifest_path,manifest)
            needs_map = op == 'extract_shot' or any(ref.get('shot_id') for ref in task.get('references', []))
            if needs_map and mapping is None:
                mapping = refine_shots(resolved[root], observation, output_path=work_root/'shot_map.json')
            if op == 'source':
                shutil.copy2(source, staged)
            elif task.get('clip_edits'):
                edited = dict(resolved)
                child_records = record.setdefault('clip_edits',[])
                for part_index, part in enumerate(task['clip_edits']):
                    child = dict(part['generation'], kind='video')
                    part_job = job/f'clip-{part_index+1}'
                    part_job.mkdir()
                    part_output = part_job/'edited.mp4'
                    part_record = (child_records[part_index] if len(child_records)>part_index else dict(placeholder=part['placeholder']))
                    if len(child_records)<=part_index:
                        child_records.append(part_record)
                    part_started = time.monotonic()
                    yield from generate_reviewed(child, part_job, part_output, task_seed+part_index, part_record,
                                                attempt_token=clip_placeholder(token, part_index+1))
                    part_record['elapsed_seconds'] = round(time.monotonic()-part_started,6)
                    edited[part['placeholder']] = part_output
                record['clip_edits'] = child_records
                record['generation_seconds'] = sum(r['generation_seconds'] for r in child_records)
                record['model_reload_seconds'] = sum(r.get('model_reload_seconds', 0.0)
                                                     for r in child_records)
                from tools.video_editing import edit_tool_call
                record['tool_call']=dict(id=stem+'_python',**edit_tool_call(task,edited,staged))
                save(manifest_path,manifest)
                execute_edit(task, edited, observation, staged, shot_map=mapping)
            elif op in DETERMINISTIC:
                execute_edit(task, resolved, observation, staged, shot_map=mapping)
            else:
                yield from generate_reviewed(task, job, staged, task_seed, record)
            if op in DETERMINISTIC and op != 'source':
                if task['kind'] == 'video' and not native_video_output:
                    stream = next(s for s in probe(staged)['streams'] if s['codec_type'] == 'video')
                    if [stream['width'], stream['height']] != task['output_size']:
                        normalized = job/'normalized.mp4'
                        finish_video(staged, normalized, seconds=duration(staged), size=task['output_size'])
                        normalized.replace(staged)
                elif task['kind'] == 'image' and task.get('render_state', {}).get('aspect_ratio'):
                    from PIL import Image, ImageOps
                    with Image.open(staged) as source_image:
                        rendered = ImageOps.fit(source_image, tuple(task['output_size']))
                        rendered.save(job/'normalized.png')
                    (job/'normalized.png').replace(staged)
            validate_media(staged, task['kind'])
            output_hash = file_hash(staged)
            if op == 'source' and output_hash != source_hash:
                raise RuntimeError('Target copy differs from the original')
            staged.replace(destination)
            record.update(status='complete', sha256=output_hash, size_bytes=destination.stat().st_size,
                          elapsed_seconds=round(time.monotonic()-asset_started,6))
            resolved[token] = destination
            save(manifest_path, manifest)
            export()
            # IPC is socket-based; deleting completed request files keeps the resident model alive.
            shutil.rmtree(job)
            if record.get('backend') or record.get('clip_edits'):
                record['runtime_cleaned']=True
                for child_record in record.get('clip_edits',[]):child_record['runtime_cleaned']=True
                save(manifest_path,manifest)
            if progress:
                print(f'[{index+1}/{len(units)}] {token}: {op} -> {record["path"]}', flush=True)
            last_attempt = (record.get('attempts') or [record])[-1]
            return AssetEvent(token, last_attempt.get('executed_operation', op), record['path'],
                              generation_time(last_attempt))

        manifest.pop('pruning', None)
        event = None
        request = yield units
        index = 0
        while pending:
            review_result = request if isinstance(request, ReviewResult) else None
            token = review_result.token if review_result is not None else request
            if token not in pending or not set(units[token]['dependencies']).isdisjoint(pending):
                raise ValueError(f'Materialization task is not ready: {token}')
            dispatched = token
            try:
                members = group_for.get(token)
                leader = members[0] if members else token
                if members:
                    from scripts.dialogue_generation.reviewed_turn import iter_reviewed_turn, materialize_turn
                    if leader not in cursors:
                        options = dict(directory=directory, tasks=tasks, records=records, resolved=resolved,
                            context=generation_context(active_dialogue, leader), fingerprint=request_fingerprint,
                            generate=generate_for_turn, execute=execute_for_turn, session=turn_session,
                            checkpoint=lambda: save(manifest_path, manifest), export=export, seed=seed,
                            progress=progress)
                        if defer_retries:
                            cursors[leader] = iter_reviewed_turn(members, **options, defer_retries=True,
                                                                 async_reviews=async_reviews)
                        else:
                            materialize_turn(members, **options)
                elif leader not in cursors:
                    cursors[leader] = execute_unit(token, index)

                event = None
                if leader in cursors:
                    response = review_result.result() if review_result is not None else None
                    try:
                        event = cursors[leader].send(response)
                    except StopIteration as finished:
                        del cursors[leader]
                        event = finished.value
                if event is not None and event.token != dispatched:
                    raise RuntimeError(f'Materialization returned an event for {event.token} while running {dispatched}')
                if event is None or (isinstance(event, AssetEvent) and event.status == 'accepted'):
                    pending.remove(dispatched)
                    index += 1
            except AssetRejected as error:
                event = discard_rejected(error)
            if pending:
                request = yield event if defer_retries or (event is not None and event.status == 'exhausted') else None
        backup = directory/'dialogue.placeholders.md'
        if not backup.exists():
            atomic_text(backup, original)
        completed = sum(row.get('status') == 'complete' for row in records.values())
        status = ('partial' if completed else 'failed') if exhausted else 'complete'
        if dialogue.get('turns'):
            export(status)
        if exhausted and dialogue.get('turns'):
            annotated = (directory/'dialogue.md').read_text()
        else:
            if exhausted:
                original = TOKEN.sub(lambda match: '' if match[0] in omitted else match[0], original)
            annotated = render_markdown(original, tasks, records)
        atomic_text(directory/'dialogue.annotated.md', annotated)
        if not dialogue.get('turns'):
            atomic_text(directory/'dialogue.md', annotated)
        dialogue_file = directory/'dialogue.json'
        if dialogue_file.exists():
            dialogue = read_json(dialogue_file)
            dialogue['media_generated'] = bool(completed)
            dialogue['materialization'] = dict(status=status, manifest='media_manifest.json',
                official_sequence='sequence.json', official_dialogue='dialogue.md')
            if exhausted:
                dialogue['materialization']['pruning'] = manifest['pruning']
            save(dialogue_file, dialogue)
        manifest.update(status=status, completed=completed, elapsed_seconds=round(time.monotonic()-started,3))
        manifest.pop('error', None)
        save(manifest_path, manifest)
        succeeded = True
        return (manifest, event) if defer_retries else manifest
    except BaseException as exc:
        if manifest_active:
            manifest.update(status='failed', error=str(exc))
            if token in manifest.get('media', {}) and manifest['media'][token].get('status') == 'running':
                manifest['media'][token].update(status='failed', error=str(exc))
            save(manifest_path, manifest)
            if 'export' in locals():
                export('failed')
        raise
    finally:
        try:
            for cursor in locals().get('cursors', {}).values():
                cursor.close()
            if model is not None:
                model.close()
        finally:
            if succeeded:
                # NFS may briefly retain .nfs files while closed workers exit.
                for attempt in range(20):
                    try:
                        shutil.rmtree(directory/'_work')
                        break
                    except FileNotFoundError:
                        break
                    except OSError:
                        if attempt==19:raise
                        time.sleep(.25)
                for completed_record in manifest['media'].values():
                    if completed_record.get('backend'):completed_record['runtime_cleaned']=True
                save(manifest_path,manifest)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def generated_crop_cache(directory, video, query):
    key = digest(dict(source_sha256=file_hash(video),query=query,version=2,video_fps=DEFAULT_VIDEO_FPS))
    path = Path(directory)/'_inputs/generated_crops'/(key+'.png')
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata = path.with_suffix('.json')
    valid = path.is_file() and metadata.is_file() and read_json(metadata).get('sha256') == file_hash(path)
    return path, valid


def crop_generated_reference(video, query, output, *, qwen=None, base_url='http://127.0.0.1:8000', timeout=600, require_running=False):
    """Localize a generated subject; start/stop vLLM only when we own it."""
    from tools.find_crop_in_video import find_crop_in_video
    from cocreation import VLLMQwen
    import urllib.request
    def healthy():
        try:
            with urllib.request.urlopen(base_url+'/health', timeout=2) as response:
                return response.status == 200
        except OSError:
            return False
    process, owned_model = None, qwen is None
    repo = Path(__file__).resolve().parents[2]
    try:
        if qwen is None:
            if not healthy():
                if require_running:
                    raise RuntimeError('Shared localization service is unavailable: '+base_url)
                log = Path(output).parent/'crop-vllm.log'
                with log.open('w') as stream:
                    process = subprocess.Popen(['bash','-l',str(repo/'scripts/vllm/start.sh')],
                        cwd=repo,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                deadline = time.monotonic()+timeout
                while not healthy():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('On-demand crop vLLM failed to start; see '+str(log))
                    time.sleep(1)
            qwen = VLLMQwen(base_url=base_url,timeout=timeout)
        result = find_crop_in_video(video, query, qwen=qwen, crop_output=output,
            video_fps=DEFAULT_VIDEO_FPS, fixed_rate=True, output_dir=Path(output).parent/'localization')
        return result['crop_path']
    finally:
        if owned_model and qwen is not None:
            qwen.close()
        if process is not None and process.poll() is None:
            # Terminate only the process group started by this crop operation.
            # A pre-existing external service is never owned or stopped here.
            import os
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


def frame_query(task, observation):
    assets = {a['id']:a for a in observation.get('assets', [])}
    descriptions = [assets[aid]['description'] for aid in task.get('asset_ids', []) if aid in assets]
    if not descriptions:
        raise ValueError('Target-frame language search requires observed asset descriptions')
    return 'Find a representative frame showing: ' + '; '.join(descriptions)


def target_frame_hints(task, observation):
    assets = {a['id']:a for a in observation.get('assets', [])}
    return sorted({float(t) for aid in task.get('asset_ids', []) if aid in assets
                   for t in assets[aid].get('frames', [])
                   if type(t) in (int,float) and math.isfinite(t) and t >= 0})


def frame_fingerprint(source_hash, query, hints):
    return digest(dict(source_sha256=source_hash, query=query, search_version=3,
                       fps=DEFAULT_VIDEO_FPS, hint_timestamps=hints))


def prepare_frames(run_dir, *, source_video=None, base_url='http://127.0.0.1:8000',
                   timeout=600, qwen=None, progress=True):
    """Optional VLM preparation, while vLLM is still running. Never generate media."""
    from cocreation import VLLMQwen
    from tools.find_video_frame import find_video_frame
    directory = Path(run_dir).expanduser().resolve(strict=True)
    tasks, order, root, source, observation, _ = load_contract(directory, source_video)
    if root is None:
        completed = read_json(directory/'media_manifest.json').get('media', {}) if (directory/'media_manifest.json').exists() else {}
        prepared, pending = 0, []
        for token in order:
            for index, ref in enumerate(tasks[token].get('references', [])):
                if not ref.get('crop_query'):
                    continue
                record = completed.get(ref['placeholder'], {})
                if record.get('status') != 'complete':
                    pending.append(token)
                    continue
                source_clip = directory/record['path']
                output, valid = generated_crop_cache(directory,source_clip,ref['crop_query'])
                if not valid:
                    output.unlink(missing_ok=True)
                    crop_generated_reference(source_clip,ref['crop_query'],output,qwen=qwen,base_url=base_url,timeout=timeout)
                    save(output.with_suffix('.json'),dict(sha256=file_hash(output)))
                prepared += 1
        return dict(status='prepared',selections=prepared,pending_generated_videos=pending)
    source_hash = file_hash(source)
    selections_file = directory/'frame_selections.json'
    selected = read_json(selections_file) if selections_file.exists() else dict(version=1, references={})
    if selected.get('version') != 1:
        raise ValueError('Unsupported frame selections version')
    entries = selected['references']
    model = qwen if qwen is not None else VLLMQwen(base_url=base_url, timeout=timeout, output_dir=directory/'_inputs/target_frames/vllm')
    frame_root = directory/'_inputs/target_frames'
    frame_root.mkdir(parents=True, exist_ok=True)
    try:
        for token in order:
            task = tasks[token]
            for index, ref in enumerate(task.get('references', [])):
                if ref.get('placeholder') != root or ref['kind'] != 'image':
                    continue
                key = token+':'+str(index)
                query = frame_query(task, observation)
                hints = target_frame_hints(task, observation)
                fingerprint = frame_fingerprint(source_hash, query, hints)
                previous = entries.get(key, {})
                if previous.get('fingerprint') == fingerprint:
                    cached = directory/previous['path']
                    if cached.is_file() and file_hash(cached) == previous.get('sha256'):
                        validate_media(cached,'image')
                        continue
                job = Path(tempfile.mkdtemp(prefix=token[2:-2]+'-',dir=frame_root))
                frame = job/'selected.png'
                seconds = find_video_frame(source, query, qwen=model,
                    video_fps=DEFAULT_VIDEO_FPS, fixed_rate=True, hint_timestamps=hints,
                    output_dir=job/'selection', frame_output=frame)
                if seconds is None:
                    raise RuntimeError('No matching target-video frame found: '+token)
                validate_media(frame,'image')
                entries[key] = dict(fingerprint=fingerprint, source_sha256=source_hash, query=query,
                    hint_timestamps=hints, search_version=3, frame_seconds=seconds,
                    path=str(frame.relative_to(directory)), sha256=file_hash(frame))
                save(selections_file, selected)
                if progress:
                    print(f'{token}: selected target frame at {seconds:.6f}s',flush=True)
    finally:
        if qwen is None:
            model.close()
    return dict(status='prepared', selections=len(entries), path=str(selections_file))


def apply_frame_selections(directory, tasks, root, source_hash, observation):
    path = directory/'frame_selections.json'
    if not path.exists():
        return
    selected = read_json(path)
    if selected.get('version') != 1:
        raise ValueError('Unsupported frame selections version')
    for token, task in tasks.items():
        for index, ref in enumerate(task.get('references', [])):
            if ref.get('placeholder') != root or ref['kind'] != 'image':
                continue
            previous = selected.get('references', {}).get(token+':'+str(index))
            if previous is None:
                raise ValueError('Frame preparation is incomplete; rerun --prepare-frames: '+token)
            query = frame_query(task, observation)
            expected = frame_fingerprint(source_hash, query, target_frame_hints(task, observation))
            frame = directory/previous['path']
            if previous['fingerprint'] != expected or not frame.is_file() or file_hash(frame) != previous['sha256']:
                raise ValueError('Prepared reference frame changed; rerun --prepare-frames: '+token)
            # Extracting the same timestamp from the byte-identical target copy
            # reproduces the selected frame without invoking the VLM.
            ref['frame_seconds'] = previous['frame_seconds']


def main():
    import signal
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--prepare-frames', action='store_true', help='Preselect crops from available generated videos, then exit; normal forward materialization also localizes on demand')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000', help='vLLM URL for --prepare-frames only')
    parser.add_argument('--timeout', type=float, default=600, help='vLLM timeout for --prepare-frames only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-media', action=argparse.BooleanOptionalAction, default=True,
                        help='Brief JSON review of every generated image/video, at most two repairs, fail after third rejection')
    parser.add_argument('--review-base-url', default='http://127.0.0.1:8019')
    parser.add_argument('--review-timeout', type=float, default=600)
    parser.add_argument('--shared-vlm', action='store_true', help='Reuse the existing review service and keep the generation model loaded during critique/localization')
    parser.add_argument('--match-video-brightness', action=argparse.BooleanOptionalAction, default=True,
        help='Match each video edit to the first clip in its edit chain with a joint Y-only tone curve (default: on)')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and show the construction order without generating')
    parser.add_argument('--source-video', type=Path, help='Explicit replacement for a relocated source path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image-width', type=int, help='Override backend recommendation; H3 Image defaults to ~16 MP')
    parser.add_argument('--image-height', type=int)
    parser.add_argument('--video-width', type=int)
    parser.add_argument('--video-height', type=int)
    parser.add_argument('--video-steps', type=int, help='Override sampling steps: original H3 defaults to 50, Turbo LoRA to 6, VDN to 7')
    parser.add_argument('--video-turbo-lora', action='store_true', help='Use Turbo for every H3 request; disabled by default')
    parser.add_argument('--new-video-turbo-lora', action='store_true', help='Use Turbo only for new videos, keeping video_edit on the original model')
    parser.add_argument('--new-video-steps', type=int, help='Override sampling steps for new H3 videos')
    parser.add_argument('--edit-video-steps', type=int, help='Override sampling steps for H3 video edits')
    parser.add_argument('--edit-video-sampler', choices=('euler','res_multistep'), default='euler')
    parser.add_argument('--output-width', type=int, help='Override output canvas; defaults to 1280x720 for forward runs without metadata')
    parser.add_argument('--output-height', type=int)
    parser.add_argument('--native-video-output', action='store_true', help='Keep model dimensions, frame timing and audio without export resizing/trimming; video edits still receive default brightness matching (disable with --no-match-video-brightness)')
    parser.add_argument('--video-two-pass', action='store_true', help='Use 8 base steps plus 3 refinement steps; also set --video-steps 8')
    args = parser.parse_args()
    try:
        if args.prepare_frames:
            if args.dry_run:
                parser.error('--prepare-frames and --dry-run are separate modes')
            result = prepare_frames(args.run_dir, source_video=args.source_video,
                                    base_url=args.base_url, timeout=args.timeout)
        else:
            options = vars(args).copy()
            for key in ('prepare_frames','base_url','timeout'):
                options.pop(key)
            result = materialize(**options)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Materialization failed: {exc}\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'media'}, ensure_ascii=False, indent=2))
    if result.get('status') == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
