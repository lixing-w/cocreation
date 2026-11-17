"""Materialize one dialogue run from its executable placeholders.json catalog."""
from __future__ import annotations

import argparse
from collections import Counter
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

from scripts.dialogue_generation.generation import construction_order, FLUX_MODEL_ID
from tools.video_editing import prepare_inputs, execute_edit, refine_shots, probe

TOKEN = re.compile(r'\[\[(image|video)_\d{4}\]\]')
DETERMINISTIC = {'source', 'copy', 'extract_frame', 'extract_shot', 'concatenate'}
GENERATIVE = {'text_generation', 'asset_reconstruction', 'local_edit', 'reference_generation', 'video_edit'}


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
            if backend not in ('flux', 'vdn_h3') or not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('Missing model backend/prompt: '+token)
            if (backend == 'flux') != (row['kind'] == 'image'):
                raise ValueError('Backend and output kind disagree: '+token)
            counts = Counter()
            for ref in row.get('references', []):
                kind = ref.get('kind')
                if kind not in ('image', 'video', 'audio') or (backend == 'flux' and kind != 'image'):
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
            if part.get('placeholder') not in row['dependencies'] or child.get('backend')!='vdn_h3' or child.get('operation')!='video_edit' or not child.get('prompt') or child.get('status')!='ready':
                raise ValueError('Finish rewriting the independent clip edit: '+token)
            if child.get('dependencies') != [part['placeholder']] or len(child.get('references', []))!=1 or child['references'][0].get('placeholder')!=part['placeholder']:
                raise ValueError('An assembly edit must reference exactly one individual source clip: '+token)
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
    order = construction_order(tasks)
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


def preflight_models(tasks):
    """Check offline model installations without starting workers or allocating VRAM."""
    report = {}
    requests=[request for task in tasks.values() for request in [task]+[part['generation'] for part in task.get('clip_edits',[])]]
    for task in requests:
        if task['operation'] not in GENERATIVE:
            continue
        backend = task['backend']
        model_id = task.get('model_id') or (FLUX_MODEL_ID if backend == 'flux' else 'OpenVDN/vdn-minimax-h3-edge')
        key = backend+':'+model_id
        if key in report:
            continue
        if backend == 'flux':
            snapshot = cached_flux_snapshot(model_id)
            report[key] = dict(snapshot=str(snapshot), revision=snapshot.name)
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


def default_model(backend, model_id, output_dir):
    from cocreation import Flux, VDNH3, Runtime, VDNRuntime
    if backend == 'flux':
        model_id = model_id or FLUX_MODEL_ID
        snapshot = cached_flux_snapshot(model_id)
        return Flux(model_id=model_id,
                    runtime=Runtime(local_files_only=True, revision=snapshot.name), output_dir=output_dir)
    return VDNH3(runtime=VDNRuntime(auto_setup=False, local_files_only=True), output_dir=output_dir)


def pad_short_references(references, directory):
    """H3 requires >=1s references; hold short clips without dropping any source frames."""
    from dataclasses import replace
    result = []
    for index, ref in enumerate(references):
        if ref.kind == 'video' and duration(ref.path) < 1.0:
            path = directory/f'padded-reference-{index}.mp4'
            info = probe(ref.path)
            audio = any(s['codec_type']=='audio' for s in info['streams'])
            command = ['ffmpeg','-nostdin','-v','error','-i',str(ref.path),
                       '-vf','tpad=stop_mode=clone:stop_duration=1','-t','1',
                       '-c:v','libx264','-crf','18','-pix_fmt','yuv420p']
            if audio:
                command += ['-af','apad','-c:a','aac']
            subprocess.run(command+[str(path)],check=True)
            ref = replace(ref,path=path)
        result.append(ref)
    return result


def finish_video(source, destination, *, seconds, size, audio_source=None, grayscale=False):
    """Normalize generated clips for exact-order assembly; retain edit-source audio."""
    width, height = size
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source)]
    if audio_source is not None:
        command += ['-i', str(audio_source)]
    command += ['-map', '0:v:0', '-map', '1:a:0?' if audio_source is not None else '0:a:0?',
                '-t', str(seconds), '-vf',
                f'scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,'
                f'crop={width}:{height},setsar=1'+(',format=gray' if grayscale else ''),
                '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)


def materialize(run_dir, *, resume=False, source_video=None, seed=0,
                image_width=1024, image_height=1024, video_width=None, video_height=None,
                video_steps=16, video_two_pass=False, output_width=None, output_height=None, dry_run=False, model_factory=None, progress=True):
    directory = Path(run_dir).expanduser().resolve(strict=True)
    tasks, order, root, source, observation, original = load_contract(directory, source_video)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
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
        output_size = (1280, 736)
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
    if type(video_two_pass) is not bool or type(video_steps) is not int or not 1 <= video_steps <= 32 or (video_two_pass and video_steps != 8):
        raise ValueError('VDN steps must be 1–32; --video-two-pass requires --video-steps 8')
    if any(type(v) is not int or v < (512 if video_two_pass else 256) or v % 32 for v in (video_width, video_height)):
        raise ValueError('VDN dimensions must be multiples of 32 (>=256 single-pass, >=512 two-pass)')
    config = dict(seed=seed, image_size=[image_width,image_height],
                  video_size=[video_width,video_height], video_steps=video_steps, video_two_pass=video_two_pass, output_size=list(output_size))
    summary = dict(run_dir=str(directory), construction_order=order,
                   operations=dict(Counter(row['operation'] for row in tasks.values())), settings=config)
    if dry_run:
        return dict(status='planned', **summary)
    if model_factory is None:
        summary['models'] = preflight_models(tasks)
    work_root = directory/'_work/materialize'
    work_root.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.materialize.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another materializer is using this run directory') from None
    factory = model_factory or default_model
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
        def generate_media(task, job, staged, task_seed, record):
            nonlocal model, model_key
            # A video-to-image dependency needs a brief VLM localization
            # phase. Release the current generation worker before loading it.
            def crop_reference(path, query, output):
                nonlocal model, model_key
                cached, valid = generated_crop_cache(directory,path,query)
                if valid:
                    return str(cached)
                if model is not None:
                    model.close()
                    model, model_key = None, None
                cached.unlink(missing_ok=True)
                cropped = crop_generated_reference(path, query, cached)
                save(cached.with_suffix('.json'),dict(sha256=file_hash(cropped)))
                return cropped
            references, prompt = prepare_inputs(task, resolved, observation, job/'inputs',
                shot_map=mapping, crop_resolver=crop_reference)
            if task['backend'] == 'vdn_h3':
                references = pad_short_references(references, job/'inputs')
                from dataclasses import replace
                short_refs=[]
                for ref_index, ref in enumerate(references):
                    if ref.kind=='video' and duration(ref.path)>4.001:
                        trimmed=job/'inputs'/f'four-second-reference-{ref_index}.mp4'
                        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(ref.path),'-t','4','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac',str(trimmed)],check=True)
                        record.setdefault('reference_trims',[]).append(dict(reference=ref_index,start_seconds=0,seconds=4,source_seconds=duration(ref.path)))
                        ref=replace(ref,path=trimmed)
                    short_refs.append(ref)
                references=short_refs
            record['references'] = [
                dict(kind=ref.kind, path=str(Path(ref.path).relative_to(directory)),
                     **{key:slot[key] for key in ('label','placeholder','frame_seconds','shot_id','crop_query') if key in slot})
                for ref, slot in zip(references,task.get('references', []))]
            key = (task['backend'], task.get('model_id'))
            if model_key != key:
                if model is not None:
                    model.close()
                    model = None
                model = factory(*key, job/'runtime')
                # NFS keeps deleted-but-open logs as .nfs files. Resident logs
                # therefore live outside the disposable per-asset request tree.
                model.session_dir = work_root/'_sessions'/'__'.join(job.relative_to(work_root).parts)
                model_key = key
            options = dict(seed=task_seed)
            audio_source = None
            if key[0] == 'flux':
                options.update(width=image_width, height=image_height)
            else:
                output_seconds = min(4., float(task.get('output_seconds', task.get('seconds', 4))))
                if task.get('shot_id') and mapping is not None:
                    shot = mapping['shots'][task['shot_id']]
                    output_seconds = min(4., shot['end_seconds']-shot['start_seconds'])
                if task.get('preprocess') == 'concatenate':
                    raise ValueError('Whole-cut video inference is disabled; rewrite the contract into per-clip edits')
                input_seconds = [duration(r.path) for r in references if r.kind == 'video']
                seconds = max(1.625, output_seconds, *input_seconds)
                if not math.isfinite(seconds) or seconds > 4.001 or output_seconds <= 0:
                    raise ValueError('VDN task exceeds its 4-second clip limit: '+token)
                options.update(seconds=seconds, width=video_width, height=video_height,
                               steps=video_steps, two_pass=video_two_pass)
                if task['operation'] == 'video_edit':
                    audio_source = next((r.path for r in references if r.kind == 'video'), None)
            if hasattr(model, 'output_dir'):
                model.output_dir = job/'runtime'
            call_started = time.monotonic()
            try:
                result = model(prompt, references, **options)
            finally:
                record['generation_seconds'] = round(time.monotonic()-call_started, 6)
            if result.get('status') != 'ok' or not result.get('path'):
                raise RuntimeError('Generator did not return a successful media path: '+token)
            generated = Path(result['path'])
            validate_media(generated, task['kind'])
            if key[0] == 'flux':
                from PIL import Image
                with Image.open(generated) as image:
                    (image.convert('L').convert('RGB') if task.get('render_state',{}).get('color_mode')=='black_and_white' else image).save(staged, format='PNG')
            else:
                finish_video(generated, staged, seconds=output_seconds,
                             size=output_size, audio_source=audio_source,
                             grayscale=task.get('render_state',{}).get('color_mode')=='black_and_white')
                if abs(duration(staged)-output_seconds) > max(.15, output_seconds*.02):
                    raise RuntimeError('Generated video is shorter than its required output duration: '+token)
            record['model_id'] = task.get('model_id') or getattr(model, 'model_id', None)
            receipt = {k:v for k,v in result.items() if k not in ('path','job_dir','request_path','resident')}
            report_path = generated.with_suffix('.request.json')
            if report_path.is_file():
                receipt['upstream_report'] = read_json(report_path)
            record_dir = directory/'generation_records'
            record_dir.mkdir(exist_ok=True)
            receipt_path = record_dir/('__'.join(job.relative_to(work_root).parts)+'.json')
            save(receipt_path, dict(options=options, result=receipt, generation_seconds=record['generation_seconds']))
            record['generation_record'] = str(receipt_path.relative_to(directory))
            record['runtime_cleaned'] = False
            record['options'] = options

        for index, token in enumerate(order):
            task = tasks[token]
            op = task['operation']
            task_seed = (seed + int(hashlib.sha256(token.encode()).hexdigest()[:15],16)) % 2**63
            fingerprint = digest(dict(task=task, settings=config, source_hash=source_hash,
                observation=observation, dependencies={p:records[p]['sha256'] for p in task['dependencies']}))
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
                        print(f'[{index+1}/{len(order)}] {token}: cached', flush=True)
                    continue
            if destination.exists() and previous.get('path') != str(destination.relative_to(directory)):
                raise FileExistsError('Refusing to overwrite an untracked media file: '+str(destination))
            asset_started = time.monotonic()
            record = dict(status='running', operation=op, backend=task.get('backend'),
                          path=str(destination.relative_to(directory)), fingerprint=fingerprint, seed=task_seed)
            records[token] = record
            save(manifest_path, manifest)
            job = Path(tempfile.mkdtemp(prefix=stem+'-', dir=work_root))
            staged = job/('output'+suffix.lower())
            needs_map = op == 'extract_shot' or any(ref.get('shot_id') for ref in task.get('references', []))
            if needs_map and mapping is None:
                mapping = refine_shots(resolved[root], observation, output_path=work_root/'shot_map.json')
            if op == 'source':
                shutil.copy2(source, staged)
            elif task.get('clip_edits'):
                edited = dict(resolved)
                child_records = []
                for part_index, part in enumerate(task['clip_edits']):
                    child = dict(part['generation'], kind='video')
                    part_job = job/f'clip-{part_index+1}'
                    part_job.mkdir()
                    part_output = part_job/'edited.mp4'
                    part_record = dict(placeholder=part['placeholder'])
                    part_started = time.monotonic()
                    generate_media(child, part_job, part_output, task_seed+part_index, part_record)
                    part_record['elapsed_seconds'] = round(time.monotonic()-part_started,6)
                    child_records.append(part_record)
                    edited[part['placeholder']] = part_output
                record['clip_edits'] = child_records
                record['generation_seconds'] = sum(r['generation_seconds'] for r in child_records)
                execute_edit(task, edited, observation, staged, shot_map=mapping)
            elif op in DETERMINISTIC:
                execute_edit(task, resolved, observation, staged, shot_map=mapping)
            else:
                generate_media(task, job, staged, task_seed, record)
            validate_media(staged, task['kind'])
            output_hash = file_hash(staged)
            if op == 'source' and output_hash != source_hash:
                raise RuntimeError('Target copy differs from the original')
            staged.replace(destination)
            record.update(status='complete', sha256=output_hash, size_bytes=destination.stat().st_size,
                          elapsed_seconds=round(time.monotonic()-asset_started,6))
            resolved[token] = destination
            save(manifest_path, manifest)
            # IPC is socket-based; deleting completed request files keeps the resident model alive.
            shutil.rmtree(job)
            if record.get('backend') or record.get('clip_edits'):
                record['runtime_cleaned']=True
                for child_record in record.get('clip_edits',[]):child_record['runtime_cleaned']=True
                save(manifest_path,manifest)
            if progress:
                print(f'[{index+1}/{len(order)}] {token}: {op} -> {record["path"]}', flush=True)
        backup = directory/'dialogue.placeholders.md'
        if not backup.exists():
            atomic_text(backup, original)
        atomic_text(directory/'dialogue.md', render_markdown(original, tasks, records))
        dialogue_file = directory/'dialogue.json'
        if dialogue_file.exists():
            dialogue = read_json(dialogue_file)
            dialogue['media_generated'] = True
            dialogue['materialization'] = dict(status='complete', manifest='media_manifest.json')
            save(dialogue_file, dialogue)
        manifest.update(status='complete', completed=len(order), elapsed_seconds=round(time.monotonic()-started,3))
        manifest.pop('error', None)
        save(manifest_path, manifest)
        succeeded = True
        return manifest
    except BaseException as exc:
        if manifest_active:
            manifest.update(status='failed', error=str(exc))
            if token in manifest.get('media', {}) and manifest['media'][token].get('status') == 'running':
                manifest['media'][token].update(status='failed', error=str(exc))
            save(manifest_path, manifest)
        raise
    finally:
        try:
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
    key = digest(dict(source_sha256=file_hash(video),query=query,version=1))
    path = Path(directory)/'_inputs/generated_crops'/(key+'.png')
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata = path.with_suffix('.json')
    valid = path.is_file() and metadata.is_file() and read_json(metadata).get('sha256') == file_hash(path)
    return path, valid


def crop_generated_reference(video, query, output, *, qwen=None, base_url='http://127.0.0.1:8000', timeout=600):
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
        result = find_crop_in_video(video,query,qwen=qwen,crop_output=output,
            output_dir=Path(output).parent/'localization')
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
    return digest(dict(source_sha256=source_hash, query=query, search_version=2,
                       fps=4., coarse=32, refine=16, hint_timestamps=hints))


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
                seconds = find_video_frame(source, query, qwen=model, coarse_frames=32,
                    refine_frames=16, video_fps=4., hint_timestamps=hints,
                    output_dir=job/'selection', frame_output=frame)
                if seconds is None:
                    raise RuntimeError('No matching target-video frame found: '+token)
                validate_media(frame,'image')
                entries[key] = dict(fingerprint=fingerprint, source_sha256=source_hash, query=query,
                    hint_timestamps=hints, search_version=2, frame_seconds=seconds,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--prepare-frames', action='store_true', help='Preselect crops from available generated videos, then exit; normal forward materialization also localizes on demand')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000', help='vLLM URL for --prepare-frames only')
    parser.add_argument('--timeout', type=float, default=600, help='vLLM timeout for --prepare-frames only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and show the construction order without generating')
    parser.add_argument('--source-video', type=Path, help='Explicit replacement for a relocated source path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image-width', type=int, default=1024)
    parser.add_argument('--image-height', type=int, default=1024)
    parser.add_argument('--video-width', type=int)
    parser.add_argument('--video-height', type=int)
    parser.add_argument('--video-steps', type=int, default=16)
    parser.add_argument('--output-width', type=int, help='Override output canvas; defaults to 1280x736 for forward runs without metadata')
    parser.add_argument('--output-height', type=int)
    parser.add_argument('--video-two-pass', action='store_true', help='Opt into legacy two-pass; also set --video-steps 8')
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


if __name__ == '__main__':
    main()
