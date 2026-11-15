"""Observe video frames as catalog-bound dialogue_generation input.

Ollama receives ordered images, not an MP4 or a native video tensor. A caller
may instead pass an existing cocreation.Qwen; ownership stays with the caller.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dialogue_generation.asset_variations import CATALOG, CATEGORIES, validate_attributes
from scripts.dialogue_generation.planner import observation as validate_observation
from tools.dialogue_models import DEFAULT_OLLAMA_MODEL, resolve_model
from tools.find_video_frame import frame_timestamps, sample_indices, extract_frames
from tools.video_observer_prompts import SYSTEM, inventory_prompt, attributes_prompt, review_prompt


def _object(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def inventory_schema(frame_count, max_assets):
    text = dict(type='string', minLength=1)
    frame = dict(type='integer', minimum=1, maximum=frame_count)
    asset = _object(dict(id=text, kind=dict(type='string', enum=list(CATEGORIES)), name=text,
                         frame_ids=dict(type='array', minItems=1, maxItems=2,
                                                          uniqueItems=True, items=frame)))
    shot = _object(dict(first_frame=frame, last_frame=frame, description=text,
                        assets=dict(type='array', minItems=1, maxItems=max_assets,
                                    uniqueItems=True, items=text)))
    return _object(dict(assets=dict(type='array', minItems=1, maxItems=max_assets, items=asset),
                        shots=dict(type='array', minItems=1, maxItems=4, items=shot)))


def attribute_schema(kind):
    # Cross-field compatibility is checked by validate_attributes after decoding.
    return _object({key: dict(type='string', enum=[CATALOG['unknown_value']] + [v for v in spec['values'] if v != CATALOG['unknown_value']])
                    for key, spec in CATEGORIES[kind]['attributes'].items()})


def review_schema(attributes):
    return _object({key: dict(type='string', enum=list(dict.fromkeys([CATALOG['unknown_value'], value])))
                    for key, value in attributes.items()})


def validate_review(asset, proposed, reviewed):
    validate_attributes(asset, reviewed)
    if any(value not in (CATALOG['unknown_value'], proposed[key]) for key, value in reviewed.items()):
        raise ValueError('Review can only keep a proposed value or change it to unknown')
    return reviewed


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{label} must contain exactly: {", ".join(keys)}')


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_inventory(value, frame_count, max_assets):
    _keys(value, ['assets', 'shots'], 'Inventory')
    assets, shots = value['assets'], value['shots']
    if not isinstance(assets, list) or not 1 <= len(assets) <= max_assets:
        raise ValueError(f'Expected 1–{max_assets} assets')
    if not isinstance(shots, list) or not 1 <= len(shots) <= 4:
        raise ValueError('Expected 1–4 shots')
    ids, names = set(), set()
    for asset in assets:
        _keys(asset, ['id', 'kind', 'name', 'frame_ids'], 'Asset')
        if not all(_text(asset[k]) for k in ('id', 'kind', 'name')):
            raise ValueError('Asset id, kind and name must be nonempty strings')
        if asset['id'] in ids or asset['kind'] not in CATEGORIES:
            raise ValueError('Use unique asset ids and a listed asset kind')
        ids.add(asset['id'])
        name_key = (asset['kind'], ' '.join(asset['name'].casefold().split()))
        if name_key in names:
            raise ValueError('Repeated asset name. List the same subject once with one or two representative frames, even if it appears in many frames. Distinct subjects need distinguishable names.')
        names.add(name_key)
        frames = asset['frame_ids']
        if (not isinstance(frames, list) or not 1 <= len(frames) <= 2
                or any(type(f) is not int or not 1 <= f <= frame_count for f in frames)
                or len(set(frames)) != len(frames)):
            raise ValueError('frame_ids must contain one or two distinct supplied frame numbers')
    used, next_frame = set(), 1
    for shot in shots:
        _keys(shot, ['first_frame', 'last_frame', 'description', 'assets'], 'Shot')
        first, last = shot['first_frame'], shot['last_frame']
        if (type(first) is not int or type(last) is not int
                or first != next_frame or not first <= last <= frame_count):
            raise ValueError('Shots must cover all supplied frames in order without gaps or overlaps')
        next_frame = last + 1
        refs = shot['assets']
        if (not _text(shot['description']) or not isinstance(refs, list) or not refs
                or any(not isinstance(a, str) or a not in ids for a in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError('Each shot needs a description and distinct known asset ids')
        used.update(refs)
    if next_frame != frame_count + 1 or used != ids:
        raise ValueError('Cover all frames and use every asset in at least one shot')
    for asset in assets:
        if any(not any(s['first_frame'] <= f <= s['last_frame'] and asset['id'] in s['assets']
                       for s in shots) for f in asset['frame_ids']):
            raise ValueError(f"Asset {asset['id']} frame_ids must be in shots that list it")
    return value


def _duration(path, timestamps):
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration', '-of', 'json', str(path)], text=True)
    try:
        duration = float(json.loads(raw)['streams'][0]['duration'])
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError('Video stream must have a known duration') from exc
    if not math.isfinite(duration) or duration <= timestamps[-1]:
        raise ValueError('Invalid video stream duration')
    return duration


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class VideoObserver:
    """Reuse one stateless Qwen transport across assets and videos.

    Ollama retains the selected weights for 15 minutes, allowing the following
    dialogue generation run to reuse them. close() only closes an owned local
    Qwen, never an injected instance or the Ollama server.
    """
    def __init__(self, *, backend=None, qwen=None, model=None,
                 base_url='http://127.0.0.1:11434', timeout=600,
                 max_tokens=4096, context_tokens=32768, seed=0, retries=1):
        self.backend = backend or ('local' if qwen is not None else 'ollama')
        if self.backend not in ('ollama', 'local'):
            raise ValueError('backend must be ollama or local')
        if qwen is not None and (self.backend != 'local' or model is not None):
            raise ValueError('An injected qwen selects its own local model; omit model')
        if (not math.isfinite(timeout) or timeout <= 0 or max_tokens < 128 or context_tokens < 2048
                or type(retries) is not int or not 0 <= retries <= 3
                or type(seed) is not int or not 0 <= seed < 2**31):
            raise ValueError('Invalid timeout, token budget, retries or seed')
        self.model, mode = resolve_model(self.backend, model)
        if qwen is not None:
            self.model = getattr(qwen, 'model_id', 'injected-qwen')
        elif mode != 'instruct':
            raise ValueError('Use an explicit Instruct model for this observation tool')
        self.qwen, self.owned = qwen, False
        self.base_url, self.timeout = base_url.rstrip('/'), timeout
        self.max_tokens, self.context_tokens = max_tokens, context_tokens
        self.seed, self.retries = seed, retries

    def close(self):
        if self.owned and self.qwen is not None:
            self.qwen.close()
            self.qwen = None
            self.owned = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, prompt, images, schema, directory):
        if self.backend == 'local':
            from cocreation import Qwen, Reference
            if self.qwen is None:
                self.qwen = Qwen(model_id=self.model)
                self.owned = True
            refs = [Reference(f'frame_{i}', str(path), 'image') for i, path in enumerate(images, 1)]
            media = '\n'.join('{{artifact:' + r.artifact_id + '}}' for r in refs)
            result = self.qwen(media + '\n' + prompt, refs, system_prompt=SYSTEM,
                max_new_tokens=self.max_tokens, max_pixels=768*768, seed=self.seed,
                output_dir=directory / 'runtime', temperature=0.1, top_p=0.95, top_k=50)
            return result.get('text', '')
        # Reject clearly undersized contexts before Ollama can truncate a frame
        # sequence. Qwen's Ollama vision encoder reserves at least 1024 tokens
        # per image; use a conservative text budget and allow media framing.
        from PIL import Image
        image_budget = 0
        for path in images:
            with Image.open(path) as image:
                image_budget += max(1024, math.ceil(image.width / 28) * math.ceil(image.height / 28)) + 64
        budget = image_budget + math.ceil(len((SYSTEM + prompt).encode('utf-8')) / 2) + self.max_tokens + 256
        if budget > self.context_tokens:
            raise ValueError('Frames and output exceed the estimated context budget; reduce --frames/--max-side or increase --context-tokens')
        body = dict(model=self.model, stream=False, keep_alive='15m', format=schema,
            messages=[dict(role='system', content=SYSTEM), dict(role='user', content=prompt,
                images=[base64.b64encode(p.read_bytes()).decode('ascii') for p in images])],
            options=dict(num_ctx=self.context_tokens, num_predict=self.max_tokens,
                         temperature=0.1, top_p=0.95, top_k=50, seed=self.seed))
        request = urllib.request.Request(self.base_url + '/api/chat',
            data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f'Ollama observation request failed: {exc}') from exc
        _save(directory / 'ollama_response.json', result)
        if result.get('done_reason') == 'length':
            raise ValueError('Response hit the token limit; increase --max-tokens')
        return result.get('message', {}).get('content', '')

    def _ask(self, prompt, images, schema, validator, directory):
        directory.mkdir()
        _save(directory / 'schema.json', schema)
        _save(directory / 'images.json', [str(p) for p in images])
        (directory / 'system.txt').write_text(SYSTEM, encoding='utf-8')
        original = prompt
        for attempt in range(self.retries + 1):
            current = directory / f'attempt_{attempt + 1}'
            current.mkdir()
            (current / 'prompt.txt').write_text(prompt, encoding='utf-8')
            try:
                raw = self._request(prompt, images, schema, current)
                (current / 'response.txt').write_text(raw, encoding='utf-8')
                return validator(json.loads(raw))
            except ValueError as exc:
                (current / 'error.txt').write_text(str(exc), encoding='utf-8')
                if attempt == self.retries:
                    raise ValueError(f'{directory.name} rejected: {exc}') from exc
                prompt = original + '\n\nThe previous response was invalid: ' + str(exc) + '\nReturn a complete corrected JSON object. Use unknown for uncertain attributes.'

    def observe(self, video, output_dir, *, frames=24, max_assets=8, max_side=768, progress=True):
        if (type(frames) is not int or not 2 <= frames <= 128
                or type(max_assets) is not int or not 1 <= max_assets <= 32
                or type(max_side) is not int or not 128 <= max_side <= 2048):
            raise ValueError('Use 2–128 frames, 1–32 assets, and max_side 128–2048')
        video = Path(video).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        timestamps = frame_timestamps(video)
        duration = _duration(video, timestamps)
        indices = sample_indices(timestamps, frames)
        times = [timestamps[i] for i in indices]
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=False)
        source_frames = extract_frames(video, indices, directory / 'source_frames')
        from PIL import Image
        images = []
        image_dir = directory / 'frames'
        image_dir.mkdir()
        for i, source in enumerate(source_frames, 1):
            path = image_dir / f'{i:03d}.jpg'
            with Image.open(source) as frame:
                frame.thumbnail((max_side, max_side))
                frame.convert('RGB').save(path, quality=90)
            source.unlink()
            images.append(path)
        (directory / 'source_frames').rmdir()
        metadata = dict(video=str(video), duration_seconds=duration, backend=self.backend, model=self.model,
            sampling='uniform in presentation time; visual frames only; no audio',
            shot_boundaries='estimated at midpoints between sampled frames',
            frames=[dict(id=i, source_index=index, seconds=t, path=str(p))
                    for i, (index, t, p) in enumerate(zip(indices, times, images), 1)])
        _save(directory / 'sampling.json', metadata)
        if progress:
            print(f'Observing {len(images)} frames from {duration:.3f}s of video...', file=sys.stderr, flush=True)
        inventory = self._ask(inventory_prompt(times, max_assets), images,
            inventory_schema(len(images), max_assets),
            lambda v: validate_inventory(v, len(images), max_assets), directory / 'inventory')
        _save(directory / 'inventory.json', inventory)
        assets = []
        for index, asset in enumerate(inventory['assets'], 1):
            if progress:
                print(f"Attributes {index}/{len(inventory['assets'])}: {asset['name']} ({asset['kind']})", file=sys.stderr, flush=True)
            frame_ids = sorted(asset['frame_ids'])
            def check(attrs):
                validate_attributes(asset, attrs)
                return attrs
            attrs = self._ask(attributes_prompt(asset, [times[f-1] for f in frame_ids]),
                [images[f-1] for f in frame_ids], attribute_schema(asset['kind']), check,
                directory / f'asset_{index:02d}')
            proposed = attrs
            attrs = self._ask(review_prompt(asset, proposed), [images[f-1] for f in frame_ids],
                review_schema(proposed), lambda v: validate_review(asset, proposed, v),
                directory / f'asset_{index:02d}_review')
            assets.append(dict(id=asset['id'], kind=asset['kind'], name=asset['name'],
                description=asset['name'], frames=[times[f-1] for f in frame_ids], attributes=attrs))
        # The model selects frame numbers. Python supplies paths, version, ids,
        # timestamps, playback order and an empty production-constraint list.
        boundaries = [0.0] + [(times[s['first_frame']-2] + times[s['first_frame']-1]) / 2
                             for s in inventory['shots'][1:]] + [duration]
        shots = [dict(id=f'S{i+1}', start=boundaries[i], end=boundaries[i+1],
                      description=s['description'], assets=s['assets'])
                 for i, s in enumerate(inventory['shots'])]
        result = dict(video=str(video), attribute_schema_version=CATALOG['schema_version'],
                      assets=assets, shots=shots, target_order=[s['id'] for s in shots], constraints=[])
        validate_observation(deepcopy(result))
        _save(directory / 'observation.json', result)
        return result


def observe_video(video, output_dir, *, qwen=None, backend=None, model=None,
                  frames=24, max_assets=8, max_side=768, progress=True, **options):
    with VideoObserver(qwen=qwen, backend=backend, model=model, **options) as observer:
        return observer.observe(video, output_dir, frames=frames, max_assets=max_assets,
                                max_side=max_side, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwrite an existing run')
    parser.add_argument('--backend', choices=('ollama', 'local'), default='ollama')
    parser.add_argument('--model', help=f'Default Ollama model: {DEFAULT_OLLAMA_MODEL}')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--max-assets', type=int, default=8)
    parser.add_argument('--max-side', type=int, default=768)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--context-tokens', type=int, default=32768)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--retries', type=int, default=1, help='Extra attempts after invalid JSON/attributes')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        result = observe_video(args.video, args.output_dir, backend=args.backend, model=args.model,
            base_url=args.base_url, frames=args.frames, max_assets=args.max_assets, max_side=args.max_side,
            max_tokens=args.max_tokens, context_tokens=args.context_tokens, timeout=args.timeout,
            seed=args.seed, retries=args.retries, progress=not args.quiet)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Video observation failed: {exc}\n')
    print(json.dumps(dict(observation=str(args.output_dir.resolve() / 'observation.json'),
                          assets=len(result['assets']), shots=len(result['shots'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
