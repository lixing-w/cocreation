"""Convert H3 structured prompts / OpenVid captions to observation.json, without media.

This describes intended content, not verified video content. See docs/prompt_observer.md.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dialogue_generation.asset_variations import CATALOG, CATEGORIES, validate_attributes, register_attributes
from scripts.dialogue_generation.planner import observation as validate_observation
from tools.attribute_authority import name_assets, neutral_asset_descriptions
from tools.observation_descriptions import separate_descriptions, DESCRIPTION_POLICY_VERSION
from tools.dialogue_models import BACKENDS, DEFAULT_BACKEND, SONNET_EFFORTS
from tools.video_observer import VideoObserver, _object, _save, repair_attributes
from tools.prompt_observer_prompts import SYSTEM, inventory_prompt, attributes_prompt, review_prompt

_HEADINGS = re.compile(r'(?m)^(' + '|'.join((
    'subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
    'integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music')) + r'):\s*')
_SHOT = re.compile(r'\[Shot\s+(\d+)\]')
_TIME = re.compile(r'^\s*At\s+(\d+):([0-5]\d)\.(\d{3})\s*,?\s*')


def load_record(path, row=0):
    """Select a zero-based row from JSONL / CSV / a JSON array, or read one text/JSON record."""
    path = Path(path)
    if type(row) is not int or row < 0:
        raise ValueError('row must be a nonnegative integer')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        if path.suffix.lower() == '.csv':
            records = csv.DictReader(handle)
        elif path.suffix.lower() == '.jsonl':
            records = (json.loads(line) for line in handle if line.strip())
        elif path.suffix.lower() == '.json':
            value = json.load(handle)
            records = iter(value if isinstance(value, list) else [value])
        else:
            records = iter([handle.read()])
        for index, record in enumerate(records):
            if index == row:
                if not isinstance(record, (str, dict)):
                    raise ValueError('Expected prompt text or a JSON object')
                return record
    raise ValueError(f'No row {row} in {path}')


def _positive(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label} must be a positive finite number') from exc
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError(f'{label} must be a positive finite number')
    return result


def prepare_source(record, *, source_format='auto', duration=None, fps=None, width=None, height=None):
    """Parse only explicit H3 target shots. Captions never imply a cut on their own."""
    if source_format not in ('auto', 'h3', 'openvid'):
        raise ValueError('source_format must be auto, h3 or openvid')
    row = record if isinstance(record, dict) else {}
    text = record if isinstance(record, str) else row.get('prompt', row.get('caption'))
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Source must contain nonempty prompt or caption text')
    headings = list(_HEADINGS.finditer(text))
    sections = {m[1]: text[m.end():headings[i+1].start() if i+1 < len(headings) else len(text)].strip()
                for i, m in enumerate(headings)}
    fmt = source_format if source_format != 'auto' else (
        'h3' if 'detailed_description' in sections or 'integrated_multimodal_description' in sections else 'openvid')
    visual = (sections.get('detailed_description') or sections.get('integrated_multimodal_description')
              if fmt == 'h3' else text.strip())
    if not visual:
        raise ValueError('H3 requires detailed_description or integrated_multimodal_description')
    duration_basis = 'override' if duration is not None else 'metadata'
    if duration is None:
        duration = row.get('duration_seconds', row.get('seconds', row.get('duration')))
        if duration is None or duration == '':
            count = row.get('num_frames', row.get('frame'))
            rate = fps if fps is not None else row.get('fps')
            if count is None or rate is None:
                raise ValueError('Supply --duration, seconds/duration_seconds metadata, or frame count and explicit fps')
            duration = _positive(count, 'frame count') / _positive(rate, 'fps')
            duration_basis = 'frame_count/fps'
    duration = _positive(duration, 'duration')
    markers = list(_SHOT.finditer(visual)) if fmt == 'h3' else []
    common = visual[:markers[0].start()].strip() if markers else ''
    if markers:
        if [int(m[1]) for m in markers] != list(range(1, len(markers) + 1)):
            raise ValueError('H3 shot markers must appear once, numbered consecutively from 1')
        if len(markers) > 4:
            raise ValueError('The downstream contract supports at most four shots; split this source explicitly')
        starts, descriptions = [], []
        for i, marker in enumerate(markers):
            body = visual[marker.end():markers[i+1].start() if i+1 < len(markers) else len(visual)].strip()
            match = _TIME.match(body)
            if i and match is None:
                raise ValueError(f'Shot {i+1} needs an explicit At MM:SS.mmm timestamp')
            start = (int(match[1])*60 + int(match[2]) + int(match[3])/1000) if match else 0.0
            starts.append(start)
            descriptions.append(body[match.end():].strip() if match else body)
        if starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])) or starts[-1] >= duration:
            raise ValueError('Shot starts must increase from zero and be less than duration')
    else:
        starts, descriptions = [0.0], [visual]
    if any(not text for text in descriptions):
        raise ValueError('Each shot needs source text')
    shots = [dict(number=i+1, start=start, end=(starts+[duration])[i+1], text=body)
             for i, (start, body) in enumerate(zip(starts, descriptions))]
    geometry = {}
    for key, override in (('width', width), ('height', height)):
        value = override if override is not None else row.get(key)
        if value is not None and value != '':
            number = _positive(value, key)
            if not number.is_integer():
                raise ValueError(f'{key} must be an integer')
            geometry[key] = int(number)
    return dict(format=fmt, text=text, visual_text=visual, common=common, shots=shots,
                duration=duration, duration_basis=duration_basis, geometry=geometry,
                source_id=str(row.get('video_path', row.get('video', ''))),
                sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())


def inventory_schema(shot_count, max_assets):
    text = dict(type='string', minLength=1)
    number = dict(type='integer', minimum=1, maximum=shot_count)
    asset = _object(dict(id=text, kind=dict(type='string', enum=list(CATEGORIES)), name=text,
                         description=text))
    shot = _object(dict(number=number, description=text,
                        assets=dict(type='array', minItems=1, maxItems=max_assets, items=text)))
    return _object(dict(assets=dict(type='array', minItems=1, maxItems=max_assets, items=asset),
                        shots=dict(type='array', minItems=shot_count, maxItems=shot_count, items=shot)))


def validate_inventory(value, source, max_assets):
    if not isinstance(value, dict) or set(value) != {'assets', 'shots'}:
        raise ValueError('Inventory needs assets and shots')
    assets, shots = value['assets'], value['shots']
    if not isinstance(assets, list) or not 1 <= len(assets) <= max_assets:
        raise ValueError(f'Expected 1–{max_assets} assets')
    ids, names = set(), set()
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {'id', 'kind', 'name', 'description'}:
            raise ValueError('Invalid inventory asset fields')
        if any(not isinstance(v, str) or not v.strip() for v in asset.values()):
            raise ValueError('Asset fields must be nonempty strings')
        key = (asset['kind'], asset['name'].casefold().strip())
        if asset['id'] in ids or key in names or asset['kind'] not in CATEGORIES:
            raise ValueError('Use distinct asset ids/names and a supported kind')
        ids.add(asset['id'])
        names.add(key)
    if not isinstance(shots, list) or len(shots) != len(source['shots']):
        raise ValueError('Keep exactly the parsed source shots')
    used = set()
    for i, shot in enumerate(shots, 1):
        if not isinstance(shot, dict) or set(shot) != {'number', 'description', 'assets'}:
            raise ValueError('Invalid shot fields')
        refs = shot['assets']
        if (type(shot['number']) is not int or shot['number'] != i
                or not isinstance(shot['description'], str) or not shot['description'].strip()
                or not isinstance(refs, list) or not refs
                or any(not isinstance(a, str) or a not in ids for a in refs) or len(refs) != len(set(refs))):
            raise ValueError('Keep source shot order; reference distinct known assets in each shot')
        used.update(refs)
    if used != ids:
        raise ValueError('Every asset must be used in a shot')
    return value


def attributes_schema(kind):
    """Plain field-to-value attributes, without supporting quotations."""
    props = {key: dict(type='string', minLength=1) for key in CATEGORIES[kind]['attributes']}
    if kind == 'person':
        props['age_group']['enum'] = CATEGORIES[kind]['attributes']['age_group']['values']
    return _object(props)


def parse_attributes(asset, value):
    attributes, _ = repair_attributes(asset, value)
    validate_attributes(asset, attributes)
    return attributes


class PromptObserver(VideoObserver):
    """Use the shared Qwen transport/retries, always with an empty media list."""
    system_prompt = SYSTEM

    def __init__(self, *, enable_thinking=True, max_tokens=8192, **options):
        super().__init__(enable_thinking=enable_thinking, max_tokens=max_tokens, **options)

    def observe(self, record, output_dir, *, source_format='auto', duration=None, fps=None,
                width=None, height=None, max_assets=8, progress=True):
        if type(max_assets) is not int or not 1 <= max_assets <= 32:
            raise ValueError('max_assets must be 1–32')
        source = prepare_source(record, source_format=source_format, duration=duration, fps=fps, width=width, height=height)
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=False)
        _save(directory / 'input.json', record)
        _save(directory / 'source.json', source)
        (directory / 'source.txt').write_text(source['text'], encoding='utf-8')
        if progress:
            print(f"Extracting {source['format']} text ({len(source['shots'])} shots); no media supplied", file=sys.stderr, flush=True)
        inventory = self._ask(inventory_prompt(source, max_assets), [],
            inventory_schema(len(source['shots']), max_assets),
            lambda v: validate_inventory(v, source, max_assets), directory / 'inventory')
        _save(directory / 'inventory.json', inventory)
        aliases=name_assets(inventory['assets'])
        assets = []
        for index, asset in enumerate(inventory['assets'], 1):
            relevant = [s for s, inv in zip(source['shots'], inventory['shots']) if asset['id'] in inv['assets']]
            # Read the original target and relevant shots for this asset.
            context = '\n'.join([source['common']] + [s['text'] for s in relevant])
            if progress:
                print(f"Attributes {index}/{len(inventory['assets'])}: {asset['name']}", file=sys.stderr, flush=True)
            schema = attributes_schema(asset['kind'])
            # Candidate vocabulary must not leak into planning before review.
            result = self._ask(attributes_prompt(asset, context), [], schema,
                lambda v: parse_attributes(deepcopy(asset), v),
                directory / f'asset_{index:02d}')
            _save(directory / f'asset_{index:02d}' / 'attributes.json', result)
            reviewed = self._ask(review_prompt(asset, result, context), [], schema,
                lambda v: parse_attributes(deepcopy(asset), v),
                directory / f'asset_{index:02d}_review')
            _save(directory / f'asset_{index:02d}_review' / 'attributes.json', reviewed)
            result = reviewed
            # One accepted value per field, scoped to this identity/run.
            register_attributes(asset, result)
            validate_attributes(asset, result)
            assets.append(dict(id=asset['id'], kind=asset['kind'], name=asset['name'],
                description=asset['description'], frames=[round((s['start']+s['end'])/2, 6) for s in relevant[:2]],
                attributes=result,
                **({'attribute_extensions': deepcopy(asset['attribute_extensions'])} if asset.get('attribute_extensions') else {})))
        shots = [dict(id=f'S{s["number"]}', start=s['start'], end=s['end'],
                      description=inv['description'], assets=inv['assets'])
                 for s, inv in zip(source['shots'], inventory['shots'])]
        neutral_asset_descriptions(assets,shots,aliases)
        separate_descriptions(self, assets, shots, directory / "description_separation",
            source_context=dict(common_visual_context=source["common"],shots=source["shots"]))
        result = dict(video='prompt://' + source['sha256'],
            video_metadata=dict(duration=source['duration'], **source['geometry']),
            observation_source=dict(type='prompt', format=source['format'], source_id=source['source_id'],
                sha256=source['sha256'], video_verified=False, frames_are_nominal=True,
                timing_basis='explicit_prompt' if source['format']=='h3' and len(shots)>1 else 'whole_caption',
                duration_basis=source['duration_basis']),
            attribute_schema_version=CATALOG['schema_version'], assets=assets, shots=shots,
            target_order=[s['id'] for s in shots], constraints=[])
        result["description_policy"] = DESCRIPTION_POLICY_VERSION
        validate_observation(deepcopy(result))
        _save(directory / 'observation.json', result)
        return result


def observe_prompt(record, output_dir, *, qwen=None, backend=None, model=None, source_format='auto',
                   duration=None, fps=None, width=None, height=None, max_assets=8, progress=True, **options):
    with PromptObserver(qwen=qwen, backend=backend, model=model, **options) as observer:
        return observer.observe(record, output_dir, source_format=source_format, duration=duration,
            fps=fps, width=width, height=height, max_assets=max_assets, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--row', type=int, default=0, help='Zero-based JSONL/CSV/JSON-array row')
    parser.add_argument('--source-format', choices=('auto', 'h3', 'openvid'), default='auto')
    parser.add_argument('--duration', type=float, help='Seconds; required without duration or frame/fps metadata')
    parser.add_argument('--fps', type=float, help='Explicit fps for frame-count metadata (H3 samples: 24)')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwrite an existing run')
    parser.add_argument('--backend', choices=BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument('--think-mode', choices=SONNET_EFFORTS, help='Sonnet reasoning effort (default: low)')
    parser.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-assets', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--context-tokens', type=int, default=32768)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        result = observe_prompt(load_record(args.source, args.row), args.output_dir,
            source_format=args.source_format, duration=args.duration, fps=args.fps, width=args.width, height=args.height,
            backend=args.backend, model=args.model, base_url=args.base_url, enable_thinking=args.enable_thinking,
            think_mode=args.think_mode,
            max_assets=args.max_assets, max_tokens=args.max_tokens, context_tokens=args.context_tokens,
            timeout=args.timeout, seed=args.seed, retries=args.retries, progress=not args.quiet)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Prompt observation failed: {exc}\n')
    print(json.dumps(dict(observation=str(args.output_dir.resolve() / 'observation.json'),
                          assets=len(result['assets']), shots=len(result['shots'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
