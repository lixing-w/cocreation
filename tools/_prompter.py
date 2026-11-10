"""Validation, label mapping and owned/shared Qwen lifecycle for prompt writers."""
import json
import math
from pathlib import Path
import re

from cocreation import Reference
from cocreation.storage import scratch_root
from tools.extract_attribute_nodes_prompts import CATEGORY_RULES

LABEL = re.compile(r'<(Image|Picture|Video|Audio|Subject)\s+([^<>]+)>', re.I)
REF_SECTIONS = ('subject_definitions','summary','retention_analysis','detailed_description',
                'overall_soundscape','non_diegetic_music')
BASE_SECTIONS = ('integrated_multimodal_description','overall_soundscape','non_diegetic_music')


def prepare_inputs(attributes, references, target):
    """Validate selected attribute records and local media; return rows, refs and map.

    attributes: List of flat mappings with GLOBAL id/category/text, optional artifact_ids
        (source artifact IDs), plus optional parent_id/links/time_range context.
    references: Ordered iterable of Reference or artifact_id/path/kind mappings.
    target: flux or h3; determines generator labels (not Qwen observation labels).
    """
    if not isinstance(attributes, list):
        raise ValueError('attributes must be a list of flat attribute JSON objects')
    refs, catalog, seen = [], [], set()
    counters = {'image':0, 'video':0}
    for raw in references or []:
        ref = Reference(raw['artifact_id'], raw['path'], raw['kind']) if isinstance(raw,dict) else raw
        if not isinstance(ref,Reference) or ref.kind not in counters:
            raise ValueError('Prompters observe image/video references only')
        if ref.artifact_id in seen:
            raise ValueError('Duplicate artifact reference: ' + ref.artifact_id)
        path = Path(ref.path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError('Reference must be a local file')
        refs.append(Reference(ref.artifact_id, path, ref.kind)); seen.add(ref.artifact_id)
        counters[ref.kind] += 1
        number = counters[ref.kind]
        observed = f'<{ref.kind.title()} {number}>'
        generated = (f'<Image {len(refs)}>' if target == 'flux' else
                     f'<{"Picture" if ref.kind == "image" else "Video"} {number}>')
        catalog.append(dict(artifact_id=ref.artifact_id, kind=ref.kind,
                            observation_label=observed, generator_label=generated,
                            requires_frame=(target == 'flux' and ref.kind == 'video')))
    if target == 'h3' and len(refs) > 32:
        raise ValueError('VDNH3 supports at most 32 references')
    if target == 'flux' and len(refs) > 4:
        raise ValueError('FLUX.2-klein prompt contract supports at most four image slots')
    rows, ids, used = [], set(), set()
    for raw in attributes:
        if not isinstance(raw,dict) or not {'id','category','text'} <= set(raw):
            raise ValueError('Each attribute requires id, category, text')
        if set(raw) - {'id','category','text','artifact_ids','parent_id','links','time_range'}:
            raise ValueError('Unsupported attribute fields')
        key, category, text = raw['id'], raw['category'], raw['text']
        if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+_[A-Za-z0-9_.:-]+',key):
            raise ValueError('Attribute id must be global: origin_artifact_id_local_id (or new_local_id)')
        if key in ids:
            raise ValueError('Duplicate global attribute ID: ' + key)
        for field in ('parent_id', 'links'):
            values = [raw[field]] if field == 'parent_id' and field in raw else raw.get(field, [])
            if not isinstance(values,list) or any(not isinstance(value,str) or not re.fullmatch(
                    r'[A-Za-z0-9_.:-]+_[A-Za-z0-9_.:-]+', value) for value in values):
                raise ValueError(field + ' must use global node IDs, not extractor-local IDs')
        if not isinstance(category,str) or category not in CATEGORY_RULES:
            raise ValueError('Unknown CPG category')
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Attribute text must be nonempty')
        sources = raw.get('artifact_ids', [])
        if not isinstance(sources,list) or any(not isinstance(x,str) for x in sources):
            raise ValueError('artifact_ids must be a list of source IDs')
        if len(set(sources)) != len(sources) or not set(sources) <= seen:
            raise ValueError('Duplicate or unattached attribute source')
        row = dict(raw, artifact_ids=list(sources))
        # Round-trip ensures serializable, finite task data without mutating callers.
        row = json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False))
        rows.append(row); ids.add(key); used.update(sources)
    if not rows:
        raise ValueError('At least one selected attribute is required')
    if used != seen:
        raise ValueError('Every supplied media file must be linked by an input attribute')
    return rows, refs, catalog


def _object(pairs):
    """Reject duplicate keys when decoding a model's JSON envelope."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_prompt(response, catalog, target, seconds=None):
    """Return valid expert text or None; validate source labels and H3 structure."""
    try:
        text = response.get('text') if isinstance(response,dict) else None
        if not isinstance(text,str):
            return None
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        result = json.loads(text, object_pairs_hook=_object)
        if not isinstance(result,dict) or set(result) != {'prompt'}:
            return None
        prompt = result['prompt']
        if not isinstance(prompt,str) or not prompt.strip():
            return None
        prompt = prompt.strip()
        if '```' in prompt or '{{artifact:' in prompt or '<|' in prompt:
            return None
        expected = {row['generator_label'] for row in catalog}
        matches = list(LABEL.finditer(prompt))
        subjects = set()
        found = set()
        for match in matches:
            label = match.group()
            if re.fullmatch(r'<Subject [1-9]\d*>', label) and target == 'h3' and catalog:
                subjects.add(label)
            elif label in expected:
                found.add(label)
            else:
                return None
        if found != expected:
            return None
        if target == 'flux':
            return prompt if len(prompt.split()) <= 200 else None
        sections = REF_SECTIONS if catalog else BASE_SECTIONS
        headings = list(re.finditer(r'(?m)^('+'|'.join(REF_SECTIONS+BASE_SECTIONS)+r'):\s*', prompt))
        if tuple(m[1] for m in headings) != sections or headings[0].start() != 0:
            return None
        bodies = {m[1]:prompt[m.end():headings[i+1].start() if i+1<len(headings) else len(prompt)].strip()
                  for i,m in enumerate(headings)}
        if not all(bodies.values()):
            return None
        if subjects:
            defined = re.findall(r'(?m)^(<Subject [1-9]\d*>)\s+', bodies['subject_definitions'])
            if set(defined) != subjects or len(defined) != len(subjects):
                return None
        body = bodies['detailed_description' if catalog else 'integrated_multimodal_description']
        shots = list(re.finditer(r'\[Shot (\d+)\]',body))
        if not shots or [int(m[1]) for m in shots] != list(range(1,len(shots)+1)):
            return None
        previous = 0.0
        for i,shot in enumerate(shots):
            suffix = body[shot.end():]
            if i == 0:
                if re.match(r'\s+At\s+\d',suffix):
                    return None
                continue
            stamp = re.match(r' At (\d{2}):(\d{2})\.(\d{3}),',suffix)
            if stamp is None or int(stamp[2]) >= 60:
                return None
            time = 60*int(stamp[1])+int(stamp[2])+int(stamp[3])/1000
            if not previous < time < seconds:
                return None
            previous = time
        return prompt
    except (ValueError,TypeError,KeyError,RecursionError):
        return None


def run_prompter(target, attributes, references, brief, *, qwen, factory,
                 system_builder, user_builder, output_dir, use_system_cache,
                 max_new_tokens, video_fps, video_max_frames, max_pixels, seconds=None):
    """Validate task, invoke a supplied or owned Qwen, and return expert prompt text.

    Builders supply tool-specific prompts; factory creates only an owned instance.
    Remaining keyword arguments configure Qwen budgets, logs and dedicated cache.
    """
    if not isinstance(brief,str):
        raise TypeError('user_prompt must be text')
    if target == 'h3' and (type(seconds) not in (int,float) or not math.isfinite(seconds) or not 5<=seconds<=15):
        raise ValueError('VDNH3 seconds must be between 5 and 15')
    rows, refs, catalog = prepare_inputs(attributes, references, target)
    system = system_builder()
    user = user_builder(brief, rows, catalog, seconds) if target == 'h3' else user_builder(brief, rows, catalog)
    owned = qwen is None
    model = factory(output_dir=output_dir) if owned else qwen
    try:
        response = model(user, refs, system_prompt=system,
                         system_cache_path=(scratch_root()/f'kvcache/{target}_prompter/system_prompt.kvcache'
                                            if use_system_cache else None),
                         max_new_tokens=max_new_tokens, video_fps=video_fps,
                         video_max_frames=video_max_frames, max_pixels=max_pixels)
        return parse_prompt(response, catalog, target, seconds)
    finally:
        if owned:
            model.close()
