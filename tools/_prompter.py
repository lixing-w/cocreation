"""Validation, label mapping and owned/shared Qwen lifecycle for prompt writers."""
import json
import math
from pathlib import Path
import re
import sys

from cocreation import Reference
from cocreation.storage import scratch_root
from tools.extract_attribute_nodes_prompts import CATEGORY_RULES
from tools._prompter_prompts import labeled_payload

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


class PromptValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def parse_prompt(response, catalog, target, seconds=None, *, diagnostic_path=None):
    """Return text/None; report the precise rejection and raw response on stderr.

    Optional JSONL diagnostics are for one run/attempt, not model-generated data.
    A model's explicit null is distinguished from malformed/invalid model output.
    """
    def require(condition, code, message):
        if not condition:
            raise PromptValidationError(code, message)

    raw = response.get('text') if isinstance(response, dict) else None
    try:
        require(isinstance(raw, str), 'missing_text', 'Model response has no string text field')
        text = raw.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        try:
            result = json.loads(text, object_pairs_hook=_object)
        except (ValueError, RecursionError) as error:
            raise PromptValidationError('invalid_json', str(error)) from error
        require(isinstance(result, dict), 'invalid_envelope', 'Expected a JSON object with prompt')
        if result.get('prompt', 'missing') is None and set(result) <= {'prompt', 'reason'}:
            reason = result.get('reason')
            detail = reason if isinstance(reason, str) and reason.strip() else 'No reason supplied by model'
            raise PromptValidationError('model_declined', detail)
        require(set(result) == {'prompt'}, 'invalid_envelope',
                f'Expected only prompt; received keys {sorted(result)}')
        prompt = result['prompt']
        require(isinstance(prompt, str) and bool(prompt.strip()), 'empty_prompt', 'Expected nonempty prompt string')
        prompt = prompt.strip()
        require(not any(token in prompt for token in ('```', '{{artifact:', '<|')),
                'forbidden_markup', 'Prompt contains code fences, unresolved artifact placeholders or model tokens')
        expected = {row['generator_label'] for row in catalog}
        subjects, found = set(), set()
        for match in LABEL.finditer(prompt):
            label = match.group()
            if re.fullmatch(r'<Subject [1-9]\d*>', label) and target == 'h3' and catalog:
                subjects.add(label)
            elif label in expected:
                found.add(label)
            else:
                raise PromptValidationError('unexpected_label',
                    f'Unexpected label {label!r}; expected media labels {sorted(expected)}')
        require(found == expected, 'missing_labels', f'Missing media labels: {sorted(expected - found)}')
        if target == 'flux':
            require(len(prompt.split()) <= 200, 'flux_word_limit',
                    f'Flux prompt has {len(prompt.split())} words; maximum is 200')
            return prompt
        sections = REF_SECTIONS if catalog else BASE_SECTIONS
        headings = list(re.finditer(r'(?m)^('+'|'.join(REF_SECTIONS+BASE_SECTIONS)+r'):\s*', prompt))
        actual = tuple(m[1] for m in headings)
        require(actual == sections and bool(headings) and headings[0].start() == 0,
                'h3_sections', f'Expected headings in order {sections}, starting at character 0; found {actual}')
        bodies = {m[1]:prompt[m.end():headings[i+1].start() if i+1<len(headings) else len(prompt)].strip()
                  for i,m in enumerate(headings)}
        require(all(bodies.values()), 'empty_sections',
                f'Empty H3 sections: {[key for key, body in bodies.items() if not body]}')
        if subjects:
            defined = re.findall(r'(?m)^(<Subject [1-9]\d*>)\s+', bodies['subject_definitions'])
            require(set(defined) == subjects and len(defined) == len(subjects), 'subject_definitions',
                    f'Each subject needs one own-line definition; used {sorted(subjects)}, defined {defined}')
        body = bodies['detailed_description' if catalog else 'integrated_multimodal_description']
        shots = list(re.finditer(r'\[Shot (\d+)\]',body))
        numbers = [int(m[1]) for m in shots]
        require(bool(shots) and numbers == list(range(1,len(shots)+1)), 'shot_numbers',
                f'Shots must start at 1 and be consecutive; found {numbers}')
        previous = 0.0
        for i,shot in enumerate(shots):
            suffix = body[shot.end():]
            if i == 0:
                require(not re.match(r'\s+At\s+\d',suffix), 'first_shot_timestamp',
                        '[Shot 1] must not have an At timestamp')
                continue
            stamp = re.match(r' At (\d{2}):(\d{2})\.(\d{3}),',suffix)
            require(stamp is not None and int(stamp[2]) < 60, 'shot_timestamp_format',
                    f'[Shot {i+1}] must be followed by At MM:SS.mmm,; got {suffix[:70]!r}')
            time = 60*int(stamp[1])+int(stamp[2])+int(stamp[3])/1000
            require(seconds is not None and previous < time < seconds, 'shot_timestamp_range',
                    f'[Shot {i+1}] timestamp {time}s must be after {previous}s and before {seconds}s')
            previous = time
        return prompt
    except (PromptValidationError, ValueError, TypeError, KeyError, RecursionError) as error:
        diagnostic = dict(tool=target+'_prompter', code=getattr(error, 'code', 'invalid_response'),
            message=str(error), expected_labels=[r['generator_label'] for r in catalog],
            seconds=seconds, raw_text=raw)
        if isinstance(response, dict):
            for key in ('job_dir', 'request_path'):
                if isinstance(response.get(key), str):
                    diagnostic[key] = response[key]
        print('['+target+'_prompter] rejected response:\n'+json.dumps(diagnostic, ensure_ascii=False, indent=2),
              file=sys.stderr, flush=True)
        if diagnostic_path is not None:
            path = Path(diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(diagnostic, ensure_ascii=False)+'\n')
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
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'input.json').write_text(json.dumps(
            labeled_payload(brief, rows, catalog, seconds=seconds if target == 'h3' else None),
            ensure_ascii=False, indent=2) + '\n')
    owned = qwen is None
    model = factory(output_dir=output_dir) if owned else qwen
    try:
        response = model(user, refs, system_prompt=system,
                         system_cache_path=(scratch_root()/f'kvcache/{target}_prompter/system_prompt.kvcache'
                                            if use_system_cache else None),
                         max_new_tokens=max_new_tokens, video_fps=video_fps,
                         video_max_frames=video_max_frames, max_pixels=max_pixels, output_dir=output_dir)
        return parse_prompt(response, catalog, target, seconds,
                            diagnostic_path=Path(output_dir)/'diagnostics.jsonl' if output_dir is not None else None)
    finally:
        if owned:
            model.close()
