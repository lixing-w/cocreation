"""Validation, label mapping and owned/shared Qwen lifecycle for prompt writers."""
import json
import math
from pathlib import Path
import re
import sys

from cocreation.storage import scratch_root
from tools._attribute_protocol import media_catalog, validate_sources, SOURCE
from tools._attribute_vlm import reference


LABEL = re.compile(r'<(Image|Picture|Video|Audio|Subject)\s+([^<>]+)>', re.I)
REF_SECTIONS = ('subject_definitions','summary','retention_analysis','detailed_description',
                'overall_soundscape','non_diegetic_music')
BASE_SECTIONS = ('integrated_multimodal_description','overall_soundscape','non_diegetic_music')


def prepare_inputs(description, references, target):
    """Validate prose and construct observation-to-generator labels in source order."""
    if target not in ('flux', 'h3'):
        raise ValueError('Expected flux or h3')
    refs = [reference(r) for r in references or []]
    labels = media_catalog(refs)
    description = validate_sources(description, refs)
    if set(SOURCE.findall(description)) != set(labels.values()):
        raise ValueError('Every supplied reference must appear in the description')
    if len(refs) > (4 if target == 'flux' else 32):
        raise ValueError('Too many generation references')
    catalog = []
    for index, ref in enumerate(refs, 1):
        observed = labels[ref.artifact_id]
        generated = f'<Image {index}>' if target == 'flux' else observed.replace('<Image ', '<Picture ')
        catalog.append(dict(artifact_id=ref.artifact_id, kind=ref.kind,
                            observation_label=observed, generator_label=generated))
    return description, refs, catalog


class PromptValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def parse_prompt(response, catalog, target, seconds=None, *, diagnostic_path=None):
    """Validate direct prompt text; report the rejection and raw response on stderr.

    JSONL diagnostics are local bookkeeping, never model input or model output.
    """
    def require(condition, code, message):
        if not condition:
            raise PromptValidationError(code, message)

    raw = response.get('text') if isinstance(response, dict) else None
    try:
        require(isinstance(raw, str), 'missing_text', 'Model response has no string text field')
        prompt = raw.strip()
        if target == 'h3':
            # Repair Qwen observation labels before validating H3 generator labels.
            # Only attached images have aliases; unknown source numbers still fail.
            aliases = {row['observation_label']: row['generator_label']
                       for row in catalog if row['kind'] == 'image'}
            prompt = re.sub(r'<Image [1-9]\d*>',
                            lambda match: aliases.get(match.group(), match.group()), prompt)
        require(prompt != 'CANNOT_FULFILL', 'model_declined', 'Model could not fulfill the description')
        require(bool(prompt), 'empty_prompt', 'Expected nonempty prompt text')
        require(not prompt.startswith(('{', '[{', '"')), 'invalid_envelope',
                'Expected direct prompt text, not JSON or a quoted string')
        require(not any(token in prompt for token in ('```', '{{artifact:', '<|')),
                'forbidden_markup', 'Prompt contains code fences, unresolved artifact placeholders or model tokens')
        expected = {row['observation_label' if target == 'flux' else 'generator_label'] for row in catalog}
        subjects, found = set(), set()
        for match in LABEL.finditer(prompt):
            label = match.group()
            if re.fullmatch(r'<Subject [1-9]\d*>', label) and target == 'h3' and catalog:
                subjects.add(label)
            elif label in expected:
                require(prompt[:match.start()].endswith('like in '), 'source_attachment',
                        f'Each media label must immediately follow like in: {label}')
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
            message=str(error), expected_labels=[r['observation_label' if target == 'flux' else 'generator_label'] for r in catalog],
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
        (directory / 'input.txt').write_text(user, encoding='utf-8')
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
