"""Two independent VLM passes: revise observations, then verify selected inheritance."""
import json
import re
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.extract_attribute_nodes import validate_attributes, repair_attributes
from tools._attribute_protocol import atomize, compact_rows, validate_matches
from tools._attribute_vlm import ask_json, reference

from tools.attribute_verifier_prompts import REVISE, INHERIT



def inheritance_payload(generated, observed, sources, references):
    """VLM-local media labels/IDs, plus exact reverse atom-ID maps; never mutate inputs."""
    counts, labels = {'image': 0, 'video': 0}, {}
    for ref in references:
        counts[ref.kind] += 1
        labels[ref.artifact_id] = f'<{ref.kind.title()} {counts[ref.kind]}>'
    # Longest first + token boundary avoids artifact_2 corrupting artifact_20.
    pattern = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(
        re.escape(uid) for uid in sorted(labels, key=len, reverse=True)) + r')(?![A-Za-z0-9])')
    def convert(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: labels[match.group()], value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value
    selected = [record for source in sources for record in source['attributes']]
    maps = {name: {convert(record['id']): record['id'] for record in records}
            for name, records in (('observed', observed), ('selected', selected))}
    if len(maps['observed']) != len(observed) or len(maps['selected']) != len(selected):
        raise ValueError('Temporary verifier atom ID collision')
    payload = convert(dict(generated_artifact_id=generated.artifact_id,
                           observed=compact_rows(observed), selected_sources=sources))
    return payload, maps


def restore_matches(matches, maps):
    """Accept only supplied temporary IDs, restoring original graph IDs before validation."""
    if not isinstance(matches, list):
        raise ValueError('Expected matches array')
    result = []
    for match in matches:
        if not isinstance(match, dict) or set(match) != {'observed_id', 'source_id'}:
            raise ValueError('Match needs exactly observed_id/source_id')
        new, old = match['observed_id'], match['source_id']
        if (not isinstance(new, str) or not isinstance(old, str) or
                new not in maps['observed'] or old not in maps['selected']):
            raise ValueError('Match references an unknown temporary attribute ID')
        result.append(dict(observed_id=maps['observed'][new], source_id=maps['selected'][old]))
    return result


def attribute_verifier(*, mode, generated_media, target_attributes=None,
                       observed_attributes=None, selected_sources=None, qwen=None,
                       output_dir=None, max_new_tokens=4096, video_fps=2.0,
                       video_max_frames=32, max_pixels=512*512):
    """Return an observation (pass 1), matches list (pass 2), or None.

    revise_observation: generated_media + target_attributes only.
    verify_inheritance: generated_media + observed_attributes + selected_sources;
      each source is {reference: Reference, attributes: selected ORIGINAL atom records}.
    Pass 2 temporarily substitutes media labels into all supplied IDs, then restores
    returned pairs to original IDs. It compares conceptual/visual attribute equivalence
    and never edits observations or provenance. Input/operational errors
    raise; pass 1 repairs harmless omissions/wrapping before strict validation.
    Unrecoverable output returns None; pass 2 matches stay strict. No automatic retry.
    """
    generated = reference(generated_media)
    options = dict(qwen=qwen, output_dir=output_dir, max_new_tokens=max_new_tokens,
                   video_fps=video_fps, video_max_frames=video_max_frames, max_pixels=max_pixels)
    if mode == 'revise_observation':
        if observed_attributes is not None or selected_sources is not None:
            raise ValueError('Revision accepts only new media and target attributes')
        validate_attributes(target_attributes, generated.artifact_id, generated.kind)
        result = ask_json(REVISE, {'target': target_attributes}, [generated], **options)
        if result is None:
            return None
        try:
            return validate_attributes(repair_attributes(result), generated.artifact_id, generated.kind)
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            print(f'attribute_verifier revision: {error}', file=sys.stderr)
            return None
    if mode != 'verify_inheritance' or target_attributes is not None:
        raise ValueError('Expected revise_observation or verify_inheritance mode')
    validate_attributes(observed_attributes, generated.artifact_id, generated.kind)
    if not isinstance(selected_sources, list):
        raise ValueError('Inheritance requires selected_sources list')
    observed = atomize(observed_attributes)
    refs, selected, sources = [generated], [], []
    for source in selected_sources:
        if not isinstance(source, dict) or set(source) != {'reference', 'attributes'}:
            raise ValueError('Source needs reference and selected attributes')
        ref = reference(source['reference'])
        records = source['attributes']
        if not isinstance(records, list) or not records or any(r['carrier_artifact_id'] != ref.artifact_id for r in records):
            raise ValueError('Source records must belong to the attached carrier')
        refs.append(ref)
        selected.extend(records)
        sources.append(dict(artifact_id=ref.artifact_id, attributes=compact_rows(records)))
    if len({r.artifact_id for r in refs}) != len(refs) or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate source or attribute IDs')
    if not observed or not selected:
        return []
    payload, maps = inheritance_payload(generated, observed, sources, refs)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'input.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
        (directory / 'id_map.json').write_text(json.dumps(maps, ensure_ascii=False, indent=2)+'\n')
    result = ask_json(INHERIT, payload, refs, **options)
    if result is None:
        return None
    try:
        if not isinstance(result, dict) or set(result) != {'matches'}:
            raise ValueError('Expected matches object')
        return validate_matches(restore_matches(result['matches'], maps), observed, selected)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        print(f'attribute_verifier inheritance: {error}', file=sys.stderr)
        return None
