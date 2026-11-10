"""Seeded reverse CPG growth. Models are constructed once; null abandons an attempt."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cocreation import Qwen, Flux, VDNH3, VDNRuntime, Reference
from tools.extract_attribute_nodes import extract_attribute_nodes, validate_attributes
from tools._attribute_protocol import (
    atomize, assemble, sample_selection, apply_additions, expert_rows, compact_rows,
    attach_provenance, validate_matches, canonicalize_shared)
from tools.extract_attribute_nodes_prompts import CATEGORY_RULES
from tools.imaginer import imaginer
from tools.attribute_verifier import attribute_verifier, inheritance_payload
from tools.flux_prompter import flux_prompter
from tools._prompter import prepare_inputs
from tools._prompter_prompts import labeled_payload
from tools.h3_prompter import h3_prompter
from tools.find_video_frame import find_video_frame


def save(path, value):
    """Atomic local checkpoint; failed attempts never enter the graph."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def artifact_number(artifact_id):
    match = re.fullmatch(r'artifact_(\d+)', artifact_id)
    if not match:
        raise ValueError('Use numeric artifact IDs: artifact_0, artifact_1, ...')
    return int(match[1])


def read_roots(manifest):
    """Manifest paths are relative to the manifest, not the process directory."""
    manifest = Path(manifest).expanduser().resolve()
    roots = json.loads(manifest.read_text())
    if not isinstance(roots, list) or not roots:
        raise ValueError('Root manifest must be a nonempty array')
    seen = set()
    for root in roots:
        number = artifact_number(root['artifact_id'])
        if number in seen:
            raise ValueError('Duplicate numeric artifact ID')
        seen.add(number)
        path = Path(root['path']).expanduser()
        root['path'] = str((manifest.parent / path).resolve(strict=True))
        if not Path(root['path']).is_file() or root['kind'] not in ('image', 'video'):
            raise ValueError('Root must be a local image/video file')
        observation = root.get('observation')
        if isinstance(observation, str):
            observation = json.loads((manifest.parent / observation).read_text())
            root['observation'] = observation
        if observation is not None:
            validate_attributes(observation, root['artifact_id'], root['kind'])
    return roots


def run(manifest, output_dir, *, attempts=10, seed=0, keep=(3, 8), variate=(0, 2),
        max_parents=2, p_video=0.5, seconds=5.0, brief='', vdn_python=None, debug=True):
    """Run at most attempts trials, not attempts successful generations.

    K is the minimum retained count; K and V use inclusive random ranges.
    V counts new attributes and may exceed the retained count. Sampling is
    seeded, inference is not promised bitwise reproducible. Operational exceptions
    checkpoint and propagate; malformed/null tool outputs skip without retrying.
    Root extraction happens only when observation is absent. No historical search.
    """
    for name, pair in [('keep', keep), ('variate', variate)]:
        if len(pair) != 2 or any(type(x) is not int for x in pair) or not 0 <= pair[0] <= pair[1]:
            raise ValueError(f'{name} must be an inclusive nonnegative integer range')
    if keep[0] < 1 or type(attempts) is not int or attempts < 0:
        raise ValueError('keep must be positive and attempts nonnegative')
    if type(max_parents) is not int or not 1 <= max_parents <= 32:
        raise ValueError('max_parents must be 1–32')
    if not 0 <= p_video <= 1 or not 5 <= seconds <= 15:
        raise ValueError('p_video must be 0–1; seconds must be 5–15')
    roots = read_roots(manifest)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    def emit(title, payload):
        # Console output is stderr; stdout remains the final machine-readable summary.
        block = '\n=== ' + title + ' ===\n' + json.dumps(payload, ensure_ascii=False, indent=2)
        with (output / 'debug.log').open('a', encoding='utf-8') as stream:
            stream.write(block + '\n')
        if debug:
            print(block, file=sys.stderr, flush=True)

    def media_rows(references):
        return [dict(artifact_id=r.artifact_id, kind=r.kind, path=str(r.path)) for r in references]

    rng = random.Random(seed)
    graph = {'version': 1, 'seed': seed, 'config': dict(attempts=attempts, keep=list(keep),
        variate=list(variate), max_parents=max_parents, p_video=p_video, seconds=seconds,
        brief=brief), 'artifacts': [], 'attempts': []}
    next_number = max(artifact_number(r['artifact_id']) for r in roots) + 1
    save(output / 'graph.json', graph)
    emit('Run configuration', graph['config'])
    emit('Initializing shared Qwen / Flux / VDNH3', {'output_dir': str(output)})
    # These exact three reusable objects serve roots and every trial.
    with ExitStack() as stack:
        qwen = stack.enter_context(Qwen(output_dir=output / 'qwen'))
        flux = stack.enter_context(Flux(output_dir=output / 'flux'))
        vdn = stack.enter_context(VDNH3(runtime=VDNRuntime(bootstrap_python=vdn_python),
                                       output_dir=output / 'vdn_h3'))
        for root in roots:
            ref = Reference(root['artifact_id'], root['path'], root['kind'])
            observation = root.get('observation')
            emit('Root ' + ref.artifact_id + ' / input', media_rows([ref]))
            emit('Root ' + ref.artifact_id + ' / observation source',
                 'Calling extractor' if observation is None else 'Using supplied observation')
            if observation is None:
                observation = extract_attribute_nodes(ref, qwen=qwen, output_dir=output / 'roots')
            emit('Root ' + ref.artifact_id + ' / extractor or supplied observation result', observation)
            if observation is None:
                graph.setdefault('root_failures', []).append(ref.artifact_id)
                save(output / 'graph.json', graph)
                continue
            records = atomize(observation)
            emit('Root ' + ref.artifact_id + ' / atoms', records)
            graph['artifacts'].append(dict(artifact_id=ref.artifact_id, kind=ref.kind,
                path=str(ref.path), observation=observation,
                provenance=attach_provenance(records, [], []), parents=[]))
            save(output / 'graph.json', graph)
        for attempt in range(attempts):
            directory = output / f'attempt_{attempt:04d}'
            directory.mkdir()
            log = {'attempt': attempt, 'status': 'running'}
            graph['attempts'].append(log)
            stage = 'selection'
            emit(f'Attempt {attempt} / start', {'attempt': attempt, 'total_attempts': attempts})
            try:
                available = [n for n in graph['artifacts'] if n['observation']['attributes']]
                if not available:
                    log.update(status='skipped', stage=stage, reason='no usable parents')
                    continue
                kind = 'video' if rng.random() < p_video else 'image'
                parent_limit = min(max_parents, 4 if kind == 'image' else 32)
                count = rng.randint(1, min(parent_limit, len(available)))
                parents = rng.sample(available, count)
                k, v = rng.randint(*keep), rng.randint(*variate)
                artifact_id = f'artifact_{next_number + attempt}'
                log.update(artifact_id=artifact_id, kind=kind, keep=k, add_count=v,
                           sampled_parents=[p['artifact_id'] for p in parents])
                emit(f'Attempt {attempt} / sampled configuration', log)
                pool = [r for p in parents for r in atomize(p['observation'], p['provenance'])]
                emit(f'Attempt {attempt} / parent atom pool', pool)
                eligible = [r for r in canonicalize_shared(pool)
                            if kind != 'image' or CATEGORY_RULES[r['category']]['temporal'] != 'Yes']
                emit(f'Attempt {attempt} / deduplicated modality-compatible candidate pool', eligible)
                sampling_trace = []
                selected = sample_selection(pool, k, kind, rng, parent_count=parent_limit, trace=sampling_trace)
                emit(f'Attempt {attempt} / sampling trace (choices versus dependencies)', sampling_trace)
                emit(f'Attempt {attempt} / actual selection',
                     {'minimum_K': k, 'actual_count': len(selected) if selected is not None else None,
                      'requested_V': v, 'atoms': selected})
                if selected is None:
                    log.update(status='skipped', stage=stage, reason='no feasible K selection')
                    continue
                log['actual_keep'] = len(selected)
                # Only carriers actually used by the sample are attached.
                used = {r['carrier_artifact_id'] for r in selected}
                log.update(parent_limit=parent_limit, parent_count=len(used))
                refs = [Reference(p['artifact_id'], p['path'], p['kind']) for p in parents
                        if p['artifact_id'] in used]
                save(directory / 'selection.json', dict(selected=selected, add_count=v, sampling_trace=sampling_trace))
                stage = 'imaginer'
                emit(f'Attempt {attempt} / imaginer input',
                     {'add_count': v, 'selected_attributes': selected, 'references': media_rows(refs), 'kind': kind})
                additions = imaginer(selected, v, refs, kind=kind, qwen=qwen,
                                 output_dir=directory / stage, seed=rng.randrange(2**31))
                emit(f'Attempt {attempt} / imaginer result', additions)
                if additions is None:
                    log.update(status='skipped', stage=stage, reason='null or invalid response')
                    continue
                planned = apply_additions(selected, additions, v, kind)
                log['actual_add'] = len(planned) - len(selected)
                target = assemble(planned, artifact_id, kind)
                emit(f'Attempt {attempt} / assembled target attributes', target)
                save(directory / 'plan.json', dict(target=target, additions=additions, records=planned))
                # Experts get new desired values without falsely claiming source support.
                rows = expert_rows(planned)
                expert_used = {a for row in rows for a in row.get('artifact_ids', [])}
                expert_refs = [r for r in refs if r.artifact_id in expert_used]
                stage = 'prompt_expert'
                _, _, catalog = prepare_inputs(rows, expert_refs, 'flux' if kind == 'image' else 'h3')
                emit(f'Attempt {attempt} / generation expert input',
                     {'expert': 'flux_prompter' if kind == 'image' else 'h3_prompter',
                      **labeled_payload(brief, rows, catalog, seconds=seconds if kind == 'video' else None)})
                options = dict(qwen=qwen, output_dir=directory / stage, user_prompt=brief)
                prompt = (flux_prompter(rows, expert_refs, **options) if kind == 'image' else
                          h3_prompter(rows, expert_refs, seconds=seconds, **options))
                emit(f'Attempt {attempt} / generation expert result', prompt)
                if prompt is None:
                    diagnostics = directory / stage / 'diagnostics.jsonl'
                    if diagnostics.is_file():
                        detail = json.loads(diagnostics.read_text().splitlines()[-1])
                        emit(f'Attempt {attempt} / generation expert rejection detail', detail)
                        log['prompter_error'] = detail
                    log.update(status='skipped', stage=stage, reason='null or invalid response')
                    continue
                save(directory / 'prompt.json', {'prompt': prompt})
                generation_refs = list(expert_refs)
                stage = 'frame_search'
                if kind == 'image':
                    for i, ref in enumerate(expert_refs):
                        if ref.kind == 'video':
                            frame = directory / f'reference_{i}.png'
                            descriptions = [r for r in rows if ref.artifact_id in r.get('artifact_ids', [])]
                            emit(f'Attempt {attempt} / frame search input',
                                 {'source': media_rows([ref])[0], 'attributes': descriptions})
                            timestamp = find_video_frame(ref.path, json.dumps(descriptions), qwen=qwen,
                                output_dir=directory / 'frames', frame_output=frame)
                            emit(f'Attempt {attempt} / frame search result',
                                 {'timestamp_seconds': timestamp, 'frame_path': str(frame) if timestamp is not None else None})
                            if timestamp is None:
                                break
                            generation_refs[i] = Reference(ref.artifact_id, frame, 'image')
                            log.setdefault('frames', []).append(dict(artifact_id=ref.artifact_id,
                                timestamp_seconds=timestamp, path=str(frame)))
                    if any(r.kind == 'video' for r in generation_refs):
                        log.update(status='skipped', stage=stage, reason='no jointly matching frame')
                        continue
                stage = 'generation'
                generation_seed = rng.randrange(2**31)
                log['generation_seed'] = generation_seed
                emit(f'Attempt {attempt} / generation start',
                     {'model': 'Flux' if kind == 'image' else 'VDNH3', 'seed': generation_seed,
                      'references': media_rows(generation_refs)})
                result = (flux(prompt, generation_refs, seed=generation_seed) if kind == 'image' else
                          vdn(prompt, generation_refs, seconds=seconds, seed=generation_seed))
                emit(f'Attempt {attempt} / generation result', result)
                if (not isinstance(result, dict) or result.get('kind') != kind or
                        not isinstance(result.get('path'), str) or not Path(result['path']).is_file()):
                    log.update(status='skipped', stage=stage, reason='missing generated media')
                    continue
                save(directory / 'generation.json', result)
                generated = Reference(artifact_id, result['path'], kind)
                stage = 'revise_observation'
                emit(f'Attempt {attempt} / verifier pass 1 start', media_rows([generated]))
                observed = attribute_verifier(mode=stage, generated_media=generated,
                    target_attributes=target, qwen=qwen, output_dir=directory / stage)
                emit(f'Attempt {attempt} / verifier pass 1 result', observed)
                if observed is None:
                    log.update(status='skipped', stage=stage, reason='null or invalid response')
                    continue
                save(directory / 'observation.json', observed)
                # Retained originals only; imagined additions have no old source.
                sources = [dict(reference=r, attributes=[a for a in selected
                           if a['carrier_artifact_id'] == r.artifact_id]) for r in refs]
                stage = 'verify_inheritance'
                verifier_sources = [dict(artifact_id=r.artifact_id,
                    attributes=compact_rows([a for a in selected if a['carrier_artifact_id'] == r.artifact_id]))
                    for r in refs]
                verifier_input, _ = inheritance_payload(generated, atomize(observed), verifier_sources, [generated]+refs)
                emit(f'Attempt {attempt} / verifier pass 2 input', verifier_input)
                matches = attribute_verifier(mode=stage, generated_media=generated,
                    observed_attributes=observed, selected_sources=sources,
                    qwen=qwen, output_dir=directory / stage)
                emit(f'Attempt {attempt} / verifier pass 2 result', matches)
                if matches is None:
                    log.update(status='skipped', stage=stage, reason='null or invalid response')
                    continue
                records = atomize(observed)
                validate_matches(matches, records, selected)
                provenance = attach_provenance(records, matches, selected)
                emit(f'Attempt {attempt} / final provenance', provenance)
                save(directory / 'inheritance.json', {'matches': matches, 'provenance': provenance})
                graph['artifacts'].append(dict(artifact_id=artifact_id, kind=kind,
                    path=str(generated.path), observation=observed, provenance=provenance,
                    parents=[r.artifact_id for r in refs], attempt=attempt))
                log.update(status='accepted', stage='complete')
            except Exception as error:
                log.update(status='error', stage=stage, reason=f'{type(error).__name__}: {error}')
                raise
            finally:
                save(directory / 'attempt.json', log)
                save(output / 'graph.json', graph)
                emit(f'Attempt {attempt} / outcome', log)
    emit('Run complete', {'artifacts': len(graph['artifacts']),
         'accepted': sum(a['status'] == 'accepted' for a in graph['attempts']),
         'graph_path': str(output / 'graph.json'), 'debug_log': str(output / 'debug.log')})
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('manifest', type=Path, help='Root JSON list; media/observation paths are relative to this file')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwritten')
    parser.add_argument('--attempts', type=int, default=10, help='Maximum trials; skipped trials count, successful nodes may be fewer')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for parent/attribute sampling and generation seeds')
    parser.add_argument('--keep', type=int, nargs=2, default=(3, 8), metavar=('MIN', 'MAX'),
                        help='Inclusive range for minimum retained K; dependencies may exceed K (e.g. 3 5)')
    parser.add_argument('--add', '--variate', dest='variate', type=int, nargs=2, default=(0, 2), metavar=('MIN', 'MAX'),
                        help='Inclusive range for NEW attributes V; 1 1 adds one, 0 0 disables; --variate is an alias')
    parser.add_argument('--parent-count', '--max-parents', dest='max_parents', type=int, default=2,
                        help='Maximum old-media references per trial; also capped at Flux 4 / H3 32')
    parser.add_argument('--p-video', type=float, default=0.5, help='Output video probability: 1=video only, 0=image only')
    parser.add_argument('--seconds', type=float, default=5.0, help='Generated video duration, 5–15 seconds; ignored for images')
    parser.add_argument('--brief', default='', help='Optional creative instruction for the generation expert')
    parser.add_argument('--vdn-python', help='Optional Python interpreter for VDN runtime bootstrap')
    parser.add_argument('--quiet', dest='debug', action='store_false',
                        help='Suppress stage dumps on stderr; still save all details to debug.log')
    args = parser.parse_args()
    graph = run(**vars(args))
    print(json.dumps({'artifacts': len(graph['artifacts']),
                      'accepted': sum(a['status'] == 'accepted' for a in graph['attempts'])}))


if __name__ == '__main__':
    main()
