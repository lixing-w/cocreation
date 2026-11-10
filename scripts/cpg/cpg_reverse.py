"""Seeded CPG growth over (natural-language description, artifact) tuples."""
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
from tools.extractor import extractor
from tools._attribute_protocol import (
    description_text, sample_selection, prune_references, media_catalog, source_description)
from tools.attribute_rewriter import attribute_rewriter
from tools.attribute_verifier import attribute_verifier
from tools.flux_prompter import flux_prompter
from tools.h3_prompter import h3_prompter
from tools.find_video_frame import find_video_frame


def save(path, value):
    """JSON is local checkpoint metadata only; never sent to a VLM."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def artifact_number(artifact_id):
    match = re.fullmatch(r'artifact_(\d+)', artifact_id)
    if not match:
        raise ValueError('Use numeric artifact IDs: artifact_0, artifact_1, ...')
    return int(match[1])


def read_roots(manifest):
    manifest = Path(manifest).expanduser().resolve()
    roots = json.loads(manifest.read_text(encoding='utf-8'))
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
        if 'observation' in root:
            raise ValueError('Legacy JSON observations are unsupported; omit observation to re-extract, '
                             'or supply plain-text description/description_file')
        if 'description_file' in root:
            if 'description' in root:
                raise ValueError('Use description or description_file, not both')
            root['description'] = (manifest.parent / Path(root['description_file']).expanduser()).read_text(encoding='utf-8')
        if 'description' in root:
            root['description'] = description_text(root['description'])
    return roots


def run(manifest, output_dir, *, attempts=10, seed=0, keep=(3, 8),
        max_parents=2, p_video=0.5, seconds=5.0, brief='', vdn_python=None, debug=True):
    """Sample K sentences, creatively rewrite, generate, observe and add to the pool.

    K uses an inclusive range; max_parents limits sampled carriers. Rewriter has
    no fixed addition count. Modality is sampled before rewriting. Models are
    shared for the run. Invalid tool outputs skip an attempt; operational errors
    checkpoint and propagate. Only verified observations enter the pool.
    """
    if (len(keep) != 2 or any(type(x) is not int for x in keep)
            or not 1 <= keep[0] <= keep[1]):
        raise ValueError('keep must be an inclusive positive integer range')
    if type(attempts) is not int or attempts < 0:
        raise ValueError('attempts must be nonnegative')
    if type(max_parents) is not int or not 1 <= max_parents <= 32:
        raise ValueError('max_parents must be 1–32')
    if not 0 <= p_video <= 1 or not 5 <= seconds <= 15:
        raise ValueError('p_video must be 0–1; seconds must be 5–15')
    roots = read_roots(manifest)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    def emit(title, payload):
        detail = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        block = '\n=== ' + title + ' ===\n' + detail
        with (output / 'debug.log').open('a', encoding='utf-8') as stream:
            stream.write(block + '\n')
        if debug:
            print(block, file=sys.stderr, flush=True)

    rng = random.Random(seed)
    graph = dict(version=2, seed=seed, config=dict(attempts=attempts, keep=list(keep),
        max_parents=max_parents, p_video=p_video, seconds=seconds, brief=brief),
        artifacts=[], attempts=[])
    # Runtime pool is literally a list of (description, artifact reference) tuples.
    # graph.artifacts persists the same pairs plus attempt/parent bookkeeping.
    pool = []
    next_number = max(artifact_number(r['artifact_id']) for r in roots) + 1
    save(output / 'graph.json', graph)
    emit('Run configuration', graph['config'])
    with ExitStack() as stack:
        qwen = stack.enter_context(Qwen(output_dir=output / 'qwen'))
        flux = stack.enter_context(Flux(output_dir=output / 'flux'))
        vdn = stack.enter_context(VDNH3(runtime=VDNRuntime(bootstrap_python=vdn_python),
                                       output_dir=output / 'vdn_h3'))
        for root in roots:
            ref = Reference(root['artifact_id'], root['path'], root['kind'])
            try:
                observed = root.get('description')
                if observed is None:
                    observed = extractor(ref, qwen=qwen, output_dir=output / 'roots' / ref.artifact_id)
                emit('Root ' + ref.artifact_id + ' / description', observed)
                if observed is None:
                    graph.setdefault('root_failures', []).append(ref.artifact_id)
                    continue
                observed = description_text(observed)
                pool.append((observed, ref))
                graph['artifacts'].append(dict(artifact_id=ref.artifact_id, kind=ref.kind,
                    path=str(ref.path), description=observed, parents=[]))
            except Exception as error:
                graph.setdefault('root_errors', []).append(dict(artifact_id=ref.artifact_id, reason=str(error)))
                raise
            finally:
                save(output / 'graph.json', graph)
        for attempt in range(attempts):
            directory = output / f'attempt_{attempt:04d}'
            directory.mkdir()
            log = dict(attempt=attempt, status='running')
            graph['attempts'].append(log)
            stage = 'selection'
            try:
                kind = 'video' if rng.random() < p_video else 'image'
                k = rng.randint(*keep)
                artifact_id = f'artifact_{next_number + attempt}'
                log.update(artifact_id=artifact_id, kind=kind, sampled_count=k)
                selected = sample_selection(pool, k, rng,
                    max_parents=min(max_parents, 4 if kind == 'image' else 32))
                if selected is None:
                    log.update(status='skipped', stage=stage, reason='insufficient sample within parent cap')
                    continue
                sampled, refs = selected
                log['sampled_parents'] = [r.artifact_id for r in refs]
                (directory / 'selection.txt').write_text(sampled, encoding='utf-8')
                emit(f'Attempt {attempt} / sampled description', sampled)
                stage = 'attribute_rewriter'
                planned = attribute_rewriter(sampled, refs, kind=kind, brief=brief, qwen=qwen,
                    output_dir=directory / stage, seed=rng.randrange(2**31))
                if planned is None:
                    log.update(status='skipped', stage=stage, reason='invalid or unavailable rewrite')
                    continue
                planned, expert_refs = prune_references(planned, refs)
                (directory / 'plan.txt').write_text(planned, encoding='utf-8')
                log['references'] = [dict(artifact_id=r.artifact_id, kind=r.kind,
                    path=str(r.path), label=media_catalog(expert_refs)[r.artifact_id]) for r in expert_refs]
                emit(f'Attempt {attempt} / rewritten description', planned)
                stage = 'prompt_expert'
                options = dict(qwen=qwen, output_dir=directory / stage, user_prompt=brief)
                prompt = (flux_prompter(planned, expert_refs, **options) if kind == 'image' else
                          h3_prompter(planned, expert_refs, seconds=seconds, **options))
                if prompt is None:
                    log.update(status='skipped', stage=stage, reason='invalid or unavailable expert prompt')
                    continue
                (directory / 'prompt.txt').write_text(prompt, encoding='utf-8')
                emit(f'Attempt {attempt} / expert prompt', prompt)
                generation_refs = list(expert_refs)
                stage = 'frame_search'
                if kind == 'image':
                    labels = media_catalog(expert_refs)
                    for i, ref in enumerate(expert_refs):
                        if ref.kind == 'video':
                            frame = directory / f'reference_{i}.png'
                            query = source_description(planned, labels[ref.artifact_id])
                            timestamp = find_video_frame(ref.path, query, qwen=qwen,
                                output_dir=directory / 'frames', frame_output=frame)
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
                result = (flux(prompt, generation_refs, seed=generation_seed) if kind == 'image' else
                          vdn(prompt, generation_refs, seconds=seconds, seed=generation_seed))
                if (not isinstance(result, dict) or result.get('kind') != kind or
                        not isinstance(result.get('path'), str) or not Path(result['path']).is_file()):
                    log.update(status='skipped', stage=stage, reason='missing generated media')
                    continue
                save(directory / 'generation.json', result)
                generated = Reference(artifact_id, result['path'], kind)
                stage = 'attribute_verifier'
                observed = attribute_verifier(generated_media=generated, description=planned,
                    qwen=qwen, output_dir=directory / stage)
                if observed is None:
                    log.update(status='skipped', stage=stage, reason='invalid or unavailable observation')
                    continue
                observed = description_text(observed)
                (directory / 'observation.txt').write_text(observed, encoding='utf-8')
                emit(f'Attempt {attempt} / verified observation', observed)
                pool.append((observed, generated))
                graph['artifacts'].append(dict(artifact_id=artifact_id, kind=kind,
                    path=str(generated.path), description=observed,
                    parents=[r.artifact_id for r in expert_refs], attempt=attempt))
                log.update(status='accepted', stage='complete')
            except Exception as error:
                log.update(status='error', stage=stage, reason=f'{type(error).__name__}: {error}')
                raise
            finally:
                save(directory / 'attempt.json', log)
                save(output / 'graph.json', graph)
                emit(f'Attempt {attempt} / outcome', log)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('manifest', type=Path, help='Root media manifest; optional description or description_file')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwritten')
    parser.add_argument('--attempts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', '--keep', dest='keep', type=int, nargs=2, default=(3, 8),
                        metavar=('MIN', 'MAX'), help='Inclusive range of sampled description lines')
    parser.add_argument('--parent-count', '--max-parents', dest='max_parents', type=int, default=2)
    parser.add_argument('--p-video', type=float, default=0.5)
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--brief', default='')
    parser.add_argument('--vdn-python')
    parser.add_argument('--quiet', dest='debug', action='store_false')
    graph = run(**vars(parser.parse_args()))
    print(json.dumps(dict(artifacts=len(graph['artifacts']),
                         accepted=sum(a['status'] == 'accepted' for a in graph['attempts']))))


if __name__ == '__main__':
    main()
