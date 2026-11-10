"""Seeded CPG growth over (natural-language description, artifact) tuples."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import re
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cocreation import Qwen, Flux, VDNH3, VDNRuntime, Reference
from tools.extractor import extractor
from tools._attribute_protocol import (
    description_text, sample_count, sample_selection, prune_references, media_catalog, source_description)
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


def collect_generated(result, artifact_id, output):
    """Move a returned generated artifact to its permanent path before verification.

    This also preserves outputs whose verification later fails. Never move roots
    or files outside the generator's run directory; never overwrite an artifact.
    """
    source = Path(result['path']).resolve(strict=True)
    allowed = [output / 'flux', output / 'vdn_h3']
    if not any(source.is_relative_to(folder) for folder in allowed):
        raise ValueError('Generated media must be inside this run’s generator directory')
    destination = output / 'generated' / (artifact_id + source.suffix.lower())
    if destination.exists():
        raise FileExistsError(destination)
    shutil.move(str(source), destination)
    return dict(result, path=str(destination))


def clean_run(output, graph):
    """Delete only known transient directories after all models have closed."""
    # Check durable outputs before deleting any diagnostics/intermediate files.
    for attempt in graph['attempts']:
        media = attempt.get('generated_media')
        if media is not None and not Path(media['path']).is_file():
            raise FileNotFoundError(media['path'])
    for node in graph['artifacts']:
        if not Path(node['path']).is_file():
            raise FileNotFoundError(node['path'])
    for entry in output.iterdir():
        if entry.name in ('flux', 'vdn_h3', 'qwen') or re.fullmatch(r'attempt_\d+', entry.name):
            if entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)
    # Transport frames are not generated artifact nodes. Keep their source and
    # time, but do not leave live-looking paths pointing to deleted intermediates.
    for attempt in graph['attempts']:
        for frame in attempt.get('frames', []):
            frame.pop('path', None)
            frame['retained'] = False
    graph['cleaned_up'] = True
    save(output / 'graph.json', graph)


def run(manifest, output_dir, *, attempts=10, seed=0, keep=(0, 3),
        max_parents=2, p_video=0.5, seconds=5.0, brief='', vdn_python=None, debug=True,
        max_sampling_retries=128, cleanup=True):
    """Explore a new identity or reuse K sentences, generate, verify and grow the pool.

    K uses decreasing linear weights; max_parents limits sampled carriers.
    Each carrier contributes at most two lines. Invalid selections redraw K and
    attributes inside the attempt, for at most max_sampling_retries extra draws.
    Zero attributes creates a new identity as an image first. Otherwise p_video
    selects modality and rewriter only minimally resolves cross-source conflicts. Models are
    shared for the run. Invalid tool outputs skip an attempt; operational errors
    checkpoint and propagate. Only verified observations enter the pool. Generated
    files move immediately into generated/ even when verification fails. Optional
    cleanup runs only after all attempts complete and model contexts close.
    """
    if (len(keep) != 2 or any(type(x) is not int for x in keep)
            or not 0 <= keep[0] <= keep[1]):
        raise ValueError('keep must be an inclusive nonnegative integer range')
    if type(max_sampling_retries) is not int or max_sampling_retries < 0:
        raise ValueError('max_sampling_retries must be a nonnegative integer')
    if type(cleanup) is not bool:
        raise ValueError('cleanup must be boolean')
    if type(attempts) is not int or attempts < 0:
        raise ValueError('attempts must be nonnegative')
    if type(max_parents) is not int or not 1 <= max_parents <= 32:
        raise ValueError('max_parents must be 1–32')
    if not 0 <= p_video <= 1 or not 5 <= seconds <= 15:
        raise ValueError('p_video must be 0–1; seconds must be 5–15')
    roots = read_roots(manifest)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'generated').mkdir()
    (output / 'roots').mkdir()

    def emit(title, payload):
        detail = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
        block = '\n=== ' + title + ' ===\n' + detail
        with (output / 'debug.log').open('a', encoding='utf-8') as stream:
            stream.write(block + '\n')
        if debug:
            print(block, file=sys.stderr, flush=True)

    rng = random.Random(seed)
    graph = dict(version=2, seed=seed, config=dict(attempts=attempts, keep=list(keep),
        max_parents=max_parents, p_video=p_video, seconds=seconds, brief=brief,
        sampling_weights=list(range(keep[1]-keep[0]+1, 0, -1)),
        zero_attribute_kind="image", max_attributes_per_reference=2,
        max_sampling_retries=max_sampling_retries, cleanup=cleanup),
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
                artifact_id = f'artifact_{next_number + attempt}'
                for retry in range(max_sampling_retries + 1):
                    k = sample_count(keep, rng)
                    kind = 'image' if k == 0 else ('video' if rng.random() < p_video else 'image')
                    selected = sample_selection(pool, k, rng,
                        max_parents=min(max_parents, 4 if kind == 'image' else 32))
                    if selected is not None:
                        break
                log.update(artifact_id=artifact_id, kind=kind, sampled_count=k,
                           selection_retries=retry, selection_draws=retry + 1,
                           mode='new_identity' if k == 0 else 'reuse_attributes')
                if selected is None:
                    log.update(status='skipped', stage=stage,
                               reason='sampling retries exhausted: insufficient lines or reference limits')
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
                if (not isinstance(result, dict) or not isinstance(result.get('path'), str)
                        or not Path(result['path']).is_file()):
                    log.update(status='skipped', stage=stage, reason='missing generated media')
                    continue
                result = collect_generated(result, artifact_id, output)
                log['generated_media'] = dict(artifact_id=artifact_id, kind=result.get('kind'), path=result['path'])
                # Durable checkpoint even if verification or later inference fails.
                save(output / 'graph.json', graph)
                save(directory / 'generation.json', result)
                if result.get('kind') != kind:
                    log.update(status='skipped', stage=stage, reason='unexpected generated media kind')
                    continue
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
    if cleanup:
        clean_run(output, graph)
        emit('Run cleanup', 'Removed intermediate directories; media retained in generated/.')
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('manifest', type=Path, help='Root media manifest; optional description or description_file')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwritten')
    parser.add_argument('--attempts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', '--keep', dest='keep', type=int, nargs=2, default=(0, 3),
                        metavar=('MIN', 'MAX'), help='Sampled line range; lower counts have greater weight (default 40/30/20/10%%)')
    parser.add_argument('--parent-count', '--max-parents', dest='max_parents', type=int, default=2)
    parser.add_argument('--max-sampling-retries', type=int, default=128,
                        help='Retries within each attempt after the initial rejected sample')
    parser.add_argument('--cleanup', action=argparse.BooleanOptionalAction, default=True,
                        help='Remove intermediate run directories after completion; media always moves to generated/')
    parser.add_argument('--p-video', type=float, default=0.5,
                        help='Video probability for nonzero samples; zero samples always explore an image first')
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--brief', default='')
    parser.add_argument('--vdn-python')
    parser.add_argument('--quiet', dest='debug', action='store_false')
    graph = run(**vars(parser.parse_args()))
    print(json.dumps(dict(artifacts=len(graph['artifacts']),
                         accepted=sum(a['status'] == 'accepted' for a in graph['attempts']))))


if __name__ == '__main__':
    main()
