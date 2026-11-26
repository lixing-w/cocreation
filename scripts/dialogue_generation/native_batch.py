"""Run concurrent dialogues and schedule their H3 requests on one GPU."""
from argparse import ArgumentParser, BooleanOptionalAction
import asyncio
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Thread
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cocreation import ClaudeSonnet
from cocreation.qwen_inputs import DEFAULT_VIDEO_FPS
from cocreation.sonnet_client import SONNET_EFFORTS
from cocreation.real_prompt_policy import real_prompt_cases
from cocreation.storage import scratch_root
from scripts.dialogue_generation.materialize import atomic_text, materialization_session, model_order, read_json, save
from scripts.dialogue_generation.media_queue import AssetRejected, GenerationQueue, ReviewRequest
from scripts.dialogue_generation.parallel_vlm import Settings, load_cases, run_cases
from scripts.dialogue_generation.sglang_media import ProcessServer, SGLangMedia, free_ports, media_lane


def prepare_cases(input_file, run, reworks_policy):
    source = read_json(input_file)
    for row in source['cases']:
        path = (input_file.parent / row['input_file']).resolve(strict=True)
        if not path.is_relative_to(input_file.parent):
            raise ValueError(f'Input outside manifest directory: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['input_sha256']:
            raise ValueError(f'Input checksum changed: {path}')
    recorded = run / 'inputs.json'
    if recorded.is_file() and read_json(recorded) != source:
        raise ValueError('Resume requires the original input manifest')
    tsv = run / 'cases.tsv'
    rows, policy = real_prompt_cases(source['cases'], policy=reworks_policy,
        previous_tsv=tsv.read_text() if tsv.is_file() else None)
    atomic_text(tsv, rows)
    if not recorded.exists():
        save(recorded, source)
        save(run / 'rework_policy.json', policy)
    return load_cases(tsv)


@contextmanager
def background_vlm_loop():
    # Run subprocess-backed VLM jobs while the main thread owns the media sessions.
    ready = Future()

    async def keep_running():
        stop = asyncio.Event()
        ready.set_result((asyncio.get_running_loop(), stop))
        await stop.wait()

    worker = Thread(target=asyncio.run, args=(keep_running(),), name='dialogue-vlm')
    worker.start()
    loop, stop = ready.result()
    try:
        yield loop
    finally:
        # asyncio.run also cancels and joins any unfinished VLM subprocesses.
        loop.call_soon_threadsafe(stop.set)
        worker.join()


def run_media(cases, settings, generator, results, *, review_model=None, vlm_loop=None, state=None):
    run = settings.run_dir
    sessions, timing, counts, reviews = {}, {}, Counter(), {}
    queue = GenerationQueue({}, model_order)

    def record(phase, **data):
        if state is not None:
            state(phase, **data)

    def drained(batch, review_wait_seconds):
        record('media_wave_drained', wave=queue.round, batch=batch, buffered=len(queue.buffer),
               review_wait_seconds=round(review_wait_seconds, 6),
               completed=[case_id for case_id, row in results.items() if row['status'] == 'complete'])

    if vlm_loop is not None:
        batch_size = settings.concurrency
        batches = iter(enumerate((cases[i:i + batch_size] for i in range(0, len(cases), batch_size)), 1))
        if len(cases) > batch_size:
            (run / 'vlm-batches').mkdir(exist_ok=True)

    def prefetch():
        batch = next(batches, None)
        if batch is None:
            return None
        number, group = batch
        summary = run / (f'vlm-batches/{number:03d}.json' if len(cases) > batch_size else 'vlm_summary.json')
        record('vlm', batch=number, cases=[case.id for case in group])
        future = asyncio.run_coroutine_threadsafe(
            run_cases(group, settings, isolate_failures=True, summary_path=summary), vlm_loop)
        return number, group, future

    with ExitStack() as stack, ExitStack() as review_pool:
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="media-review")
        review_pool.callback(executor.shutdown, cancel_futures=True)

        def enqueue(group):
            units = {}
            for case in group:
                case_dir = run / case.id
                directory = case_dir / 'dialogue'
                timing[case.id] = Counter()
                session = stack.enter_context(closing(materialization_session(directory,
                    resume=(directory / 'media_manifest.json').is_file(), seed=case.seed,
                    video_width=case.width, video_height=case.height, native_video_output=True,
                    verify_media=True, shared_vlm=True, review_base_url=settings.base_url,
                    review_model=review_model, shared_generator=generator, timings=timing[case.id],
                    defer_retries=True, async_reviews=True, progress=False,
                    match_video_brightness=settings.match_video_brightness)))
                started = time.monotonic()
                try:
                    plan = next(session)
                    if not plan:
                        try:
                            session.send(None)
                        except StopIteration as done:
                            manifest, _ = done.value
                        else:
                            raise RuntimeError('An empty media plan did not finish')
                        results[case.id] = dict(status='complete', units=0, completed_units=0,
                                                media_count=len(manifest['media']))
                        continue
                    case_units = {}
                    for token, unit in plan.items():
                        model = unit['model']
                        lane = None if model is None else media_lane(
                            model[0], turbo=len(model) > 2 and model[2], has_references='t2va' not in model)
                        case_units[case.id, token] = dict(unit, model=lane,
                            dependencies=[(case.id, parent) for parent in unit['dependencies']])
                    units.update(case_units)
                    sessions[case.id] = session
                    results[case.id] = dict(status='queued_media', units=len(plan), completed_units=0)
                except Exception as exc:
                    atomic_text(case_dir / 'materialize_error.log', traceback.format_exc())
                    results[case.id] = dict(status='failed', stage='media_plan', error=str(exc))
                finally:
                    timing[case.id]['media_active'] += time.monotonic() - started
            queue.enqueue(units)
            save(run / 'results.json', results)
            return len(units)

        def advance(key, review=None):
            case_id, token = key
            case_dir = run / case_id
            status = results[case_id]
            status['status'], status['current_unit'] = 'materializing', token
            save(run / 'results.json', results)
            if review is not None:
                timing[case_id]['review'] += review.seconds
                timing[case_id]['media_active'] += review.seconds
            started = time.monotonic()
            event, manifest = None, None
            try:
                event = sessions[case_id].send(token if review is None else review)
                if isinstance(event, ReviewRequest):
                    reviews[key] = executor.submit(event.run)
            except StopIteration as done:
                manifest, event = done.value
                sessions.pop(case_id)
                status.update(status=manifest.get('status', 'complete'))
                status['media_count'] = manifest.get('completed', len(manifest['media']))
                if 'pruning' in manifest:
                    status['pruning'] = manifest['pruning']
                if status['status'] == 'failed':
                    status['stage'] = 'media'
            except Exception as exc:
                if isinstance(exc, AssetRejected):
                    event = exc.event
                sessions.pop(case_id)
                queue.discard_case(case_id)
                for other in [other for other in reviews if other[0] == case_id]:
                    reviews.pop(other).cancel()
                atomic_text(case_dir / 'materialize_error.log', traceback.format_exc())
                status.update(status='failed', stage='media', error=str(exc))
            else:
                status['status'] = 'queued_media'
                if not isinstance(event, ReviewRequest) and event is not None and event.status == 'rejected':
                    if queue.units[key]['model'] in ('video-base', 'video-text'):
                        queue.units[key]['model'] = 'video-turbo'
                    queue.defer(key)
            finally:
                timing[case_id]['media_active'] += time.monotonic() - started
                status.pop('current_unit', None)
            if status['status'] != 'failed' and not isinstance(event, ReviewRequest) and (
                    event is None or event.status == 'accepted'):
                counts[case_id] += 1
            if event is not None and getattr(event, 'status', None) == 'exhausted':
                discarded = {(case_id, member) for member in event.discarded}
                queue.discard(discarded)
                for other in discarded.intersection(reviews):
                    reviews.pop(other).cancel()
            status['completed_units'] = manifest.get('completed', counts[case_id]) if manifest is not None else counts[case_id]
            save(case_dir / 'media_timings.json', {key: round(value, 6) for key, value in timing[case_id].items()})
            save(run / 'results.json', results)
            if event is not None and not isinstance(event, ReviewRequest):
                outcome = '' if event.status == 'accepted' else f' ({event.status})'
                position = queue.units[key]['plan_position']
                print(f'[generation][T{queue.round}][{case_id}:{position}/{status["units"]}] '
                      f'{event.token}: {event.operation} -> {event.path}{outcome} {event.seconds:.1f}s '
                      f'[main remain:{len(queue.main)}][queued:{len(queue.buffer)}]', flush=True)

        if vlm_loop is None:
            pending, buffered_batch = None, 1
            enqueue(cases)
        else:
            pending, buffered_batch = prefetch(), None
        wave_open = False
        while pending is not None or buffered_batch is not None or queue or wave_open:
            # Only the dispatcher resumes materializers and commits verdicts.
            for key in list(reviews):
                future = reviews.get(key)
                if future is not None and future.done():
                    del reviews[key]
                    advance(key, future.result())
            if wave_open and not queue.main and not reviews:
                drained(active_batch, review_wait_seconds)
                wave_open = False

            # Await a whole VLM batch before promoting deferred work to the next wave.
            if pending is not None and (not wave_open or pending[2].done()):
                buffered_batch, group, future = pending
                errors = future.result()
                for case_id, error in errors.items():
                    results[case_id] = dict(status='failed', stage='vlm', error=error)
                buffered_units = enqueue([case for case in group if case.id not in errors])
                record('media_buffered', batch=buffered_batch, cases=[case.id for case in group],
                       vlm_failures=len(errors), units=buffered_units, buffered=len(queue.buffer))
                pending = None
            if not wave_open:
                if buffered_batch is None and not queue:
                    break
                queue.promote(getattr(generator, 'model', None))
                active_batch, buffered_batch = buffered_batch, None
                wave_open, review_wait_seconds = True, 0.0
                if queue.round == 1:
                    save(run / 'media_queue.json', [dict(case=case_id, unit=token, **queue.units[case_id, token])
                                                    for case_id, token in queue.main])
                record('media', wave=queue.round, batch=active_batch, units=len(queue.main))
                if active_batch is not None and vlm_loop is not None:
                    pending = prefetch()

            key = queue.pop_ready(reviews)
            if key is not None:
                advance(key)
            elif reviews:
                started = time.monotonic()
                wait(reviews.values(), return_when=FIRST_COMPLETED)
                review_wait_seconds += time.monotonic() - started
            elif queue.main:
                raise RuntimeError('No ready media unit or outstanding review')


def run_batch(args):
    repo = Path(__file__).resolve().parents[2]
    run, input_file = args.run_dir.resolve(), args.manifest.resolve(strict=True)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = prepare_cases(input_file, run, args.reworks_policy)
        if args.concurrency < 1:
            raise ValueError('Concurrency must be positive')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                       '--format=csv,noheader'], text=True)
        if len(gpu.splitlines()) != 1 or 'B200' not in gpu:
            raise RuntimeError('Expected exactly one visible B200 GPU')
        atomic_text(run / 'gpu.csv', gpu)
        scratch = scratch_root()
        runtime = scratch / 'runtime/qwen3.6-35b-a3b'
        url, port = None, None
        if args.backend == 'vllm':
            (port,) = free_ports(1)
            url = f'http://127.0.0.1:{port}'
            os.environ['COCREATION_VLLM_VIDEO_FPS'] = str(args.vlm_video_fps)
            os.environ['COCREATION_VLLM_MAX_PIXELS'] = str(args.vlm_max_pixels)
        settings = Settings(repo, run, input_file.parent, url, concurrency=args.concurrency,
                            validate_plan=False, context_tokens=args.vlm_context_tokens,
                            match_video_brightness=args.match_video_brightness,
                            backend=args.backend, think_mode=args.think_mode)
        model_config = dict(backend=args.backend,
                            think_mode=args.think_mode if args.backend == 'sonnet' else None)
        config_path = run / 'vlm_execution.json'
        if config_path.is_file() and read_json(config_path) != model_config:
            raise ValueError('Resume requires the original VLM backend and think mode')
        if (run / 'pipeline_state.json').is_file() and not config_path.is_file() and args.backend != 'vllm':
            raise ValueError('Resume a legacy vLLM batch with the vLLM backend')
        save(config_path, model_config)
        results = {case.id: dict(status='queued_vlm') for case in cases}

        started = time.monotonic()

        def state(phase, **data):
            event = dict(phase=phase, job_id=os.environ.get('SLURM_JOB_ID'),
                node=socket.gethostname(), updated_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=round(time.monotonic() - started, 6), **model_config, **data)
            save(run / 'pipeline_state.json', event)
            with (run / 'pipeline_events.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(event) + '\n')
            print(f'[batch] {phase}: {run}', flush=True)

        try:
            state('initializing', concurrency=args.concurrency, submitted=len(cases),
                  resident_media_models=args.resident_media_models, h3_vae=args.h3_vae,
                  match_video_brightness=args.match_video_brightness)
            save(run / 'results.json', results)
            with ExitStack() as stack:
                generator = stack.enter_context(closing(SGLangMedia(
                    repo, run, scratch, resident=args.resident_media_models, h3_vae=args.h3_vae)))
                review_model = None
                if args.backend == 'vllm':
                    state('starting_vllm')
                    vllm = stack.enter_context(closing(ProcessServer(
                        [str(runtime / 'vllm/bin/python'), str(repo / 'scripts/qwen/serve.py'), '--port', str(port),
                         '--no-enable-thinking', '--kv-cache-memory-gib', str(args.vlm_kv_cache_gib),
                         '--max-model-len', str(args.vlm_context_tokens), '--max-num-seqs', str(args.vlm_max_num_seqs),
                         '--max-num-batched-tokens', str(args.vlm_batched_tokens), '--video-fps', str(args.vlm_video_fps),
                         '--max-pixels', str(args.vlm_max_pixels),
                         '--allowed-local-media-path', str(run), '--server-manifest', str(run / 'vllm_server.json')],
                        url, run / 'vllm.log', repo)))
                    vllm.wait_ready(generator.client)
                else:
                    if not shutil.which('claude'):
                        raise RuntimeError('Claude Code is missing from the Slurm job PATH')
                    review_model = stack.enter_context(ClaudeSonnet(effort=args.think_mode,
                        video_fps=args.vlm_video_fps, output_dir=run / 'sonnet_review'))
                loop = stack.enter_context(background_vlm_loop())
                run_media(cases, settings, generator, results, review_model=review_model,
                          vlm_loop=loop, state=state)
            counts = Counter(row['status'] for row in results.values())
            if counts['complete'] + counts['partial'] != len(cases):
                phase = 'complete_with_failures'
            else:
                phase = 'complete_with_partial' if counts['partial'] else 'complete'
            state(phase, counts=dict(counts), sglang_loads=generator.loads)
            return 0 if phase != 'complete_with_failures' or (args.backend == 'vllm' and counts['complete']) else 1
        except BaseException as exc:
            state('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', error=str(exc))
            raise


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--concurrency', type=int, default=24)
    parser.add_argument('--resident-media-models', action=BooleanOptionalAction, default=False,
                        help='Keep H3-Image and H3 services resident and switch Turbo LoRA in place')
    parser.add_argument('--h3-vae', choices=('native', 'mt'), default='native',
                        help='Use the original or Lovart H3-MT video and audio VAEs (default: native)')
    parser.add_argument('--match-video-brightness', action=BooleanOptionalAction, default=True,
                        help='Match the brightness of video edits to their source (default: on)')
    parser.add_argument('--backend', choices=('vllm', 'sonnet'), default='vllm')
    parser.add_argument('--think-mode', choices=SONNET_EFFORTS, default='low', help='Sonnet reasoning effort')
    parser.add_argument('--vlm-max-num-seqs', type=int, default=16)
    parser.add_argument('--vlm-context-tokens', type=int, default=65536)
    parser.add_argument('--vlm-batched-tokens', type=int, default=16384)
    parser.add_argument('--vlm-video-fps', type=float, choices=(DEFAULT_VIDEO_FPS,), default=DEFAULT_VIDEO_FPS)
    parser.add_argument('--vlm-max-pixels', type=int, default=1024*1024)
    parser.add_argument('--reworks-policy', default='random_0_1', choices=('random_0_1', 'manifest', '0', '1', '2'))
    parser.add_argument('--vlm-kv-cache-gib', type=float, default=16.)
    args = parser.parse_args()

    def terminate(number, _frame):
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, terminate)
    return run_batch(args)


if __name__ == '__main__':
    raise SystemExit(main())
