"""Run bounded concurrent VLM clients for independent sequences."""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


@dataclass(frozen=True)
class Case:
    id: str
    source_format: str
    input_file: str
    seed: int
    reworks: int
    width: int
    height: int


@dataclass(frozen=True)
class Settings:
    repo: Path
    run_dir: Path
    data_dir: Path
    base_url: str
    concurrency: int = 8
    python: str = sys.executable
    video_steps: int = 40
    edit_video_sampler: str = 'res_multistep'
    match_video_brightness: bool = True
    validate_plan: bool = True
    context_tokens: int = 32768
    backend: str = "vllm"
    think_mode: str = "low"

    def model_args(self, *, thinking):
        if self.backend == "sonnet":
            return ["--backend", "sonnet", "--think-mode", self.think_mode]
        return ["--backend", "vllm", "--base-url", self.base_url,
                "--enable-thinking" if thinking else "--no-enable-thinking"]


def load_cases(path):
    cases = []
    with Path(path).open() as source:
        for row in csv.reader(source, delimiter='\t'):
            if len(row) == 8:
                del row[5]  # Legacy launch-time aspect rule.
            if len(row) != 7:
                raise ValueError('Each cases.tsv row must have seven columns')
            case_id, source_format, input_file, seed, reworks, width, height = row
            case = Case(case_id, source_format, input_file, int(seed), int(reworks), int(width), int(height))
            if not case.id or case.id in ('.', '..') or Path(case.id).name != case.id:
                raise ValueError('Case ids must be single directory names')
            if case.width < 1 or case.height < 1 or not 0 <= case.reworks <= 2:
                raise ValueError('Invalid case dimensions or rework count')
            cases.append(case)
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError('Expected nonempty, distinct case ids')
    return cases


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


async def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def command(arguments, settings, stdout, stderr):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    process = await asyncio.create_subprocess_exec(
        settings.python, '-u', *map(str, arguments), cwd=settings.repo, env=env,
        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        code = await process.wait()
        if code:
            raise subprocess.CalledProcessError(code, arguments)
    except BaseException:
        await stop_process(process)
        raise


async def run_case(case, settings):
    directory = settings.run_dir / case.id
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phase = 'observing'
    phase_started = started
    phase_seconds = {}

    def state(status, **extra):
        nonlocal phase_started
        now = time.monotonic()
        if status != 'observing':
            previous = ('observing' if status == 'dialogue_and_generation_prompts'
                        else 'dialogue_and_generation_prompts' if status == 'validating_generation_plan'
                        else phase)
            phase_seconds[previous] = round(now-phase_started, 6)
        phase_started = now
        write_json(directory / 'vlm_state.json', dict(case=case.id, phase=status, stage=phase,
            updated_utc=datetime.now(timezone.utc).isoformat(), job_id=os.environ.get('SLURM_JOB_ID'),
            node=socket.gethostname(), elapsed_seconds=round(now-started, 6), phase_seconds=phase_seconds, **extra))
        print(f'[vlm] {case.id}: {status}', flush=True)

    try:
        state(phase)
        observation = directory / 'observation'
        if not (observation / 'observation.json').is_file():
            if observation.exists():
                observation.rename(directory / f'observation.incomplete-{time.time_ns()}')
            arguments = ['tools/prompt_observer.py', settings.data_dir / case.input_file,
                '--source-format', case.source_format, '--output-dir', observation,
                *settings.model_args(thinking=True),
                '--seed', case.seed, '--max-tokens', 8192, '--context-tokens', settings.context_tokens, '--timeout', 600]
            if case.source_format == 'h3':
                arguments += ['--fps', 24]
            with (directory / 'observer.log').open('w') as log:
                await command(arguments, settings, log, subprocess.STDOUT)

        phase = 'dialogue_and_generation_prompts'
        state(phase)
        dialogue = directory / 'dialogue'
        arguments = ['scripts/dialogue_generation/generate.py', observation / 'observation.json',
            '--output-dir', dialogue, '--seed', case.seed, '--rewrite-seed', 20261118+case.seed,
            '--reworks', case.reworks,
            *settings.model_args(thinking=False),
            '--max-tokens', 4096, '--context-tokens', settings.context_tokens, '--timeout', 600]
        if (dialogue / 'dialogue.json').is_file():
            arguments.append('--resume')
        with (directory / 'dialogue.log').open('a') as log:
            await command(arguments, settings, log, subprocess.STDOUT)

        if not settings.validate_plan:
            state('complete')
            return

        phase = 'validating_generation_plan'
        state(phase)
        arguments = ['scripts/dialogue_generation/materialize.py', dialogue, '--dry-run',
            '--seed', case.seed, '--video-width', case.width, '--video-height', case.height,
            '--native-video-output', '--video-steps', settings.video_steps,
            '--edit-video-sampler', settings.edit_video_sampler,
            '--match-video-brightness' if settings.match_video_brightness else '--no-match-video-brightness']
        with (directory / 'materialize-plan.json').open('w') as output, (directory / 'preflight.log').open('w') as errors:
            await command(arguments, settings, output, errors)
        state('complete')
    except asyncio.CancelledError:
        state('cancelled')
        raise
    except Exception as exc:
        state('failed', error=str(exc))
        raise RuntimeError(f'VLM case {case.id} failed during {phase}; see {directory}') from exc


async def run_cases(cases, settings, *, isolate_failures=False, summary_path=None):
    if type(settings.concurrency) is not int or settings.concurrency < 1:
        raise ValueError('VLM concurrency must be a positive integer')
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    semaphore = asyncio.Semaphore(settings.concurrency)
    aborted = asyncio.Event()
    failures = {}

    async def worker(case):
        async with semaphore:
            if aborted.is_set():
                return
            try:
                await run_case(case, settings)
            except Exception as exc:
                if isolate_failures:
                    failures[case.id] = str(exc)
                else:
                    aborted.set()
                    raise

    tasks = [asyncio.create_task(worker(case)) for case in cases]
    status, error = 'complete', None
    try:
        await asyncio.gather(*tasks)
        if failures:
            status = 'complete_with_failures'
        return failures
    except BaseException as exc:
        status = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed'
        error = str(exc)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        write_json(summary_path or settings.run_dir / 'vlm_summary.json', dict(status=status, error=error,
            concurrency=settings.concurrency, effective_concurrency=min(settings.concurrency, len(cases)),
            base_url=settings.base_url, backend=settings.backend,
            think_mode=settings.think_mode if settings.backend == 'sonnet' else None, cases=[c.id for c in cases], failures=failures,
            started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic()-started, 6)))


async def main_async(cases, settings):
    task = asyncio.current_task()
    received = []

    def cancel(number):
        if not received:
            received.append(number)
            task.cancel()

    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, cancel, number)
    try:
        await run_cases(cases, settings)
        return 0
    except asyncio.CancelledError:
        return 128 + (received[0] if received else signal.SIGTERM)
    finally:
        for number in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--backend', choices=('vllm', 'sonnet'), default='vllm')
    parser.add_argument('--think-mode', choices=('low', 'medium'), default='low')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--video-steps', type=int, default=40)
    parser.add_argument('--edit-video-sampler', default='res_multistep')
    parser.add_argument('--match-video-brightness', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    settings = Settings(**vars(args))
    try:
        cases = load_cases(settings.run_dir / 'cases.tsv')
        return asyncio.run(main_async(cases, settings))
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
