"""Rewrite every generated placeholder after dialogue completion, or backfill a run."""
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.dialogue_models import DEFAULT_BACKEND

from scripts.dialogue_generation.generation import build_tasks, construction_order, VERSION
from scripts.dialogue_generation.catalog import placeholder_catalog
from tools.placeholder_prompter import rewrite_placeholder
from tools.dialogue_realizer import QwenRealizer


def rewrite_run(result, observation, output_dir, *, writer=None, progress=True, force=False, **model_options):
    from scripts.dialogue_generation.generate import save
    if result.get('realization', {}).get('status') != 'complete':
        raise ValueError('Finish the main dialogue before rewriting generation prompts')
    from scripts.dialogue_generation.references import assign_reference_materials
    assign_reference_materials(result)
    tasks = build_tasks(result, observation)
    order = construction_order(tasks)
    output = Path(output_dir)
    rows = {r['placeholder']: r for r in result['placeholders']}
    state = dict(status='running', version=VERSION, text_only=True, construction_order=order,
                 completed=0, model=getattr(writer, 'model', model_options.get('model')))
    if not isinstance(state['model'], (str, type(None))):
        state['model'] = 'injected'
    result['prompt_rewriting'] = state
    owned, model = False, writer
    started = time.perf_counter()
    def checkpoint():
        save(output/'dialogue.json', result)
        save(output/'placeholders.json', placeholder_catalog(result))
        save(output/'generation_plan.json', dict(version=VERSION, status=state['status'],
            construction_order=order, tasks={p: rows[p]['generation'] for p in order if 'generation' in rows[p]}))
    checkpoint()
    try:
        for index, token in enumerate(order):
            task, row = tasks[token], rows[token]
            saved = row.get('generation', {})
            if (not force and saved.get('fingerprint') == task['fingerprint']
                    and saved.get('status') == 'ready' and (not task['backend'] or saved.get('prompt'))):
                state['completed'] += 1
                continue
            row.pop('generation_prompt', None)
            row['generation'] = task
            task['status'] = 'pending'
            requests = ([task] if task['backend'] else []) + [part['generation'] for part in task.get('clip_edits', [])]
            for request_index, request_task in enumerate(requests):
                if model is None:
                    model = QwenRealizer(output_dir=output/'prompt_rewriter/runtime', **model_options)
                    owned = True
                state['model'] = getattr(model, 'model', 'injected')
                if not isinstance(state['model'], str):
                    state['model'] = 'injected'
                request = model.generation_request if callable(getattr(type(model), 'generation_request', None)) else model
                name = token.strip('[]') + (f'-clip-{request_index+1}' if task.get('clip_edits') else '')
                request_task['prompt'] = rewrite_placeholder(request_task, request,
                    seed=(result['realization'].get('rewrite_seed', result['seed']) + 100000 + index*31+request_index*3) % (2**31-3),
                    output_dir=output/'prompt_rewriter'/name)
                request_task['status'] = 'ready'
            task['status'] = 'ready'
            row['generation_prompt'] = task['prompt']
            state['completed'] += 1
            checkpoint()
            if progress:
                print(f"[generation prompts {state['completed']}/{len(order)}] {token}: {task['operation']}", file=sys.stderr, flush=True)
        state.update(status='complete', elapsed_seconds=round(time.perf_counter()-started, 3))
        checkpoint()
    except BaseException as exc:
        state.update(status='failed', error=str(exc), elapsed_seconds=round(time.perf_counter()-started,3))
        checkpoint()
        raise
    finally:
        if owned:
            model.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--backend', choices=('ollama','local','vllm'), default=DEFAULT_BACKEND)
    parser.add_argument('--model')
    parser.add_argument('--base-url', help='Default: localhost:11434 (Ollama), localhost:8000 (vLLM)')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    result = json.loads((args.run_dir/'dialogue.json').read_text())
    observation = json.loads((args.run_dir/'observation.json').read_text())
    rewrite_run(result, observation, args.run_dir, backend=args.backend, model=args.model,
                base_url=args.base_url, timeout=args.timeout, force=args.force, progress=not args.quiet)


if __name__ == '__main__':
    main()
