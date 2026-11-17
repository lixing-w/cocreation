"""Generate independent complete sequences concurrently through one vLLM service."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dialogue_generation.generate import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('observations', nargs='+', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='qwen3.6-35b-a3b')
    p.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--reworks', type=int, choices=(0,1,2), default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--timeout', type=float, default=600)
    a = p.parse_args()
    if a.workers < 1 or len({path.stem for path in a.observations}) != len(a.observations):
        p.error('Use positive workers and unique input filenames')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures = {executor.submit(run, path, a.output_dir/path.stem, backend='vllm', model=a.model,
            base_url=a.base_url, model_mode="thinking" if a.enable_thinking else "instruct", seed=a.seed, reworks=a.reworks, resume=a.resume,
            timeout=a.timeout, max_tokens=512): path for path in a.observations}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results[path.stem] = dict(turns=len(result['turns']), placeholders=len(result['placeholders']),
                    realization=result['realization']['status'])
            except Exception as exc:
                failures[path.stem] = str(exc)
    report = dict(completed=results, failed=failures)
    (a.output_dir/'batch-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
