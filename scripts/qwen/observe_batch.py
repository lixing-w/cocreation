"""Observe independent videos concurrently through a shared vLLM model instance."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.video_observer import observe_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('videos', nargs='+', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--max-assets', type=int, default=8)
    a = p.parse_args()
    if a.workers < 1 or len({path.stem for path in a.videos}) != len(a.videos):
        p.error('Use positive workers and unique video filenames')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(completed={}, failed={})
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures = {executor.submit(observe_video, path, a.output_dir/path.stem,
            backend='vllm', base_url=a.base_url, enable_thinking=a.enable_thinking, frames=a.frames, max_assets=a.max_assets): path
            for path in a.videos}
        for future in as_completed(futures):
            path = futures[future]
            try:
                observation = future.result()
                report['completed'][path.stem] = dict(assets=len(observation['assets']), shots=len(observation['shots']))
            except Exception as exc:
                report['failed'][path.stem] = str(exc)
    (a.output_dir/'batch-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
