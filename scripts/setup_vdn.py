"""Prepare pinned FreeVideo CLI for torch 2.10/cu128. Linux only; large model download."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import VDNRuntime
from cocreation.storage import runtime_environment
from cocreation.vdn_setup import prepare_runtime, environment, run
from cocreation.vdn_worker import prepare_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=None, help='Optional existing Python 3.12; default downloads Python into runtime root')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--offline', action='store_true', help='Use an already installed runtime and cached assets')
    parser.add_argument('--probe', action='store_true', help='Run GPU kernel probes after setup')
    parser.add_argument('--runtime-only', action='store_true', help='Install code/dependencies without model downloads')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('Run setup on the Linux NVIDIA server; Mac supports dry-run and contract tests only.')
    runtime_environment()  # establish the shared cache environment below
    rt = asdict(VDNRuntime(bootstrap_python=args.python, local_files_only=args.offline,
                          **({'root': str(args.root.resolve())} if args.root else {})))
    root = Path(rt['root']).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    import os
    os.environ.update(runtime_environment())
    with (root / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root, python = prepare_runtime(rt)
        env = environment(root)
        if not args.runtime_only:
            print(json.dumps(prepare_assets(root, python, rt, env), indent=2))
        if args.probe:
            run([python, '-m', 'freevideo_engine.cli', 'doctor', '--probe',
                 '--out', root / 'gpu-probe.json'], env=env)
    print('Prepared runtime: ' + str(root))
    print('GPU generation remains unvalidated until a real multi-reference request succeeds.')


if __name__ == '__main__':
    main()
