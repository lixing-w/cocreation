"""Repeat identical requests with object-owned models; --run enables CUDA inference."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from cocreation import Qwen, Flux, MiniMaxH3, VDNH3, VDNRuntime, Reference
from cocreation.storage import scratch_root


def benchmark(image_path, video_path, output_dir, *, rounds=3, model='all',
              dry_run=True, vdn_python=None):
    """Reuse instances and fixed inputs; return per-model wall times in round order.

    Args:
        image_path: Fixed input image, reused in every round.
        video_path: Fixed reference video, reused in every round.
        output_dir: Parent directory for each model's execution records.
        rounds: Positive number of repetitions.
        model: all, qwen, flux, minimax_h3, or vdn_h3.
        dry_run: Plan requests without loading models when True.
        vdn_python: Optional existing Python 3.12 for the VDN runtime.
    """
    if type(rounds) is not int or rounds < 1:
        raise ValueError('rounds must be a positive integer')
    if model not in ('all', 'qwen', 'flux', 'minimax_h3', 'vdn_h3'):
        raise ValueError('Unknown model: ' + model)
    output = Path(output_dir).expanduser().resolve()
    image = Reference('image_2', Path(image_path).expanduser().resolve(), 'image')
    video = Reference('video_4', Path(video_path).expanduser().resolve(), 'video')
    # Construct all four once, before the rounds; constructors do not load weights.
    with ExitStack() as stack:
        qwen = stack.enter_context(Qwen(output_dir=output / 'qwen'))
        flux = stack.enter_context(Flux(output_dir=output / 'flux'))
        minimax_h3 = stack.enter_context(MiniMaxH3(output_dir=output / 'minimax_h3'))
        vdn_h3 = stack.enter_context(VDNH3(
            runtime=VDNRuntime(bootstrap_python=vdn_python), output_dir=output / 'vdn_h3'))
        requests = {
            'qwen': (qwen,
                'Describe the main subjects, colors, background and composition in {{artifact:image_2}}. '
                'Then describe the subject motion and camera motion in {{artifact:video_4}}.',
                [image, video], dict(dry_run=dry_run)),
            'flux': (flux,
                'Create a watercolor illustration of {{artifact:image_2}}. '
                'Preserve the main subjects and composition, change only the visual style.',
                [image], dict(seed=42, dry_run=dry_run)),
            # 'minimax_h3': (minimax_h3,
            #     'Create a new shot with the main subject and visual appearance from {{artifact:image_2}}. '
            #     'The camera moves just like in {{artifact:video_4}}. Keep the action coherent.',
            #     [video, image], dict(seconds=15, seed=42, dry_run=dry_run)),
            'vdn_h3': (vdn_h3,
                'Create a new shot with the subject from {{artifact:image_2}}. '
                'Follow the camera motion in {{artifact:video_4}}.',
                [image, video], dict(seconds=15, seed=42, dry_run=dry_run)),
        }
        if model != 'all' and model not in requests:
            raise ValueError('minimax_h3 call is commented out; uncomment its requests entry to benchmark it')
        names = list(requests) if model == 'all' else [model]
        if not dry_run:
            for name in names:
                for ref in requests[name][2]:
                    if not Path(ref.path).is_file():
                        raise FileNotFoundError(ref.path)
        times = {name: [] for name in names}
        for round_index in range(rounds):
            for name in names:
                instance, prompt, references, options = requests[name]
                print(f'Round {round_index+1}/{rounds}: {name}', file=sys.stderr, flush=True)
                started = time.perf_counter()
                instance(prompt, references, **options)
                times[name].append(time.perf_counter() - started)
        labels = {name: requests[name][0].model_id.rsplit('/', 1)[-1] for name in names}
    # Cleanup happens once after every round, outside the timed calls.
    for name in names:
        durations = ' -> '.join(f'{value:.1f}s' for value in times[name])
        suffix = '; dry-run only' if dry_run else ''
        print(f'{labels[name]}: {durations} ({rounds} rounds{suffix})', flush=True)
    return times


def main():
    """Parse CLI options and run the benchmark; default is side-effect-free planning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=scratch_root() / 'data/samples/image.jpg')
    parser.add_argument('--video', type=Path, default=scratch_root() / 'data/samples/video.mp4')
    parser.add_argument('--output-dir', type=Path, default=scratch_root() / 'runs/benchmark')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--model', choices=['all','qwen','flux','minimax_h3','vdn_h3'], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--vdn-python', default=None)
    args = parser.parse_args()
    try:
        benchmark(args.image, args.video, args.output_dir, rounds=args.rounds, model=args.model,
                  dry_run=not args.run, vdn_python=args.vdn_python)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
