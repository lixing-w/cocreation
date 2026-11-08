"""This example demonstrates three simple independent calls to external models. Execute them with --run on a CUDA server."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from cocreation import Flux, MiniMaxH3, Qwen, Reference, Runtime, H3Runtime
from cocreation.storage import scratch_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=scratch_root() / "data/samples/image.jpg")
    parser.add_argument("--video", type=Path, default=scratch_root() / "data/samples/video.mp4")
    parser.add_argument("--output-dir", type=Path, default=scratch_root() / "runs/example")
    parser.add_argument("--model", choices=["all", "qwen", "flux", "minimax_h3"], default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Actually load/download models and run inference")
    mode.add_argument("--dry-run", action="store_true", help="Plan only (the default)")
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if args.run:
        required = [image_path] + ([] if args.model == "flux" else [video_path])
        for path in required:
            if not path.is_file():
                parser.error(f"Missing {path}. Run python scripts/download_samples.py, or supply --image/--video.")
    image = Reference("image_2", image_path, "image")
    video = Reference("video_4", video_path, "video")
    output = args.output_dir.expanduser().resolve()
    dry_run = not args.run
    calls = {
        "qwen": lambda: Qwen(output_dir=output / "qwen")(
            "Describe the main subjects, colors, background and composition in {{artifact:image_2}}. "
            "Then describe the subject motion and camera motion in {{artifact:video_4}}.",
            [image, video], dry_run=dry_run),
        "flux": lambda: Flux(output_dir=output / "flux")(
            "Create a watercolor illustration of {{artifact:image_2}}. "
            "Preserve the main subjects and composition, change only the visual style.",
            [image], seed=42, dry_run=dry_run),
        "minimax_h3": lambda: MiniMaxH3(output_dir=output / "minimax_h3")(
            "Create a new shot with the main subject and visual appearance from {{artifact:image_2}}. "
            "The camera moves just like in {{artifact:video_4}}. Keep the action coherent.",
            [video, image], seconds=15, seed=42, dry_run=dry_run),
    }
    names = list(calls) if args.model == "all" else [args.model]
    for name in names:
        print(f"[{name}] {'planning' if dry_run else 'running'}; output root: {output / name}", file=sys.stderr, flush=True)
        result = calls[name]()
        print(json.dumps(dict(model=name, **result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
