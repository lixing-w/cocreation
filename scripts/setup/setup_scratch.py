"""Create repo/scratch as a symlink; never replace or migrate existing files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def create_link(repo, target):
    target = target.expanduser().resolve()
    link = repo / "scratch"
    if target == repo.resolve() or repo.resolve() in target.parents:
        raise ValueError("Scratch target must be outside the repository")
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Existing scratch link points elsewhere: {link.resolve()}")
    elif link.exists():
        raise ValueError(f"{link} already exists and is not a symlink; nothing was changed")
    target.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    for name in ("data", "runs", "cache", "runtime", "tmp"):
        (target / name).mkdir(exist_ok=True)
    return link


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path.home() / "scratch/cocreation")
    args = parser.parse_args()
    try:
        link = create_link(ROOT, args.target)
    except ValueError as error:
        parser.error(str(error))
    print(f"{link} -> {link.resolve()}")
    print("Before pip/model setup, run: source scripts/scratch_env.sh")


if __name__ == "__main__":
    main()
