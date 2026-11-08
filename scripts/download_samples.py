"""Download a tiny Open Images V7 + DAVIS 2017 subset, not the full datasets."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from cocreation.storage import scratch_root, ensure_scratch
DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
OI_METADATA = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
OI_BUCKET = "https://open-images-dataset.s3.amazonaws.com/validation/"


def selected_frames(names, sequence, limit):
    prefix = f"DAVIS/JPEGImages/480p/{sequence}/"
    frames = sorted(n for n in names if n.startswith(prefix)
                    and re.fullmatch(r"\d{5}\.jpg", n[len(prefix):]))[:limit]
    if not frames:
        raise ValueError(f"No DAVIS frames found for {sequence}")
    return frames


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_jpeg(path, content):
    from PIL import Image
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG for {path}")
        image.verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jpg.part")
    temp.write_bytes(content)
    temp.replace(path)


def download_images(directory, count):
    results = []
    # Stream only the first few metadata rows; do not materialize the full CSV.
    with urllib.request.urlopen(OI_METADATA, timeout=60) as response:
        rows = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8-sig", newline=""))
        for row in rows:
            image_id = row["ImageID"]
            if not re.fullmatch(r"[a-fA-F0-9]{16}", image_id):
                raise ValueError(f"Unexpected image ID: {image_id}")
            path = directory / "openimages" / f"{image_id}.jpg"
            url = OI_BUCKET + image_id + ".jpg"
            if not path.is_file():
                print(f"Downloading Open Images: {image_id}", flush=True)
                with urllib.request.urlopen(url, timeout=60) as media:
                    data = media.read(10 * 1024 * 1024 + 1)
                if len(data) > 10 * 1024 * 1024:
                    raise ValueError("Sample image exceeded 10 MiB")
                save_jpeg(path, data)
            results.append(dict(dataset="Open Images V7", split="validation", id=image_id,
                path=str(path), download_url=url, source_metadata=row))
            if len(results) == count:
                break
    if len(results) != count:
        raise ValueError(f"Requested {count} images, found {len(results)}")
    return results


def encode_video(frame_dir, output, count, fps, ffmpeg):
    temp = output.with_suffix(".part.mp4")
    try:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-start_number", "0", "-i", str(frame_dir / "%05d.jpg"),
            "-frames:v", str(count), "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
            str(temp)], check=True)
        if not temp.is_file() or not temp.stat().st_size:
            raise RuntimeError("ffmpeg produced no video")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def download_videos(directory, sequences, frame_limit, fps, ffmpeg, local_zip=None):
    if local_zip:
        archive = zipfile.ZipFile(local_zip)
    else:
        from remotezip import RemoteZip
        # HTTP Range reads only the ZIP index and selected JPEG members.
        # remotezip raises if Range is unsupported; never fall back to a full download.
        archive = RemoteZip(DAVIS_URL, timeout=60)
    results = []
    with archive:
        for sequence in sequences:
            names = selected_frames(archive.namelist(), sequence, frame_limit)
            if len(names) / fps < 1:
                raise ValueError(f"{sequence}: need at least one second of frames")
            frame_dir = directory / "davis" / sequence / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading DAVIS: {sequence}, {len(names)} frames", flush=True)
            for index, name in enumerate(names):
                path = frame_dir / f"{index:05d}.jpg"
                if not path.is_file():
                    if archive.getinfo(name).file_size > 10 * 1024 * 1024:
                        raise ValueError("Unexpectedly large DAVIS frame")
                    save_jpeg(path, archive.read(name))
            output = frame_dir.parent / "video.mp4"
            encode_video(frame_dir, output, len(names), fps, ffmpeg)
            results.append(dict(dataset="DAVIS 2017", subset="trainval-480p", id=sequence,
                path=str(output), archive_url=DAVIS_URL, source_members=names,
                frame_count=len(names), playback_fps=fps, duration_seconds=len(names) / fps,
                timing_note="Playback FPS chosen for this demo; original capture timing not asserted."))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=scratch_root() / "data" / "samples")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--sequences", nargs="+", default=["bear", "blackswan"])
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=24, help="Demo playback FPS, not claimed source FPS")
    parser.add_argument("--davis-zip", type=Path, help="Optional existing local official ZIP")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without network or writes")
    args = parser.parse_args()
    if not 1 <= args.images <= 100 or not 1 <= args.fps <= 60 or not args.fps <= args.max_frames <= 15 * args.fps:
        parser.error("Use 1–100 images, FPS 1–60, and 1–15 seconds of frames")
    if len(args.sequences) > 10 or len(set(args.sequences)) != len(args.sequences) or any(
        not re.fullmatch(r"[a-zA-Z0-9_-]+", s) for s in args.sequences):
        parser.error("Choose up to 10 unique, valid sequence names")
    directory = args.output_dir.expanduser().resolve()
    if args.dry_run:
        print(json.dumps(dict(images=args.images, sequences=args.sequences, max_frames=args.max_frames,
            playback_fps=args.fps, output_dir=str(directory), davis_archive=DAVIS_URL,
            openimages_metadata=OI_METADATA, image_path=str(directory / "image.jpg"),
            video_path=str(directory / "video.mp4")), indent=2))
        return
    ensure_scratch()
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        parser.error("ffmpeg is required; install it with conda install -c conda-forge ffmpeg")
    # Check lightweight dependencies before starting any network transfer.
    import PIL.Image  # noqa: F401
    if not args.davis_zip:
        import remotezip  # noqa: F401
    directory.mkdir(parents=True, exist_ok=True)
    images = download_images(directory, args.images)
    write_json(directory / "images.json", images)
    videos = download_videos(directory, args.sequences, args.max_frames, args.fps, ffmpeg, args.davis_zip)
    write_json(directory / "videos.json", videos)
    # Stable entry points for examples/plan.py; exact provenance stays in the manifest.
    aliases = {}
    for kind, source, filename in [("image", images[0], "image.jpg"), ("video", videos[0], "video.mp4")]:
        dest = directory / filename
        temporary = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(source["path"], temporary)
        temporary.replace(dest)
        aliases[kind] = dict(path=str(dest), source_id=source["id"],
                            sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    write_json(directory / "manifest.json", dict(images=images, videos=videos, examples=aliases))
    print(json.dumps(aliases, indent=2))


if __name__ == "__main__":
    main()
