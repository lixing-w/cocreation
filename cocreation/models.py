"""Lightweight controller. Heavy runtimes exist only in inference children."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from .references import Reference, prepare

_GPU_LOCK = threading.RLock()
FREEVIDEO_REVISION = "7c3a03c1a93580454f71628b78ac3df9d9e1880e"


@dataclass(frozen=True)
class Runtime:
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    cache_dir: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    cpu_offload: bool = False
    python: str = sys.executable

    def __post_init__(self):
        if self.device != "cpu" and not (self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("Only CPU and single CUDA devices are supported; no automatic MPS")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU development requires dtype='float32'")
        if self.cpu_offload and self.device == "cpu":
            raise ValueError("CPU offload requires CUDA")


@dataclass(frozen=True)
class VideoRuntime:
    source_dir: str = "~/.cache/cocreation/FreeVideo"
    root: str = "~/.cache/cocreation/freevideo-runtime"
    gpu: str | None = None
    prepared_format: str = "auto"
    auto_setup: bool = True
    accept_model_license: bool = False
    python: str = sys.executable

    def __post_init__(self):
        if self.prepared_format not in ("auto", "int8", "fp8"):
            raise ValueError("prepared_format must be auto, int8 or fp8")


class _System:
    backend: str

    def _invoke(self, prompt, references, options, dry_run):
        request = prepare(prompt, list(references or []))
        request.update(backend=self.backend, model_id=self.model_id,
                       runtime=asdict(self.runtime), options=options)
        self._validate(request)
        if dry_run:
            return dict(status="planned", **request)
        for ref in request["references"]:
            if not Path(ref["path"]).is_file():
                raise FileNotFoundError(ref["path"])
        with _GPU_LOCK:
            job_dir = self.output_dir / uuid.uuid4().hex
            job_dir.mkdir(parents=True)
            request_path = job_dir / "request.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2))
            self._execute(request, job_dir)
            result_path = job_dir / "result.json"
            result = json.loads(result_path.read_text())
            result.update(job_dir=str(job_dir), request_path=str(request_path),
                          bindings=request["bindings"])
            return result

    def _validate(self, request):
        pass

    def _execute(self, request, job_dir):
        env = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        _run([self.runtime.python, "-m", "cocreation.worker", str(job_dir)], job_dir, env)


def _run(command, job_dir, env=None):
    log = job_dir / "runtime.log"
    with log.open("a") as stream:
        try:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Inference/setup failed (exit {exc.returncode}); see {log}") from exc


class Qwen(_System):
    backend = "qwen"

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", *, runtime=None, output_dir="runs"):
        self.model_id = model_id
        self.runtime = runtime or Runtime()
        if self.runtime.cpu_offload:
            raise ValueError("Qwen wrapper does not support cpu_offload; use a smaller model or FP8 checkpoint")
        self.output_dir = Path(output_dir).expanduser().resolve()

    def __call__(self, prompt: str, references: list[Reference] | None = None, *,
                 max_new_tokens=1024, video_fps=1.0, video_max_frames=16,
                 max_pixels=512 * 512, seed=0, dry_run=False):
        if max_new_tokens < 1 or video_fps <= 0 or video_max_frames < 2 or max_pixels < 1024:
            raise ValueError("Invalid Qwen token/frame/pixel budget")
        return self._invoke(prompt, references, dict(max_new_tokens=max_new_tokens,
            video_fps=video_fps, video_max_frames=video_max_frames,
            max_pixels=max_pixels, seed=seed), dry_run)


class Flux(_System):
    backend = "flux"

    def __init__(self, model_id="black-forest-labs/FLUX.2-klein-4B", *, runtime=None, output_dir="runs"):
        self.model_id = model_id
        self.runtime = runtime or Runtime()
        self.output_dir = Path(output_dir).expanduser().resolve()

    def _validate(self, request):
        if any(r["kind"] != "image" for r in request["references"]):
            raise ValueError("FLUX accepts image references; extract video frames before this call")

    def __call__(self, prompt: str, references: list[Reference] | None = None, *,
                 width=1024, height=1024, steps=4, guidance_scale=1.0, seed=0, dry_run=False):
        if min(width, height) < 64 or width % 16 or height % 16 or steps < 1:
            raise ValueError("Image dimensions must be positive multiples of 16 (>=64); steps >=1")
        return self._invoke(prompt, references, dict(width=width, height=height, steps=steps,
            guidance_scale=guidance_scale, seed=seed), dry_run)


class FreeVideo(_System):
    backend = "freevideo"
    model_id = "MiniMax-H3/FreeVideo"

    def __init__(self, *, runtime=None, output_dir="runs"):
        self.runtime = runtime or VideoRuntime()
        self.output_dir = Path(output_dir).expanduser().resolve()

    def __call__(self, prompt: str, references: list[Reference] | None = None, *,
                 seconds=5.0, width=768, height=512, seed=0,
                 first: Reference | None = None, last: Reference | None = None, dry_run=False):
        if not 0 < seconds <= 15 or min(width, height) < 64 or width % 64 or height % 64:
            raise ValueError("Video duration must be (0,15]; dimensions must be multiples of 64")
        if (first or last) and references:
            raise ValueError("FreeVideo keyframes and references cannot be mixed")
        anchors = [r for r in (first, last) if r is not None]
        if any(r.kind != "image" for r in anchors):
            raise ValueError("First/last keyframes must be images")
        refs = list(references or anchors)
        task = "fl2va" if first and last else "i2va" if first else "l2va" if last else "ref2va" if refs else "t2va"
        return self._invoke(prompt, refs, dict(seconds=seconds, width=width, height=height,
            seed=seed, task=task, first=first.artifact_id if first else None,
            last=last.artifact_id if last else None), dry_run)

    def _execute(self, request, job_dir):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("FreeVideo execution requires a Linux NVIDIA host; use dry_run=True on Mac")
        rt = self.runtime
        source = Path(rt.source_dir).expanduser().resolve()
        root = Path(rt.root).expanduser().resolve()
        if not source.exists():
            if not rt.auto_setup:
                raise FileNotFoundError(source)
            source.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "https://github.com/FlashML-org/FreeVideo.git", str(source)], job_dir)
            _run(["git", "-C", str(source), "checkout", "--detach", FREEVIDEO_REVISION], job_dir)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != FREEVIDEO_REVISION:
            raise RuntimeError(f"FreeVideo source must be pinned to {FREEVIDEO_REVISION}; got {revision}")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(source), env.get("PYTHONPATH")]))
        # Preserve scheduler-provided visibility unless explicitly overridden.
        gpu = rt.gpu or env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [rt.python, "-m", "freevideo_engine.managed", "--root", str(root)]
        machine_path = root / "machine.json"
        ready = machine_path.is_file() and json.loads(machine_path.read_text()).get("ready", False)
        if not ready:
            if not rt.auto_setup:
                raise RuntimeError("FreeVideo runtime is not prepared and auto_setup=False")
            setup = command + ["setup", "--yes", "--no-launch", "--network", "official",
                               "--gpu", gpu, "--prepared-format", rt.prepared_format]
            if rt.accept_model_license:
                setup.append("--accept-model-license")
            _run(setup, job_dir, env)
        prompt_file = job_dir / "prompt.txt"
        prompt_file.write_text(request["prompt"], encoding="utf-8")
        media = freevideo_media(request)
        media_file = job_dir / "media.json"
        media_file.write_text(json.dumps(media, indent=2))
        opts = request["options"]
        output = job_dir / "video.mp4"
        args = command + ["generate", "--prompt-file", str(prompt_file), "--out", str(output)]
        for key in ("task", "seconds", "width", "height", "seed"):
            args += ["--" + key, str(opts[key])]
        if request["references"]:
            args += ["--media", str(media_file)]
        _run(args, job_dir, env)
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError(f"FreeVideo returned no video; see {job_dir / 'runtime.log'}")
        (job_dir / "result.json").write_text(json.dumps(dict(status="ok", path=str(output),
            kind="video", source_revision=revision)))


def freevideo_media(request):
    opts = request["options"]
    media = {"version": 1}
    for key in ("first", "last"):
        if opts.get(key):
            media[key] = request["bindings"][opts[key]]["path"]
    if not (opts.get("first") or opts.get("last")):
        media["references"] = [dict(path=r["path"], kind=r["kind"],
            **({"sound": False} if r["kind"] == "video" else {})) for r in request["references"]]
    return media
