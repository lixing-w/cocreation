"""Lightweight controller. Heavy runtimes exist only in inference children."""
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import datetime

from .references import Reference, prepare
from .h3 import H3Runtime, MODEL_REPO, frame_count
from .storage import scratch_root, ensure_scratch, runtime_environment

_GPU_LOCK = threading.RLock()


@dataclass(frozen=True)
class Runtime:
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    cache_dir: str | None = field(default_factory=lambda: str(scratch_root() / "cache/huggingface/hub"))
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
            ensure_scratch()
            now = datetime.datetime.now()
            formatted_time = now.strftime("%y-%m-%d-%H-%M-%S-") + now.strftime("%f")[:3]
            # e.g. 26-11-08-17-29-43-823
            job_dir = self.output_dir / formatted_time
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
        env = runtime_environment()
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        _run([self.runtime.python, "-m", "cocreation.worker", str(job_dir)], job_dir, env)


def _run(command, job_dir, env=None):
    log = job_dir / "runtime.log"
    with log.open("a") as stream:
        try:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env if env is not None else runtime_environment())
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Inference/setup failed (exit {exc.returncode}); see {log}") from exc


class Qwen(_System):
    backend = "qwen"

    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", *, runtime=None, output_dir=None):
        self.model_id = model_id
        self.runtime = runtime or Runtime()
        if self.runtime.cpu_offload:
            raise ValueError("Qwen wrapper does not support cpu_offload; use a smaller model or FP8 checkpoint")
        self.output_dir = Path(output_dir if output_dir is not None else scratch_root() / "runs").expanduser().resolve()

    def _validate(self, request):
        if any(r["kind"] == "audio" for r in request["references"]):
            raise ValueError("Qwen wrapper accepts images and videos, not audio")

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

    def __init__(self, model_id="black-forest-labs/FLUX.2-klein-4B", *, runtime=None, output_dir=None):
        self.model_id = model_id
        self.runtime = runtime or Runtime()
        self.output_dir = Path(output_dir if output_dir is not None else scratch_root() / "runs").expanduser().resolve()

    def _validate(self, request):
        if any(r["kind"] != "image" for r in request["references"]):
            raise ValueError("FLUX accepts image references; extract video frames before this call")

    def __call__(self, prompt: str, references: list[Reference] | None = None, *,
                 width=1024, height=1024, steps=4, guidance_scale=1.0, seed=0, dry_run=False):
        if min(width, height) < 64 or width % 16 or height % 16 or steps < 1:
            raise ValueError("Image dimensions must be positive multiples of 16 (>=64); steps >=1")
        return self._invoke(prompt, references, dict(width=width, height=height, steps=steps,
            guidance_scale=guidance_scale, seed=seed), dry_run)


class MiniMaxH3(_System):
    backend = "minimax_h3"
    model_id = MODEL_REPO

    def __init__(self, *, runtime=None, output_dir=None):
        self.runtime = runtime or H3Runtime()
        self.output_dir = Path(output_dir if output_dir is not None else scratch_root() / "runs").expanduser().resolve()

    def __call__(self, prompt: str, references: list[Reference] | None = None, *,
                 seconds=15.0, width=768, height=512, steps=50, seed=0, dry_run=False):
        import math
        if not math.isfinite(seconds) or not 5 <= seconds <= 15:
            raise ValueError("H3 duration must be between 5 and 15 seconds")
        if any(not isinstance(v, int) or v < 64 or v % 32 for v in (width, height)):
            raise ValueError("H3 dimensions must be integer multiples of 32 (>=64)")
        if not isinstance(steps, int) or not 1 <= steps <= 10000 or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("Invalid H3 steps or seed")
        refs = list(references or [])
        counts = {kind: sum(r.kind == kind for r in refs) for kind in ("image", "video", "audio")}
        if len(refs) > 12 or counts["image"] > 9 or max(counts["video"], counts["audio"]) > 3:
            raise ValueError("H3 allows 9 images, 3 videos, 3 audios; 12 references total")
        if refs and not (counts["image"] or counts["video"]):
            raise ValueError("H3 audio references require an image or video reference")
        variant = "ref2va" if refs else "fl2va"
        return self._invoke(prompt, refs, dict(seconds=seconds, frames=frame_count(seconds),
            width=width, height=height, steps=steps, seed=seed, variant=variant,
            task="ref2va" if refs else "t2va"), dry_run)

    def _validate(self, request):
        # Re-render from original placeholders rather than replacing literal text.
        request.update(prepare(request["original_prompt"], [Reference(r["artifact_id"], r["path"], r["kind"])
                       for r in request["references"]], labels={"image": "Picture", "video": "Video", "audio": "Audio"}))

    def _execute(self, request, job_dir):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("H3 execution requires a Linux NVIDIA host; use dry_run=True on Mac")
        if not self.runtime.accept_model_license:
            raise ValueError("Read the MiniMax H3 model license, then set accept_model_license=True")
        env = runtime_environment()
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
        _run([self.runtime.python, "-m", "cocreation.comfy_worker", str(job_dir)], job_dir, env)
