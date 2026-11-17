"""Lightweight controller. Heavy runtimes exist only in inference children."""
from dataclasses import asdict, dataclass, field
import json
import secrets
import os
from pathlib import Path
import sys
import threading
import datetime
import uuid

from .references import Reference, prepare
from .qwen_sampling import generation_options
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

    def close(self):
        """Release this object's worker and weights. A later call loads afresh."""
        with _GPU_LOCK:
            session = getattr(self, '_session', None)
            if session is not None:
                session.close()
                self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)

    def _persistent_execute(self, request, job_dir):
        from .session import ProcessSession
        session = getattr(self, '_session', None)
        if session is None:
            env = runtime_environment()
            package_root = str(Path(__file__).resolve().parent.parent)
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [package_root, env.get('PYTHONPATH')]))
            directory = Path(getattr(self, 'session_dir', job_dir / 'session'))
            env['COCREATION_SESSION_DIR'] = str(directory)
            session = ProcessSession(self.backend, self.runtime.python, env, directory)
            self._session = session
        (job_dir / 'runtime.log').write_text('Persistent worker log: ' + str(session.directory / 'session.log') + '\n')
        try:
            session.call(job_dir, timeout=getattr(self.runtime, 'timeout', 7200))
        except BaseException:
            self.close()
            raise

    def _invoke(self, prompt, references, options, dry_run, *, output_dir=None, extra_request=None):
        request = prepare(prompt, list(references or []))
        request.update(backend=self.backend, model_id=self.model_id,
                       runtime=asdict(self.runtime), options=options)
        request.update(extra_request or {})
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
            run_root = self.output_dir if output_dir is None else Path(output_dir).expanduser().resolve()
            job_dir = run_root / (formatted_time + '-' + uuid.uuid4().hex[:8])
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
        self._persistent_execute(request, job_dir)



class Qwen(_System):
    backend = "qwen"

    def __init__(self, model_id=None, *, runtime=None, output_dir=None):
        self.model_id = model_id or os.environ.get('COCREATION_QWEN_MODEL') or (
            'Qwen/Qwen3.6-35B-A3B' if sys.platform.startswith('linux')
            else 'Qwen/Qwen3-VL-8B-Instruct')
        self.supports_system_cache = 'Qwen3-VL' in self.model_id
        deployment = scratch_root()/'runtime/qwen3.6-35b-a3b'
        native_python = deployment/'vllm/bin/python'
        if runtime is None and sys.platform.startswith('linux'):
            options = {'python': str(native_python)} if native_python.is_file() else {}
            manifest = deployment/'model.json'
            if manifest.is_file():
                checkpoint = json.loads(manifest.read_text())
                if checkpoint.get('model_id') == self.model_id:
                    options.update(revision=checkpoint['revision'], local_files_only=True)
            runtime = Runtime(**options)
        self.runtime = runtime or Runtime()
        if self.runtime.cpu_offload:
            raise ValueError("Qwen wrapper does not support cpu_offload; use a smaller model or FP8 checkpoint")
        self.output_dir = Path(output_dir if output_dir is not None else scratch_root() / "runs").expanduser().resolve()

    def _validate(self, request):
        if any(r["kind"] == "audio" for r in request["references"]):
            raise ValueError("Qwen wrapper accepts images and videos, not audio")

    def __call__(self, prompt: str | None = None, references: list[Reference] | None = None, *,
                 max_new_tokens=1024, video_fps=1.0, video_max_frames=16,
                 max_pixels=512 * 512, seed=None, dry_run=False, system_prompt=None,
                 assistant_prefill=None, system_cache_path=None, output_dir=None, temperature=0.8, top_p=0.95, top_k=50,
                 messages=None, response_schema=None, enable_thinking=False, _conversations=None):
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be bool")
        sampling = generation_options(dict(temperature=temperature, top_p=top_p, top_k=top_k))
        if seed is None:
            seed = secrets.randbits(63)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Qwen seed must be an integer in [0, 2**63), or None')
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if assistant_prefill is not None and not isinstance(assistant_prefill, str):
            raise TypeError("assistant_prefill must be a string or None")
        if system_cache_path is not None:
            if "Qwen3-VL" not in self.model_id:
                raise ValueError("Custom prefix cache supports legacy native Qwen3-VL only; Qwen3.6 uses vLLM caching")
            if not system_prompt:
                raise ValueError("system_cache_path requires a nonempty system_prompt")
            system_cache_path = str(Path(system_cache_path).expanduser().resolve())
        if response_schema is not None:
            if not isinstance(response_schema, dict):
                raise TypeError('response_schema must be a JSON Schema dictionary')
            json.dumps(response_schema)
            if system_cache_path or assistant_prefill is not None:
                raise ValueError('Schema decoding uses standard generate without a custom cache or assistant prefill')
        if max_new_tokens < 1 or video_fps <= 0 or video_max_frames < 2 or max_pixels < 1024:
            raise ValueError("Invalid Qwen token/frame/pixel budget")
        extra = None
        if messages is not None or _conversations is not None:
            if messages is not None and _conversations is not None:
                raise ValueError('Supply a single conversation or a batch, not both')
            if prompt is not None or references or system_prompt is not None or assistant_prefill is not None:
                raise ValueError('Structured messages cannot be combined with prompt/references/system_prompt/assistant_prefill')
            if system_cache_path:
                raise ValueError('Structured messages and batching use the official uncached generation path')
            from .qwen_inputs import normalize_messages
            conversations = [messages] if messages is not None else _conversations
            if not isinstance(conversations, (list, tuple)) or not conversations:
                raise ValueError('Batch must contain at least one conversation')
            normalized = [normalize_messages(m, check_files=not dry_run, max_pixels=max_pixels,
                video_fps=video_fps, video_max_frames=video_max_frames) for m in conversations]
            extra = dict(conversations=normalized, batched=messages is None)
            prompt = ''
        if prompt is None:
            raise ValueError('Supply prompt or messages')
        return self._invoke(prompt, references, dict(max_new_tokens=max_new_tokens,
            video_fps=video_fps, video_max_frames=video_max_frames,
            max_pixels=max_pixels, seed=seed, system_prompt=system_prompt,
            assistant_prefill=assistant_prefill, system_cache_path=system_cache_path,
            response_schema=response_schema, enable_thinking=enable_thinking, **sampling), dry_run,
            output_dir=output_dir, extra_request=extra)

    def chat(self, messages, **options):
        return self(messages=messages, **options)

    def generate_batch(self, conversations, **options):
        """One official padded generate call; decoding options/seed apply to this batch."""
        return self(_conversations=conversations, **options)


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
                 width=1024, height=1024, steps=None, guidance_scale=None, seed=0, dry_run=False):
        is_dev = self.model_id == 'black-forest-labs/FLUX.2-dev'
        steps = (50 if is_dev else 4) if steps is None else steps
        guidance_scale = (4.0 if is_dev else 1.0) if guidance_scale is None else guidance_scale
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
        self._persistent_execute(request, job_dir)


class VDNH3(_System):
    """VDN FP8 through FreeVideo, with ordered image/video/audio references."""
    backend = 'vdn_h3'
    model_id = 'OpenVDN/vdn-minimax-h3-edge'

    def __init__(self, *, runtime=None, output_dir=None):
        from .vdn import VDNRuntime
        self.runtime = runtime or VDNRuntime()
        self.output_dir = Path(output_dir if output_dir is not None else scratch_root() / 'runs').expanduser().resolve()

    def __call__(self, prompt, references=None, *, seconds=4.0, width=1280, height=736,
                 seed=0, steps=None, two_pass=False, dry_run=False):
        import math
        if isinstance(seconds, bool) or not math.isfinite(seconds) or not 1.625 <= seconds <= 15:
            raise ValueError('VDN duration must be 1.625–15 seconds')
        if any(type(v) is not int or v < (512 if two_pass else 256) or v % 32 for v in (width, height)):
            raise ValueError('VDN dimensions must be multiples of 32, >=512 for two-pass or >=256 for single-pass')
        if type(seed) is not int or not 0 <= seed < 2**63 or type(two_pass) is not bool:
            raise ValueError('Invalid seed or two_pass')
        steps = (8 if two_pass else 16) if steps is None else steps
        if type(steps) is not int or not 1 <= steps <= 32 or (two_pass and steps != 8):
            raise ValueError('VDN steps must be 1–32; two-pass requires 8 base steps')
        refs = list(references or [])
        if len(refs) > 32:
            raise ValueError('VDN supports at most 32 reference items')
        return self._invoke(prompt, refs, dict(seconds=seconds, frames=frame_count(seconds),
            width=width, height=height, seed=seed, two_pass=two_pass, base_steps=steps,
            refine_steps=3 if two_pass else 0, precision='fp8',
            task='ref2va' if refs else 't2va'), dry_run)

    def _validate(self, request):
        request.update(prepare(request['original_prompt'], [Reference(r['artifact_id'], r['path'], r['kind'])
            for r in request['references']], labels={'image': 'Picture', 'video': 'Video', 'audio': 'Audio'}))

    def _execute(self, request, job_dir):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('VDN execution requires a Linux NVIDIA host; use dry_run=True on Mac')
        if not self.runtime.accept_model_license:
            raise ValueError('Accept the MiniMax H3 model license before running VDN')
        self._persistent_execute(request, job_dir)
