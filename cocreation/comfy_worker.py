"""One local Comfy process per H3 request; first real call prepares runtime/assets."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .h3 import COMFY_REVISION, MODEL_REPO, MODEL_REVISION, build_graph, model_files


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def prepare_runtime(rt):
    root = Path(rt["root"]).expanduser().resolve()
    source = root / "ComfyUI"
    python = root / "venv/bin/python"
    ready = root / "ready.json"
    if source.exists():
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMFY_REVISION:
            raise RuntimeError(f"ComfyUI must be pinned to {COMFY_REVISION}; found {actual}")
    if ready.exists() and python.is_file() and source.is_dir():
        if json.loads(ready.read_text()).get("revision") == COMFY_REVISION:
            return source, python
    if not rt["auto_setup"] or rt["local_files_only"]:
        raise RuntimeError("Comfy runtime is not prepared; enable auto_setup with network access once")
    if not source.exists():
        run(["git", "clone", "https://github.com/Comfy-Org/ComfyUI.git", source])
        run(["git", "-C", source, "checkout", "--detach", COMFY_REVISION])
    run([sys.executable, "-m", "venv", root / "venv"])
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    constraints = root / "constraints.txt"
    constraints.write_text("torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\n")
    run([python, "-m", "pip", "install", "torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
         "--index-url", "https://download.pytorch.org/whl/cu128"])
    run([python, "-m", "pip", "install", "-c", constraints, "-r", source / "requirements.txt",
         "huggingface-hub>=0.34,<2"])
    run([python, "-m", "pip", "check"])
    ready.write_text(json.dumps({"revision": COMFY_REVISION, "torch": "2.10.0+cu128"}))
    return source, python


def download_models(python, root, variant, local_files_only):
    """Use HF cache and immutable revision; expose files through Comfy model paths."""
    files = model_files(variant)
    # Executed in the isolated environment containing huggingface_hub, not the controller.
    script = '''import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, revision, root, offline, files = json.loads(sys.argv[1])
for name in files:
    cached = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision,
                                local_files_only=offline)).resolve()
    target = Path(root) / "models" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == cached:
            continue
        raise RuntimeError("Model link points to another revision: " + str(target))
    if target.exists():
        raise RuntimeError("Refusing to overwrite an unmanaged model file: " + str(target))
    target.symlink_to(cached)
'''
    run([python, "-c", script, json.dumps([MODEL_REPO, MODEL_REVISION, str(root), local_files_only, files])])
    return files


def probe_media(path, kind):
    result = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                      "-of", "json", str(path)], text=True)
    data = json.loads(result)
    stream = next((s for s in data["streams"] if s["codec_type"] == kind), None)
    if stream is None:
        raise ValueError(f"No {kind} stream in {path}")
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0)
    if not 2 <= duration <= 15:
        raise ValueError(f"Reference {kind} must be 2–15 seconds: {path} ({duration}s)")
    return duration


def stage_media(request, directory):
    directory.mkdir(parents=True, exist_ok=True)
    names, totals = [], {"video": 0.0, "audio": 0.0}
    metadata = []
    for i, ref in enumerate(request["references"]):
        path, kind = Path(ref["path"]).expanduser().resolve(strict=True), ref["kind"] # resolve to absolute dir
        if kind == "image":
            name = f"ref_{i}{path.suffix.lower()}"
            target = directory / name 
            # known issue for comfy, doesnt accept ref files outside comfy work dir
            # cannot use symlink
            shutil.copy2(path, target)
        else:
            if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
                raise RuntimeError("Install ffmpeg and ffprobe before using video/audio references")
            duration = probe_media(path, kind)
            totals[kind] += duration
            if totals[kind] > 15.001:
                raise ValueError(f"Total reference {kind} duration exceeds 15 seconds")
            if kind == "video":
                # The upstream node trims refs to the output duration. Reject rather than silently trim.
                if duration > request["options"]["frames"] / 24 + 0.05:
                    raise ValueError("Reference video exceeds output duration; explicitly trim it or increase seconds")
                name = f"ref_{i}.mp4"
                run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0", "-an",
                     "-vf", "fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "18",
                     "-pix_fmt", "yuv420p", directory / name])
            else:
                name = f"ref_{i}{path.suffix.lower()}"
                target = directory / name 
                shutil.copy2(path, target)
            metadata.append(dict(artifact_id=ref["artifact_id"], duration=duration,
                                 video_audio_used=False if kind == "video" else None))
        names.append(name)
    return names, metadata


def api(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Comfy API {route}: {exc.read().decode(errors='replace')}") from exc


def wait_result(base, prompt_id, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Comfy exited during inference; see runtime.log")
        entry = api(base, "/history/" + prompt_id).get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("Comfy execution failed: " + json.dumps(status))
            if status.get("completed"):
                return entry
        time.sleep(1)
    raise TimeoutError("H3 inference timed out")


@contextlib.contextmanager
def server(source, python, root, job, startup_timeout):
    (root / "user").mkdir(parents=True, exist_ok=True)
    (job / "comfy_output").mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # JSON is valid YAML; model directories contain symlinks into the HF cache.
    paths = job / "model_paths.yaml"
    paths.write_text(json.dumps({"cocreation": {"base_path": str(root / "models"),
        "diffusion_models": "diffusion_models", "text_encoders": "text_encoders", "vae": "vae"}}))
    command = [str(python), str(source / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
        "--input-directory", str(job / "input"), "--output-directory", str(job / "comfy_output"),
        "--user-directory", str(root / "user"), "--extra-model-paths-config", str(paths),
        "--disable-auto-launch", "--disable-all-custom-nodes", "--offline", "--cache-none",
        "--use-pytorch-cross-attention", "--preview-method", "none"]
    process = subprocess.Popen(command, cwd=source, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError("Comfy failed to start; see runtime.log")
            try:
                api(base, "/system_stats")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Comfy startup timed out")
                time.sleep(0.5)
        yield base, process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def execute(request, job):
    job_start_time = time.perf_counter()

    rt = request["runtime"]
    if not rt["accept_model_license"]:
        raise ValueError("Model license must be accepted before download/inference")
    names, media_info = stage_media(request, job / "input")
    root = Path(rt["root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runtime.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source, python = prepare_runtime(rt)
        files = download_models(python, root, request["options"]["variant"], rt["local_files_only"])
        graph = build_graph(request, names)
        (job / "workflow.json").write_text(json.dumps(graph, indent=2))
        with server(source, python, root, job, rt["startup_timeout"]) as (base, process):
            info = api(base, "/object_info")
            missing = {n["class_type"] for n in graph.values()} - info.keys()
            if missing:
                raise RuntimeError(f"Missing Comfy nodes: {sorted(missing)}")
            submitted = api(base, "/prompt", {"prompt": graph})
            if submitted.get("node_errors") or not submitted.get("prompt_id"):
                raise RuntimeError("Invalid H3 workflow: " + json.dumps(submitted))
            history = wait_result(base, submitted["prompt_id"], process, rt["timeout"])
            (job / "history.json").write_text(json.dumps(history, indent=2))
        outputs = list((job / "comfy_output").rglob("*.mp4"))
        if len(outputs) != 1 or outputs[0].stat().st_size == 0:
            raise RuntimeError("Comfy did not produce exactly one nonempty MP4")
        target = str(outputs[0])
        job_end_time = time.perf_counter()
        execution_time = job_end_time - job_start_time
        execution_time_f = f"{execution_time:.2f}s"
        return dict(status="ok", kind="video", path=str(target), model_id=MODEL_REPO,
                    model_revision=MODEL_REVISION, comfy_revision=COMFY_REVISION,
                    weights=files, variant=request["options"]["variant"],
                    frames=request["options"]["frames"], fps=24,
                    seconds=request["options"]["frames"] / 24, reference_media=media_info,
                    time_elapsed=execution_time_f)


def main():
    job = Path(sys.argv[1]).resolve()
    request = json.loads((job / "request.json").read_text())
    result = execute(request, job)
    (job / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
