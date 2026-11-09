"""One managed FreeVideo CLI invocation; terminate its entire process group on failure."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .vdn import generation_command, media_request, FREEVIDEO_REVISION, TORCH_VERSION
from .vdn_setup import prepare_runtime, environment, run
from .vdn_assets import verify_fp8


def validate_media(request):
    """Reject implicit truncation; video audio is always ignored by this adapter."""
    result = []
    limit = min(15., request['options']['frames'] / 24)
    for ref in request['references']:
        if ref['kind'] == 'image':
            continue
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', ref['path']], text=True))
        stream = next((s for s in info['streams'] if s['codec_type'] == ref['kind']), None)
        if stream is None:
            raise ValueError('Missing reference ' + ref['kind'] + ' stream: ' + ref['path'])
        duration = float(stream.get('duration') or info.get('format', {}).get('duration') or 0)
        if not 1 <= duration <= limit + .001:
            raise ValueError(f'Reference {ref["artifact_id"]} is {duration}s; explicitly trim it to 1–{limit:.3f}s '
                             'or increase output seconds. References are never silently truncated.')
        result.append(dict(artifact_id=ref['artifact_id'], duration=duration,
                           video_audio_used=False if ref['kind'] == 'video' else None))
    return result


def prepare_assets(root, python, rt, env):
    receipt = root / 'assets-ready.json'
    if not receipt.exists() and not rt['auto_setup']:
        raise RuntimeError('VDN assets not prepared; run scripts/setup_vdn.py first')
    if not receipt.exists() or rt['auto_setup']:
        command = [python, Path(__file__).with_name('vdn_assets.py'), root]
        if rt['local_files_only']:
            command.append('--offline')
        run(command, env=env)
    installation = json.loads(receipt.read_text())
    verify_fp8(installation['cache'])
    return installation


def stop(process):
    # Also stop descendants if the parent already exited (e.g. a failed controller).
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(request, job):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('VDN inference requires Linux/NVIDIA')
    rt = request['runtime']
    if not rt['accept_model_license']:
        raise ValueError('MiniMax H3 license acceptance required')
    metadata = validate_media(request)
    root = Path(rt['root']).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root, python = prepare_runtime(rt)
        env = environment(root)
        if rt['local_files_only']:
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        installation = prepare_assets(root, python, rt, env)
        (job / 'prompt.txt').write_text(request['prompt'])
        (job / 'media.json').write_text(json.dumps(media_request(request), indent=2))
        command = generation_command(request, job, installation)
        (job / 'command.json').write_text(json.dumps(command, indent=2))
        started = time.perf_counter()
        process = subprocess.Popen(command, env=env, cwd=root / 'vendor/FreeVideo', start_new_session=True)
        try:
            code = process.wait(timeout=rt['timeout'])
            if code:
                raise RuntimeError(f'FreeVideo exited {code}; inspect runtime.log and video.debug.json')
        finally:
            stop(process)
        output = job / 'video.mp4'
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError('FreeVideo did not produce a nonempty video.mp4')
        return dict(status='ok', kind='video', path=str(output), model_id=request['model_id'],
            model_revision=installation['model_revision'], freevideo_revision=FREEVIDEO_REVISION,
            torch=TORCH_VERSION, precision='fp8', base_steps=8,
            refine_steps=request['options']['refine_steps'], two_pass=request['options']['two_pass'],
            fps=24, frames=request['options']['frames'], seconds=request['options']['frames']/24,
            reference_media=metadata, time_elapsed=time.perf_counter()-started,
            upstream_report=str(output.with_suffix('.request.json')))


def main():
    job = Path(sys.argv[1]).resolve()
    request = json.loads((job / 'request.json').read_text())
    try:
        result = execute(request, job)
    except Exception as error:
        (job / 'error.json').write_text(json.dumps(dict(type=type(error).__name__, message=str(error)), indent=2))
        raise
    (job / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
