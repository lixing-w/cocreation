"""Object-owned FreeVideo resident session with per-request CLI orchestration."""
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

from .vdn import generation_command, media_request, FREEVIDEO_REVISION, TORCH_VERSION
from .vdn_setup import prepare_runtime, environment, run
from .vdn_assets import verify_fp8


def validate_media(request):
    """Reject implicit truncation and record reference video soundtracks."""
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
                           video_audio_used=any(s['codec_type'] == 'audio' for s in info['streams']) if ref['kind'] == 'video' else None))
    return result



def reference_prompt(request, metadata):
    """Account for embedded soundtracks in FreeVideo's ordered audio labels."""
    sounded = {m['artifact_id'] for m in metadata if m.get('video_audio_used')}
    standalone = total = 0
    labels = {}
    for ref in request['references']:
        if ref['kind'] == 'video' and ref['artifact_id'] in sounded:
            total += 1
        elif ref['kind'] == 'audio':
            standalone += 1
            total += 1
            labels[standalone] = total
    return re.sub(r'<Audio (\d+)>', lambda m: f"<Audio {labels.get(int(m[1]), int(m[1]))}>", request['prompt'])


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


class VDNSession:
    """Prepare once, then own a persistent controller in the isolated environment."""
    def __init__(self):
        self.session = None
        self.runtime = None
        self.installation = None

    def execute(self, request, job):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('VDN inference requires Linux/NVIDIA')
        rt = request['runtime']
        if not rt['accept_model_license']:
            raise ValueError('MiniMax H3 license acceptance required')
        if self.runtime is not None and self.runtime != rt:
            raise ValueError('Close the object before changing runtime configuration')
        if self.session is None:
            from .session import ProcessSession
            root = Path(rt['root']).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            with (root / 'runtime.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                root, python = prepare_runtime(rt)
                env = environment(root)
                if rt['local_files_only']:
                    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                self.installation = prepare_assets(root, python, rt, env)
            env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env['PYTHONPATH']
            env['FREEVIDEO_KEEP_MODELS'] = 'on'
            self.session = ProcessSession('vdn_native', python, env, job / 'vdn_session')
            self.runtime = dict(rt)
        (job / 'installation.json').write_text(json.dumps(self.installation))
        self.session.call(job, rt['timeout'])
        return json.loads((job / 'result.json').read_text())

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None


class NativeVDNSession:
    """Own FreeVideo's native model bank, not just a long-lived CLI shell."""
    def __init__(self):
        from freevideo_engine.resident_process import SessionOwner
        self.owner = SessionOwner()

    def execute(self, request, job):
        from freevideo_engine.resident_process import ENV, status
        rt = request['runtime']
        root = Path(rt['root']).expanduser().resolve()
        metadata = validate_media(request)
        installation = json.loads((job / 'installation.json').read_text())
        env = environment(root)
        env['FREEVIDEO_KEEP_MODELS'] = 'on'
        if rt['local_files_only']:
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # No idle timer/watch is scheduled: ownership lasts until close().
        env[ENV] = self.owner.start(root, root / 'vendor/FreeVideo', installation['python'], env, busy=True)
        (job / 'prompt.txt').write_text(reference_prompt(request, metadata))
        (job / 'media.json').write_text(json.dumps(media_request(request), indent=2))
        command = generation_command(request, job, installation)
        (job / 'command.json').write_text(json.dumps(command, indent=2))
        started = time.perf_counter()
        process = None
        try:
            process = subprocess.Popen(command, env=env, cwd=root / 'vendor/FreeVideo', start_new_session=True)
            code = process.wait(timeout=rt['timeout'])
            if code:
                raise RuntimeError(f'FreeVideo exited {code}; inspect session.log and video.debug.json')
        finally:
            if process is not None:
                stop(process)
            self.owner.active = False
        output = job / 'video.mp4'
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError('FreeVideo did not produce a nonempty video.mp4')
        return dict(status='ok', kind='video', path=str(output), model_id=request['model_id'],
            model_revision=installation['model_revision'], freevideo_revision=FREEVIDEO_REVISION,
            torch=TORCH_VERSION, precision='fp8', base_steps=8,
            refine_steps=request['options']['refine_steps'], two_pass=request['options']['two_pass'],
            fps=24, frames=request['options']['frames'], seconds=request['options']['frames']/24,
            reference_media=metadata, time_elapsed=time.perf_counter()-started,
            resident=status(env), upstream_report=str(output.with_suffix('.request.json')))

    def close(self):
        self.owner.close()


def execute(request, job):
    session = VDNSession()
    try:
        return session.execute(request, job)
    finally:
        session.close()


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
