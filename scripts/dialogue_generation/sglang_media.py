"""Serial generation on one GPU with optional resident SGLang services."""
from contextlib import ExitStack
import base64
import math
import mimetypes
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

import requests

from cocreation.h3 import frame_count
from scripts.dialogue_generation.materialize import probe
from scripts.dialogue_generation.generation import H3_IMAGE_MODEL_ID

VIDEO_MODEL_ID = 'MiniMaxAI/MiniMax-H3'
VIDEO_OUTPUT_COMPRESSION = 64  # SGLang's 0–100 quality scale maps this to x264 CRF 18.
TURBO_FILE = 'minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors'
TURBO_NICKNAME = 'lightx2v-ref2va-v1.1'


def media_lane(backend, *, turbo=False, has_references=True):
    if backend == 'h3_image':
        return 'image'
    if turbo:
        return 'video-turbo'
    return 'video-base' if has_references else 'video-text'


def free_ports(count):
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        return [sock.getsockname()[1] for sock in sockets]


class ProcessServer:
    def __init__(self, command, url, log, cwd):
        self.url = url
        self.log = log
        with log.open('a') as output:
            self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, client, timeout=1200):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'Server exited with status {self.process.returncode}; see {self.log}')
            try:
                with client.get(self.url + '/health', timeout=2) as response:
                    if response.ok:
                        return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f'Server readiness timed out; see {self.log}')

    def close(self):
        pgid = self.process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        # Kill any surviving GPU workers even if their parent already exited.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()


class SGLangMedia:
    def __init__(self, repo, run_dir, scratch, *, resident=False, h3_vae='native'):
        self.repo = repo
        self.directory = run_dir / 'sglang'
        self.directory.mkdir(exist_ok=True)
        runtime = Path(os.environ.get('COCREATION_SGLANG_RUNTIME', scratch / 'runtime/sglang-h3-benchmark'))
        self.turbo_path = Path(os.environ.get('H3_VIDEO_TURBO_LORA', runtime / 'weights' / TURBO_FILE))
        self.client = requests.Session()
        self.client.trust_env = False
        self.resident = resident
        self.h3_vae = h3_vae
        self.video_profiles = {}
        self.servers = {}
        self.server = None
        self.model = None
        self.turbo_loaded = self.turbo_enabled = False
        self.loads = 0

    @staticmethod
    def recipe(operation):
        turbo = operation == 'video_edit'
        return dict(turbo=turbo, steps=8 if turbo else 50, shift_video=12., shift_audio=3.,
                    precision='int8', **({'lora': TURBO_FILE} if turbo else {}))

    def launch_command(self, kind, *options):
        command = ['bash', str(self.repo / 'scripts/serve_h3_sglang.sh'), kind]
        if kind != 'image':
            command.extend(('--h3-vae', self.h3_vae))
        return command + list(options)

    def video_profile(self, kind):
        if kind not in self.video_profiles:
            arguments = shlex.split(subprocess.check_output(
                self.launch_command(kind, '--print-config'), text=True))
            path = arguments[arguments.index('--component-weights-paths.transformer') + 1]
            name = Path(path).name.lower()
            precision = next((p for p in ('nvfp4_mixed', 'nvfp4', 'int8', 'fp8', 'bf16')
                              if p in name), 'custom')
            profile = dict(transformer=path, precision=precision)
            if self.h3_vae != 'native':
                profile['h3_vae'] = self.h3_vae
            self.video_profiles[kind] = profile
        return self.video_profiles[kind]

    def video_recipe(self, operation, *, has_references):
        recipe = self.recipe(operation)
        kind = 'video' if has_references or recipe['turbo'] else 'text-video'
        return dict(recipe, **self.video_profile(kind))

    @staticmethod
    def service_key(lane):
        if lane == 'video-turbo':
            return 'video-base'
        return lane

    def stop_service(self, key):
        server = self.servers.pop(key, None)
        if server is None:
            return
        if self.server is server:
            self.server = self.model = None
        if key == 'video-base':
            self.turbo_loaded = self.turbo_enabled = False
        server.close()

    def stop_server(self):
        self.stop_service(self.service_key(self.model))

    def close(self):
        with ExitStack() as stack:
            stack.callback(self.client.close)
            for key in self.servers:
                stack.callback(self.stop_service, key)

    def start_service(self, key):
        started = time.monotonic()
        if not self.resident:
            self.stop_server()
        self.stop_service(key)
        port, master, scheduler = free_ports(3)
        self.loads += 1
        kind = {'image': 'image', 'video-text': 'text-video'}.get(key, 'video')
        launch = self.directory / f'{os.environ.get("SLURM_JOB_ID", "manual")}-{self.loads:02}-{key}'
        launch.mkdir(exist_ok=True)
        command = self.launch_command(kind, '--port', str(port), '--master-port', str(master),
                                      '--scheduler-port', str(scheduler))
        log = launch / 'server.log'
        try:
            self.servers[key] = ProcessServer(command, f'http://127.0.0.1:{port}', log, launch)
            self.servers[key].wait_ready(self.client)
        except BaseException:
            self.stop_service(key)
            print(f'[sglang] reload failed {key}: {log} {time.monotonic() - started:.1f}s', flush=True)
            raise
        elapsed = time.monotonic() - started
        print(f'[sglang] reloaded {key}: {log} {elapsed:.1f}s', flush=True)
        return elapsed

    def set_turbo(self, enabled):
        if self.turbo_enabled == enabled:
            return
        started = time.monotonic()
        endpoint = '/v1/unmerge_lora_weights'
        body = {'target': 'transformer'}
        if enabled:
            endpoint = '/v1/set_lora'
            body.update(lora_nickname=TURBO_NICKNAME, merge_mode='dynamic', strength=1.0)
            if not self.turbo_loaded:
                body.update(lora_path=str(self.turbo_path), lora_alpha=8)
        with self.call('POST', endpoint, json=body, timeout=120) as response:
            if response.json().get('status') != 'ok':
                raise RuntimeError(f'SGLang did not acknowledge {endpoint}')
        self.turbo_enabled = enabled
        if enabled:
            self.turbo_loaded = True
        print(f'[sglang] Turbo LoRA {"on" if enabled else "off"} {time.monotonic() - started:.1f}s', flush=True)

    def select(self, model):
        key = self.service_key(model)
        server = self.servers.get(key)
        elapsed = self.start_service(key) if server is None or server.process.poll() is not None else 0.0
        self.server, self.model = self.servers[key], model
        if key == 'video-base':
            try:
                self.set_turbo(model == 'video-turbo')
            except BaseException:
                # A failed call can leave adapter state uncertain.
                self.stop_service(key)
                raise
        return elapsed

    def call(self, http_method, endpoint, *, timeout=60, **options):
        response = self.client.request(http_method, self.server.url + endpoint, timeout=timeout, **options)
        if not response.ok:
            error = response.text[:1600]
            response.close()
            raise RuntimeError(f'SGLang {http_method} {endpoint} returned HTTP {response.status_code}: {error}')
        return response

    def generate(self, backend, prompt, references, recipe, output_dir, options):
        lane = media_lane(backend, turbo=bool(recipe and recipe['turbo']), has_references=bool(references))
        reload_seconds = self.select(lane)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if lane == 'image':
                result = self.image(prompt, references, output_dir, options)
            else:
                result = self.video(prompt, references, recipe, output_dir, options)
            return dict(result, model_reload_seconds=reload_seconds)
        except (requests.RequestException, TimeoutError):
            # Release any timed-out in-flight request before the next sequence.
            self.stop_server()
            raise

    def image(self, prompt, references, directory, options):
        model = H3_IMAGE_MODEL_ID
        payload = dict(model=model, prompt=prompt, seed=options['seed'],
            size=f"{options['width']}x{options['height']}", output_format='png',
            response_format='b64_json', generator_device='cpu', num_inference_steps=6)
        with ExitStack() as stack:
            if references:
                files = []
                for ref in references:
                    if ref.kind != 'image':
                        raise ValueError('H3 Image requires prepared image references')
                    path = Path(ref.path)
                    files.append(('image[]', (path.name, stack.enter_context(path.open('rb')),
                                             mimetypes.guess_type(path.name)[0] or 'image/png')))
                response = self.call('POST', '/v1/images/edits', data=payload, files=files, timeout=900)
            else:
                response = self.call('POST', '/v1/images/generations', json=payload, timeout=900)
            with response:
                result = response.json()
        path = directory / 'generated.png'
        path.write_bytes(base64.b64decode(result['data'][0]['b64_json'], validate=True))
        return dict(status='ok', path=str(path), model_id=model, backend='sglang',
                    effective_parameters=dict(steps=6, precision='int8'))

    def video(self, prompt, references, recipe, directory, options):
        conditions = []
        for ref in references:
            kind = ref.kind
            path = Path(ref.path).resolve(strict=True)
            if kind == 'video' and any(stream['codec_type'] == 'audio' for stream in probe(path)['streams']):
                kind = 'video_audio'
            conditions.append(dict(type=kind, role='reference', uri=path.as_uri()))
        width, height, seconds = (options[field] for field in ('width', 'height', 'seconds'))
        aspect = '16:9' if width > height else '9:16' if width < height else '1:1'
        payload = dict(model=VIDEO_MODEL_ID, prompt=prompt, seed=options['seed'],
            seconds=math.ceil(seconds), task='ref2va' if conditions else 't2va',
            conditions=conditions, reference_resize_mode='match', output_compression=VIDEO_OUTPUT_COMPRESSION,
            target=dict(short_edge=min(width, height), aspect_ratio=aspect, duration_seconds=seconds),
            flow_shift=recipe['shift_video'], audio_flow_shift=recipe['shift_audio'],
            num_inference_steps=recipe['steps'], num_outputs_per_prompt=1)
        with self.call('POST', '/v1/videos', json=payload) as response:
            endpoint = '/v1/videos/' + response.json()['id']
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            with self.call('GET', endpoint) as response:
                status = response.json()
            if status['status'] == 'completed':
                path = directory / 'generated.mp4'
                with self.call('GET', endpoint + '/content', stream=True, timeout=120) as response, path.open('wb') as output:
                    for chunk in response.iter_content(1024**2):
                        output.write(chunk)
                return dict(status='ok', path=str(path), model_id=VIDEO_MODEL_ID, backend='sglang',
                            response=status, effective_parameters=dict(
                                recipe, **self.video_profile('video' if conditions or recipe['turbo'] else 'text-video'),
                                fps=24, frames=frame_count(seconds), output_compression=VIDEO_OUTPUT_COMPRESSION))
            if status['status'] in ('failed', 'cancelled'):
                raise RuntimeError(f'SGLang video generation failed: {status}')
            if self.server.process.poll() is not None:
                raise RuntimeError(f'SGLang stopped during {endpoint}; see {self.server.log}')
            time.sleep(1)
        raise TimeoutError(f'SGLang video generation timed out: {endpoint}')
