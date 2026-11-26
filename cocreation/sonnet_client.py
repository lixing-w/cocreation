"""Claude Code transport for the project's ordered VLM inputs."""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
from urllib.parse import unquote_to_bytes
import uuid

from .qwen_inputs import DEFAULT_VIDEO_FPS, normalize_messages
from .references import prepare
from .storage import scratch_root
from .worker import qwen_messages


SONNET_MODEL_ID = 'claude-sonnet-5-5'
SONNET_EFFORTS = ('low', 'medium')


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _open_media(source, timeout):
    if source.startswith(('http://', 'https://')):
        return urllib.request.urlopen(source, timeout=timeout)
    if source.startswith('data:'):
        header, sep, payload = source.partition(',')
        if not sep:
            raise ValueError('Invalid media data URI')
        data = unquote_to_bytes(payload)
        return BytesIO(base64.b64decode(data, validate=True) if header.endswith(';base64') else data)
    return Path(source).open('rb')


def _image_block(source, max_pixels, timeout, *, video_frame=False):
    from PIL import Image, ImageOps

    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError('Image max_pixels must be a positive integer')
    with _open_media(source, timeout) as stream, Image.open(stream) as original:
        image = ImageOps.exif_transpose(original)
        width, height = image.size
        scale = min(1, math.sqrt(max_pixels / (width * height)), 3000 / max(width, height))
        image.thumbnail((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        data = BytesIO()
        if video_frame or original.format == 'JPEG':
            media_type = 'image/jpeg'
            image.convert('RGB').save(data, format='JPEG', quality=92)
        else:
            media_type = 'image/png'
            mode = 'RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB'
            image.convert(mode).save(data, format='PNG')
    return dict(type='image', source=dict(type='base64', media_type=media_type,
                                          data=base64.b64encode(data.getvalue()).decode('ascii')))


def _video_frames(item, directory, timeout):
    from tools.video_frames import extract_frames, frame_timestamps, rate_indices, sample_indices

    source, fps, limit = item['video'], item['fps'], item['max_frames']
    if not math.isfinite(fps) or fps <= 0 or type(limit) is not int or limit < 2:
        raise ValueError('Video sampling requires positive fps and at least two max_frames')
    if isinstance(source, list):
        timestamps = [index / fps for index in range(len(source))]
        indices = sample_indices(timestamps, limit)
        return [(timestamps[i], source[i]) for i in indices]
    temporary = Path(tempfile.mkdtemp(dir=directory))
    path = Path(source)
    if source.startswith(('http://', 'https://', 'data:')):
        path = temporary / 'video'
        with _open_media(source, timeout) as stream, path.open('wb') as output:
            shutil.copyfileobj(stream, output)
    timestamps = frame_timestamps(path, timeout=timeout)
    indices = rate_indices(timestamps, fps)
    if len(indices) > limit:
        selected = sample_indices([timestamps[i] for i in indices], limit)
        indices = [indices[i] for i in selected]
    frames = extract_frames(path, indices, temporary / 'frames', timeout=timeout)
    return [(timestamps[i], str(frame)) for i, frame in zip(indices, frames)]


def _text(text):
    return dict(type='text', text=text)


def _content_blocks(content, directory, timeout):
    for item in [_text(content)] if isinstance(content, str) else content:
        kind = item['type']
        if kind == 'text':
            if item['text'].strip():
                yield _text(item['text'])
        elif kind == 'image':
            yield _image_block(item['image'], item['max_pixels'], timeout)
        elif kind == 'video':
            frames = _video_frames(item, directory, timeout)
            yield _text(f'[Video reference: {len(frames)} sampled frames in chronological order.]')
            for index, (seconds, frame) in enumerate(frames, 1):
                yield _text(f'[Video frame {index}/{len(frames)} at {seconds:.3f}s]')
                yield _image_block(frame, item['max_pixels'], timeout, video_frame=True)
            yield _text('[End of video reference.]')


def _split_messages(messages):
    system, turns = [], []
    for message in messages:
        content = message['content']
        if message['role'] == 'system':
            if turns or (isinstance(content, list) and any(block['type'] != 'text' for block in content)):
                raise ValueError('Sonnet requires text-only system messages before the conversation')
            system.append(content if isinstance(content, str) else '\n'.join(block['text'] for block in content))
        else:
            turns.append(message)
    if not turns or turns[-1]['role'] != 'user':
        raise ValueError('Sonnet requests must end with a user message')
    return '\n\n'.join(system), turns


def _stream_input(turns, directory, timeout):
    # A print request sends one user event. Earlier turns are a role-labelled
    # transcript so the CLI generates only the answer to the final user turn.
    blocks = []
    if len(turns) > 1:
        blocks.append(_text('[Conversation history]'))
        for turn in turns[:-1]:
            blocks.append(_text(f'[{turn["role"]}]'))
            blocks.extend(_content_blocks(turn['content'], directory, timeout))
        blocks.append(_text('[End of conversation history. Current user message follows.]'))
    blocks.extend(_content_blocks(turns[-1]['content'], directory, timeout))
    if not blocks:
        raise ValueError('Sonnet requires nonempty user content')
    return json.dumps(dict(type='user', message=dict(role='user', content=blocks),
                           parent_tool_use_id=None), ensure_ascii=False) + '\n'


class ClaudeSonnet:
    """A stateless Claude Code backend with the VLLMQwen calling convention."""

    supports_system_cache = False
    prefers_public_summaries = True

    def __init__(self, model_id=SONNET_MODEL_ID, *, timeout=600, output_dir=None,
                 video_fps=DEFAULT_VIDEO_FPS, video_max_frames=16, enable_thinking=False, effort='low', executable='claude'):
        if effort not in SONNET_EFFORTS:
            raise ValueError('Sonnet effort must be low or medium')
        self.model_id = model_id
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir is not None else scratch_root() / 'runs'
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.enable_thinking = enable_thinking
        self.effort = effort
        self.executable = executable

    def close(self):
        """Each call owns its subprocess and exits before returning."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)

    def chat(self, messages, **options):
        return self(messages=messages, **options)

    def generate_batch(self, conversations, *, concurrency=4, **options):
        if not isinstance(conversations, (list, tuple)) or not conversations:
            raise ValueError('Batch must contain at least one conversation')
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError('concurrency must be a positive integer')
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(lambda messages: self.chat(messages, **options), conversations))
        return dict(status='ok', model_id=self.model_id, batch_size=len(results), results=results)

    def __call__(self, prompt=None, references=None, *, messages=None,
                 max_new_tokens=1024, video_fps=None, video_max_frames=None, max_pixels=512*512,
                 seed=None, dry_run=False, system_prompt=None, assistant_prefill=None,
                 system_cache_path=None, output_dir=None, temperature=0.8, top_p=0.95, top_k=50,
                 response_schema=None, enable_thinking=None, thinking_token_budget=None, effort=None):
        # Seed and sampling options are accepted for compatibility. Sonnet 5.5
        # only exposes adaptive thinking and does not accept Qwen's sampling knobs.
        if assistant_prefill or system_cache_path or thinking_token_budget is not None:
            raise ValueError('Sonnet CLI does not support prefill, explicit system cache or fixed thinking budgets')
        enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        video_fps = self.video_fps if video_fps is None else video_fps
        video_max_frames = self.video_max_frames if video_max_frames is None else video_max_frames
        if (type(enable_thinking) is not bool or type(max_new_tokens) is not int or max_new_tokens < 1
                or not math.isfinite(video_fps) or video_fps <= 0
                or type(video_max_frames) is not int or video_max_frames < 2
                or type(max_pixels) is not int or max_pixels < 1024):
            raise ValueError('Invalid Sonnet thinking flag or token/frame/pixel budget')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('Sonnet timeout must be positive')
        if response_schema is not None and not isinstance(response_schema, dict):
            raise TypeError('response_schema must be a JSON Schema dictionary')
        bindings = {}
        if messages is None:
            if not isinstance(prompt, str):
                raise ValueError('Supply prompt or messages')
            request = prepare(prompt, list(references or []))
            bindings = request['bindings']
            request['options'] = dict(max_pixels=max_pixels, video_fps=video_fps,
                                      video_max_frames=video_max_frames, system_prompt=system_prompt)
            messages = qwen_messages(request)
        elif prompt is not None or references or system_prompt is not None:
            raise ValueError('Structured messages cannot be combined with prompt/references/system_prompt')
        messages = normalize_messages(messages, check_files=not dry_run, max_pixels=max_pixels,
                                      video_fps=video_fps, video_max_frames=video_max_frames)
        system, turns = _split_messages(messages)
        effort = self.effort if effort is None else effort
        if effort not in SONNET_EFFORTS:
            raise ValueError('Sonnet effort must be low or medium')
        request = dict(model=self.model_id, messages=messages, max_tokens=max_new_tokens,
                       effort=effort, response_schema=response_schema)
        if dry_run:
            return dict(status='planned', backend='sonnet', request=request, bindings=bindings)
        directory = (Path(output_dir) if output_dir is not None else self.output_dir).expanduser().resolve()
        directory /= uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        request_path = directory / 'request.json'
        _write_json(request_path, request)
        with tempfile.TemporaryDirectory(prefix='cocreation-sonnet-') as temporary:
            temp = Path(temporary)
            command = [self.executable, '--print', '--model', self.model_id, '--effort', effort,
                       '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
                       '--safe-mode', '--tools', '', '--strict-mcp-config', '--no-session-persistence']
            if system:
                system_file = temp / 'system.txt'
                system_file.write_text(system, encoding='utf-8')
                command.extend(['--system-prompt-file', str(system_file)])
            if response_schema is not None:
                command.extend(['--json-schema', json.dumps(response_schema, ensure_ascii=False)])
            env = dict(os.environ, CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(max_new_tokens),
                       CLAUDE_CODE_DISABLE_ATTACHMENTS='1')
            try:
                stdin = _stream_input(turns, temp, self.timeout)
                process = subprocess.run(command, input=stdin, capture_output=True, text=True,
                                         encoding='utf-8', cwd=temp, env=env, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f'Sonnet request timed out after {self.timeout:g}s') from exc
            except FileNotFoundError as exc:
                raise RuntimeError(f'Sonnet request could not find {exc.filename}') from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'Sonnet media conversion failed ({exc.cmd[0]}, exit {exc.returncode})') from exc
        result = self._result(process, directory, response_schema)
        result.update(job_dir=str(directory), request_path=str(request_path), bindings=bindings,
                      enable_thinking=enable_thinking, thinking_token_budget=None, effort=effort)
        _write_json(directory / 'result.json', result)
        return result

    def _result(self, process, directory, response_schema):
        if process.stderr.strip():
            (directory / 'stderr.txt').write_text(process.stderr, encoding='utf-8')
        try:
            events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        except ValueError as exc:
            raise RuntimeError(f'Invalid Claude Code stream-json output; see {directory}') from exc
        raw = next((event for event in reversed(events)
                    if isinstance(event, dict) and event.get('type') == 'result'), {})
        if raw:
            _write_json(directory / 'response.json', raw)
        stop = raw.get('stop_reason')
        if not raw or ((process.returncode or raw.get('is_error')) and stop != 'max_tokens'):
            detail = raw.get('errors') or raw.get('result') or process.stderr.strip() or raw.get('subtype')
            raise RuntimeError(f'Claude Code request failed (exit {process.returncode}): {str(detail)[:1000]}')
        text = raw.get('result') or ''
        if response_schema is not None and stop != 'max_tokens':
            if 'structured_output' not in raw:
                raise RuntimeError(f'Claude Code returned no validated structured output; see {directory}')
            text = json.dumps(raw['structured_output'], ensure_ascii=False)
        usage = dict(raw.get('usage') or {})
        input_tokens = sum((usage.get(key) or 0) for key in
                           ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')) if usage else None
        output_tokens = usage.get('output_tokens')
        usage.update(prompt_tokens=input_tokens, completion_tokens=output_tokens,
                     completion_tokens_details=dict(reasoning_tokens=(usage.get('output_tokens_details') or {}).get('thinking_tokens')))
        return dict(status='ok', text=text, model_id=self.model_id, usage=usage,
                    finish_reason={'max_tokens': 'length', 'end_turn': 'stop', 'tool_use': 'stop'}.get(stop, stop),
                    input_tokens=input_tokens, output_tokens=output_tokens, reasoning=None)
