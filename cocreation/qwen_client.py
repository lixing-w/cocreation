"""Stateless vLLM transport; no Torch imports and no changes to Ollama's protocol."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import secrets
import urllib.error
import urllib.request
import uuid

from .qwen_inputs import normalize_messages
from .qwen_sampling import generation_options
from .references import prepare
from .storage import scratch_root

MODEL_ID = 'nvidia/Qwen3.6-35B-A3B-NVFP4'


def serving_messages(messages):
    """Translate ordered official Qwen blocks to vLLM's multimodal chat protocol."""
    result = []
    for message in messages:
        row = dict(message)
        if isinstance(message['content'], list):
            content = []
            for item in message['content']:
                kind = item['type']
                if kind == 'text':
                    content.append(dict(type='text', text=item['text']))
                    continue
                source = item[kind]
                if not isinstance(source, str):
                    raise ValueError('vLLM video inputs require a video file/URL; use the native adapter for frame lists')
                url = source if source.startswith(('http://', 'https://', 'data:')) else Path(source).as_uri()
                content.append(dict(type=kind+'_url', **{kind+'_url': {'url': url}}))
            row['content'] = content
        result.append(row)
    return result


class VLLMQwen:
    """One external model service, with isolated messages/KV state per request.

    Local paths must exist on the serving host under --allowed-local-media-path.
    generate_batch submits concurrent requests; vLLM performs continuous batching.
    """
    supports_system_cache = False  # vLLM owns prefix/KV state.

    def __init__(self, model_id=MODEL_ID, *, base_url='http://127.0.0.1:8000',
                 timeout=600, output_dir=None, video_fps=1.0, video_max_frames=16, enable_thinking=False):
        self.model_id = model_id
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be bool")
        self.enable_thinking = enable_thinking
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.timeout = timeout
        self.video_fps, self.video_max_frames = video_fps, video_max_frames
        self.output_dir = Path(output_dir) if output_dir is not None else scratch_root()/'runs'

    def close(self):
        """The caller owns the external server's lifetime."""

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
        # Distinct, recorded per-request seeds; preserve input/output association.
        seed = options.pop('seed', None)
        seed = secrets.randbits(62) if seed is None else seed
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.chat, messages, seed=seed+i, **options)
                       for i, messages in enumerate(conversations)]
            results = [f.result() for f in futures]
        return dict(status='ok', model_id=self.model_id, batch_size=len(results), results=results)

    def __call__(self, prompt=None, references=None, *, messages=None,
                 max_new_tokens=1024, video_fps=None, video_max_frames=None,
                 max_pixels=512*512, seed=None, dry_run=False, system_prompt=None,
                 assistant_prefill=None, system_cache_path=None, output_dir=None,
                 temperature=0.8, top_p=0.95, top_k=50, response_schema=None, enable_thinking=None):
        from .worker import qwen_messages
        enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be bool")
        video_fps = self.video_fps if video_fps is None else video_fps
        video_max_frames = self.video_max_frames if video_max_frames is None else video_max_frames
        sampling = generation_options(dict(temperature=temperature, top_p=top_p, top_k=top_k))
        seed = secrets.randbits(63) if seed is None else seed
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Qwen seed must be an integer in [0, 2**63), or None')
        if type(max_new_tokens) is not int or max_new_tokens < 1 or video_fps <= 0 or video_max_frames < 2 or max_pixels < 1024:
            raise ValueError('Invalid Qwen token/frame/pixel budget')
        if system_cache_path:
            raise ValueError('Use vLLM automatic prefix caching instead of a native system_cache_path')
        if assistant_prefill:
            raise ValueError('This vLLM adapter uses explicit Instruct messages without assistant prefill')
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
        normalized = normalize_messages(messages, check_files=not dry_run, max_pixels=max_pixels,
            video_fps=video_fps, video_max_frames=video_max_frames)
        # One processor budget applies to this request. Different budgets for
        # different blocks remain available through the native official processor.
        for message in normalized:
            for item in message['content'] if isinstance(message['content'], list) else []:
                if item['type'] in ('image', 'video'):
                    allowed = {'type', item['type'], 'max_pixels'}
                    if item['type'] == 'video':
                        allowed.update(('fps', 'max_frames'))
                    if set(item) - allowed:
                        raise ValueError('Unsupported per-media serving options; use native Qwen for these processor controls')
                    if item.get('max_pixels') != max_pixels:
                        raise ValueError('Set media budgets on the vLLM server; per-media overrides are native-only')
                    if item['type'] == 'video' and (item.get('fps') != video_fps or item.get('max_frames') != video_max_frames):
                        raise ValueError('Set video sampling on the vLLM server; per-media overrides are native-only')
        body = dict(model=self.model_id, messages=serving_messages(normalized),
                    max_tokens=max_new_tokens, temperature=sampling['temperature'],
                    top_p=sampling['top_p'], top_k=sampling['top_k'], seed=seed, stream=False,
                    chat_template_kwargs=dict(enable_thinking=enable_thinking))
        kinds = {item['type'] for message in normalized
                 for item in (message['content'] if isinstance(message['content'], list) else [])}
        if kinds & {'image', 'video'}:
            body['mm_processor_kwargs'] = dict(max_pixels=max_pixels)
            if 'video' in kinds:
                if video_fps != self.video_fps or video_max_frames != self.video_max_frames:
                    raise ValueError('Video sampling is fixed at server launch in this deployment; match the client/server FPS and frame limit, or use native Qwen')
                body['mm_processor_kwargs'].update(fps=video_fps, do_sample_frames=False)
        if response_schema is not None:
            body['response_format'] = dict(type='json_schema',
                json_schema=dict(name='response', strict=True, schema=response_schema))
        if dry_run:
            return dict(status='planned', backend='vllm', request=body, bindings=bindings)
        directory = Path(output_dir) if output_dir is not None else self.output_dir
        directory = directory.expanduser().resolve() / (
            datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-') + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'request.json').write_text(json.dumps(body, ensure_ascii=False, indent=2))
        req = urllib.request.Request(self.base_url+'/chat/completions',
            data=json.dumps(body).encode('utf-8'), headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:2000]
            (directory/'error.txt').write_text(detail)
            raise RuntimeError(f'vLLM HTTP {exc.code}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            (directory/'error.txt').write_text(str(exc))
            raise RuntimeError(f'Cannot reach vLLM at {self.base_url}: {exc}') from exc
        (directory/'response.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        choice = raw['choices'][0]
        usage = raw.get('usage', {})
        result = dict(status='ok', text=choice['message'].get('content') or '',
            model_id=self.model_id, finish_reason=choice.get('finish_reason'),
            input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'),
            usage=usage, reasoning=choice['message'].get('reasoning') or choice['message'].get('reasoning_content'),
            enable_thinking=enable_thinking, job_dir=str(directory), request_path=str(directory/'request.json'), bindings=bindings)
        (directory/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
