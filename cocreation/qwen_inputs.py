"""Ordered, JSON-serializable Qwen messages shared by native and serving adapters."""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlparse, unquote


def _media_path(value, check_files):
    value = str(value) if isinstance(value, Path) else value
    if not isinstance(value, str) or not value:
        raise ValueError('Media must be a path, URL, or data URI')
    parsed = urlparse(value)
    if parsed.scheme in ('http', 'https', 'data'):
        return value
    if parsed.scheme not in ('', 'file'):
        raise ValueError('Unsupported media URI scheme: ' + parsed.scheme)
    path = Path(unquote(parsed.path) if parsed.scheme == 'file' else value).expanduser().resolve()
    if check_files and not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def normalize_messages(messages, *, check_files=True, max_pixels=512*512,
                       video_fps=1.0, video_max_frames=16):
    """Preserve turns/content order; leave encoding and positions to Qwen's processor."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError('messages must be a nonempty list of conversations turns')
    result = deepcopy(list(messages))
    for message in result:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Each message requires a supported role')
        content = message.get('content')
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError('Message content must be text or an ordered list of content blocks')
        for item in content:
            if not isinstance(item, dict):
                raise ValueError('Content blocks must be dictionaries')
            kind = item.get('type')
            if kind == 'text':
                if not isinstance(item.get('text'), str):
                    raise ValueError('Text blocks require a text string')
            elif kind in ('image', 'video'):
                source = item.get(kind)
                if kind == 'video' and isinstance(source, (list, tuple)):
                    if not source:
                        raise ValueError('Video frame lists must not be empty')
                    item[kind] = [_media_path(p, check_files) for p in source]
                else:
                    item[kind] = _media_path(source, check_files)
                item.setdefault('max_pixels', max_pixels)
                if kind == 'video':
                    item.setdefault('fps', video_fps)
                    item.setdefault('max_frames', video_max_frames)
            else:
                raise ValueError('Qwen accepts text, image and video content blocks')
    json.dumps(result)
    return result
