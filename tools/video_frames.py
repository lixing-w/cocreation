"""Presentation-time video sampling and exact source frame extraction."""
from bisect import bisect_left
import json
import math
import subprocess

def frame_timestamps(path, *, timeout=None):
    """Read presentation timestamps in decoded frame order (including VFR)."""
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)],
        text=True, timeout=timeout)
    frames = json.loads(raw).get('frames', [])
    if not frames:
        raise ValueError('Video has no decodable frames')
    try:
        values = [float(frame['best_effort_timestamp_time']) for frame in frames]
    except (KeyError, ValueError) as error:
        raise ValueError('Video is missing frame presentation timestamps') from error
    if any(not math.isfinite(x) for x in values) or any(a > b for a, b in zip(values, values[1:])):
        raise ValueError('Video has invalid presentation timestamps')
    return [x-values[0] for x in values]


def sample_indices(timestamps, count, lower=0, upper=None):
    """Uniform time coverage; return actual source indices, including endpoints."""
    import bisect
    upper = len(timestamps)-1 if upper is None else upper
    if upper-lower+1 <= count:
        return list(range(lower, upper+1))
    start, end = timestamps[lower], timestamps[upper]
    result = {lower, upper}
    for i in range(count):
        target = start + (end-start)*i/(count-1)
        index = min(upper, bisect.bisect_left(timestamps, target, lower, upper+1))
        if index > lower and abs(timestamps[index-1]-target) < abs(timestamps[index]-target):
            index -= 1
        result.add(index)
    return sorted(result)


def rate_indices(timestamps, fps):
    """Take the nearest actual frames at 0, 1/fps, 2/fps, ... ."""
    indices = []
    for tick in range(math.floor(timestamps[-1] * fps + 1e-6) + 1):
        target = tick / fps
        right = min(bisect_left(timestamps, target), len(timestamps)-1)
        index = right
        if right and target - timestamps[right-1] < abs(timestamps[right] - target):
            index = right - 1
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def extract_frames(path, indices, directory, *, timeout=None):
    directory.mkdir(parents=True, exist_ok=False)
    expression = '+'.join(f'eq(n\\,{i})' for i in indices)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-vf', 'select=' + expression, '-vsync', '0', '-frames:v', str(len(indices)),
        str(directory / '%06d.png')], check=True, timeout=timeout)
    files = sorted(directory.glob('*.png'))
    if len(files) != len(indices):
        raise RuntimeError('Decoded frame count did not match timestamp index')
    return files

