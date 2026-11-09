"""Find a visually matching frame. Returns seconds from first displayed frame, or None."""
import argparse
import bisect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import datetime
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocreation import Qwen, Reference
from cocreation.storage import scratch_root, ensure_scratch
from tools.find_video_frame_prompts import localization_prompt, selection_prompt


def frame_timestamps(path):
    """Read presentation timestamps in decoded frame order (including VFR)."""
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)], text=True)
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


def extract_frames(path, indices, directory):
    directory.mkdir(parents=True, exist_ok=False)
    expression = '+'.join(f'eq(n\\,{i})' for i in indices)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-vf', 'select=' + expression, '-vsync', '0', '-frames:v', str(len(indices)),
        str(directory / '%06d.png')], check=True)
    files = sorted(directory.glob('*.png'))
    if len(files) != len(indices):
        raise RuntimeError('Decoded frame count did not match timestamp index')
    return files


def parse_object(text, field):
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {field}:
        raise ValueError(f'Expected JSON with exactly {field}')
    return result[field]


def parse_selection(text, count):
    value = parse_object(text, 'frame_id')
    if value is not None and (type(value) is not int or not 0 <= value < count):
        raise ValueError('Model returned an invalid candidate frame ID')
    return value


def parse_location(text, last_timestamp):
    value = parse_object(text, 'timestamp_seconds')
    if value is not None and (type(value) not in (float, int) or not math.isfinite(value)
                              or not 0 <= value <= last_timestamp + 0.1):
        raise ValueError('Model returned an invalid video timestamp')
    # Official temporal labels are rounded to tenths; allow only that endpoint rounding.
    return None if value is None else min(float(value), last_timestamp)


def find_video_frame(video_path, description, *, qwen=None, coarse_frames=24,
                     refine_frames=16, video_fps=2.0, output_dir=None, frame_output=None):
    """Return float seconds or None. Optional frame_output exports the exact selected PNG.

    qwen may be a shared persistent Qwen object; this function never closes a supplied
    instance. A locally created instance is closed on exit. The coarse pass uses the official Qwen video preprocessing path;
    None means not found among inspected frames.
    Operational/model-output failures raise instead of masquerading as not found.
    """
    if not isinstance(description, str) or not description.strip():
        raise ValueError('description must be nonempty')
    for value in (coarse_frames, refine_frames):
        if type(value) is not int or not 2 <= value <= 64:
            raise ValueError('Frame budgets must be integers between 2 and 64')
    if isinstance(video_fps, bool) or not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError('video_fps must be finite and positive')
    path = Path(video_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('video_path must be a local video file')
    for binary in ('ffprobe', 'ffmpeg'):
        if shutil.which(binary) is None:
            raise RuntimeError('Install ffmpeg and ffprobe before frame search')
    export = Path(frame_output).expanduser().absolute() if frame_output is not None else None
    if export is not None and export.exists():
        raise FileExistsError(f'Refusing to overwrite frame output: {export}')
    if output_dir is None:
        ensure_scratch()
        output_dir = scratch_root() / 'runs' / 'find_video_frame'
    now = datetime.datetime.now()
    formatted_time = now.strftime("%y-%m-%d-%H-%M-%S-") + now.strftime("%f")[:3]
    # e.g. 26-11-08-17-29-43-823
    directory = Path(output_dir).expanduser().resolve() / (formatted_time + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    timestamps = frame_timestamps(path)
    owned = qwen is None
    model = Qwen(output_dir=directory / 'qwen') if owned else qwen
    rounds = []
    def choose(indices, name):
        files = extract_frames(path, indices, directory / name)
        refs = [Reference(f'frame_{i}', image, 'image') for i, image in enumerate(files)]
        response = model(selection_prompt(description, len(indices)), refs, max_new_tokens=128)
        selected = parse_selection(response['text'], len(indices))
        rounds.append({'indices': indices, 'timestamps': [timestamps[i] for i in indices],
                       'selected': selected, 'request_path': response.get('request_path')})
        return selected, files
    try:
        # Pass the actual video through qwen-vl-utils + Qwen3VLProcessor.
        # Do not impersonate its video token/temporal-patch format using image inputs.
        response = model(localization_prompt(description),
                         [Reference('source_video', path, 'video')],
                         video_fps=video_fps, video_max_frames=coarse_frames, max_new_tokens=128)
        location = parse_location(response['text'], timestamps[-1])
        rounds.append({'mode': 'official_video', 'timestamp_seconds': location,
                       'video_fps': video_fps, 'max_frames': coarse_frames,
                       'request_path': response.get('request_path')})
        indices, files, selected = [], [], None
        if location is not None:
            # Temporal patch timestamps describe groups of sampled frames, not one exact
            # original frame. Search both sides, then ground the final answer in PTS.
            effective_cap = max(2, coarse_frames // 2 * 2)
            radius = max(1.0, 2.0/video_fps, 2.0*timestamps[-1]/(effective_cap-1))
            lower = max(0, bisect.bisect_left(timestamps, location-radius))
            upper = min(len(timestamps)-1, bisect.bisect_right(timestamps, location+radius)-1)
            nearest = min(range(lower, upper+1), key=lambda i: abs(timestamps[i]-location))
            indices = sorted(set(sample_indices(timestamps, refine_frames, lower, upper)) | {nearest})
            selected, files = choose(indices, 'refine')
        timestamp = None if selected is None else timestamps[indices[selected]]
        selected_path = None
        if selected is not None:
            selected_path = directory / 'selected.png'
            shutil.copy2(files[selected], selected_path)
            if export is not None:
                export.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects against concurrent exporters.
                with export.open('xb') as dest, selected_path.open('rb') as source:
                    shutil.copyfileobj(source, dest)
        result = dict(video_path=str(path), description=description, timestamp_seconds=timestamp,
                      source_frame_index=None if selected is None else indices[selected],
                      frame_path=str(selected_path) if selected_path else None,
                      timestamp_origin='first displayed frame', rounds=rounds)
        (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return timestamp
    finally:
        if owned:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('description', help='Visible properties/state to find, optionally with event context')
    parser.add_argument('--frame-output', type=Path, help='Optional path to export the exact selected PNG frame')
    parser.add_argument('--output-dir', type=Path, help='Optional root directory for this run')
    parser.add_argument('--coarse-frames', type=int, default=24, help='Official video sampling frame cap')
    parser.add_argument('--video-fps', type=float, default=2.0, help='Video FPS for coarse localization')
    parser.add_argument('--refine-frames', type=int, default=16, help='Number of frames to refine the search')
    args = parser.parse_args()
    print(json.dumps(find_video_frame(**vars(args))))


if __name__ == '__main__':
    main()
