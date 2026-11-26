"""Observation-anchored frame cuts and deterministic shot assembly (ffmpeg only)."""
import bisect
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from tools.video_frames import frame_timestamps

EXPORT_VIDEO_CRF = '11'


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))


def _destination(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def refine_shots(video, observation, *, window=0.75, threshold=0.25, output_path=None):
    """Search only near each logical observation boundary; never renumber shots.

    Missing cuts fail explicitly. Candidates, errors and actual source frame
    indices are returned for audit; visual semantics still come from observation.
    """
    if not math.isfinite(window) or window <= 0 or not 0 < threshold < 1:
        raise ValueError('Invalid shot refinement window/threshold')
    data = probe(video)
    video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
    stamps = frame_timestamps(video)
    step = stamps[-1]-stamps[-2] if len(stamps)>1 else 1/24
    duration = float(video_stream.get('duration', data['format'].get('duration', stamps[-1]+step)))
    shots = sorted(observation['shots'], key=lambda s: s.get('start', float('inf')))
    if not shots:
        raise ValueError('Observation has no shots')
    for shot in shots:
        start, end = shot.get('start'), shot.get('end')
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,end)) or not 0 <= start < end <= duration+window:
            raise ValueError('Shot extraction requires finite start/end times in observation: ' + shot['id'])
    if abs(shots[0]['start']) > window or abs(shots[-1]['end']-duration) > window:
        raise ValueError('Observation shot ranges must cover the full target video')
    coarse = [(a['end']+b['start'])/2 for a,b in zip(shots,shots[1:])]
    for a,b in zip(shots,shots[1:]):
        if abs(a['end']-b['start']) > window or a['start'] >= b['start'] or a['end'] >= b['end']:
            raise ValueError('Observation shots must form one ordered contiguous timeline')
    indices, boundaries = [0], []
    for i, expected in enumerate(coarse):
        left = coarse[i-1] if i else 0
        right = coarse[i+1] if i+1<len(coarse) else duration
        radius = min(window, (expected-left)/2, (right-expected)/2)
        low, high = max(0, expected-radius), min(duration, expected+radius)
        # Decode one extra frame on the left to score the first candidate.
        seek = max(0, low-step*2)
        result = subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','info',
            '-ss',str(seek),'-i',str(video),'-t',str(high-seek+step),
            '-vf',f"select='gt(scene,{threshold})',showinfo",'-an','-fps_mode','vfr','-f','null','-'],
            capture_output=True,text=True,check=True)
        candidates = [seek+float(x) for x in re.findall(r'pts_time:([0-9.eE+\-]+)',result.stderr)]
        candidates = [x for x in candidates if low <= x <= high]
        if not candidates:
            raise ValueError(f'No detected cut near observation boundary {expected:.3f}s ({shots[i]["id"]} -> {shots[i+1]["id"]}); inspect the observation or adjust window/threshold')
        actual = min(candidates, key=lambda x: abs(x-expected))
        index = min(range(max(0,bisect.bisect_left(stamps,actual)-1),min(len(stamps),bisect.bisect_left(stamps,actual)+2)),key=lambda n:abs(stamps[n]-actual))
        if not indices[-1] < index < len(stamps):
            raise ValueError('Refined shot boundaries overlap or produce an empty shot')
        indices.append(index)
        boundaries.append(dict(observed_seconds=expected, actual_seconds=stamps[index], frame_index=index,
                               candidates=candidates, error_seconds=stamps[index]-expected))
    indices.append(len(stamps))
    mapping = {shot['id']: dict(start_frame=indices[i],end_frame=indices[i+1],
        start_seconds=stamps[indices[i]], end_seconds=stamps[indices[i+1]] if indices[i+1]<len(stamps) else duration,
        description=shot['description']) for i,shot in enumerate(shots)}
    result = dict(video=str(Path(video).resolve()),frame_count=len(stamps),duration=duration,
                  shot_order=[s['id'] for s in shots],boundaries=boundaries,shots=mapping)
    if output_path:
        Path(output_path).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def extract_shot(video, shot, destination):
    """Decode/re-encode exact frame intervals; stream-copy would snap to keyframes."""
    dest = _destination(destination)
    has_audio = any(s['codec_type']=='audio' for s in probe(video)['streams'])
    filters = f"[0:v:0]trim=start_frame={shot['start_frame']}:end_frame={shot['end_frame']},setpts=PTS-STARTPTS[v]"
    if has_audio:
        filters += f";[0:a:0]atrim=start={shot['start_seconds']}:end={shot['end_seconds']},asetpts=PTS-STARTPTS[a]"
    command = ['ffmpeg','-nostdin','-v','error','-i',str(video),'-filter_complex',filters,'-map','[v]']
    if has_audio:
        command += ['-map','[a]','-c:a','aac']
    subprocess.run(command+['-c:v','libx264','-crf',EXPORT_VIDEO_CRF,'-pix_fmt','yuv420p','-fps_mode','vfr',str(dest)],check=True)
    return dest


def extract_frame(video, seconds, destination):
    dest = _destination(destination)
    stamps = frame_timestamps(video)
    if not math.isfinite(seconds) or not 0 <= seconds <= stamps[-1]:
        raise ValueError('Frame timestamp outside the target video')
    index = min(range(len(stamps)), key=lambda i: abs(stamps[i]-seconds))
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-vf',f"select='eq(n,{index})'",
                    '-frames:v','1','-fps_mode','vfr',str(dest)],check=True)
    return dest


def concatenate(clips, destination, *, size=None):
    """Exact input order, with synchronized audio; normalize timebase, not content."""
    if not clips:
        raise ValueError('Cannot assemble an empty clip list')
    dest = _destination(destination)
    metadata = [probe(path) for path in clips]
    sizes = {(v['width'],v['height']) for m in metadata for v in m['streams'] if v['codec_type']=='video'}
    if size is None and len(sizes) != 1:
        raise ValueError('Assembly clips need matching image dimensions')
    command = ['ffmpeg','-nostdin','-v','error']
    for path in clips:
        command += ['-i',str(path)]
    audio = any(any(s['codec_type']=='audio' for s in m['streams']) for m in metadata)
    filters, inputs = [], ''
    for i,m in enumerate(metadata):
        canvas = (f'scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                  f'pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2,' if size else '')
        filters.append(f'[{i}:v:0]{canvas}settb=AVTB,setpts=PTS-STARTPTS,setsar=1[v{i}]')
        inputs += f'[v{i}]'
        if audio:
            if any(s['codec_type']=='audio' for s in m['streams']):
                filters.append(f'[{i}:a:0]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]')
            else:
                duration = float(m['format']['duration'])
                filters.append(f'anullsrc=r=48000:cl=stereo,atrim=duration={duration}[a{i}]')
            inputs += f'[a{i}]'
    filters.append(inputs+f'concat=n={len(clips)}:v=1:a={int(audio)}[v]'+('[a]' if audio else ''))
    command += ['-filter_complex',';'.join(filters),'-map','[v]']
    if audio:
        command += ['-map','[a]','-c:a','aac']
    subprocess.run(command+['-c:v','libx264','-crf',EXPORT_VIDEO_CRF,'-pix_fmt','yuv420p','-fps_mode','vfr',str(dest)],check=True)
    return dest


def prepare_inputs(task, resolved, observation, output_dir, *, shot_map=None, crop_resolver=None):
    """Materialize the SAME ordered slots whose labels appear in generation_prompt.

    Accepts a compact placeholders.json entry or a full generation task.
    resolved maps placeholder -> local file. Returns (Reference list, prompt).
    This is the media boundary: the prompt rewriter never calls this function.
    """
    from cocreation import Reference
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    references = []
    for i,ref in enumerate(task.get('references', [])):
        if task.get('preprocess') and not ref.get('placeholder'):
            path = concatenate([resolved[c['placeholder']] for c in task['clips']],directory/f'assembly-{i}.mp4')
        else:
            path = Path(resolved[ref['placeholder']])
            if 'crop_query' in ref:
                if crop_resolver is None:
                    from tools.find_crop_in_video import find_crop_in_video
                    crop_resolver = lambda source, query, output: find_crop_in_video(source, query, crop_output=output)['crop_path']
                path = Path(crop_resolver(path, ref['crop_query'], directory/f'crop-{i}.png'))
            elif 'frame_seconds' in ref:
                path = extract_frame(path,ref['frame_seconds'],directory/f'frame-{i}.png')
            elif 'shot_id' in ref:
                mapping = shot_map or refine_shots(path,observation)
                path = extract_shot(path,mapping['shots'][ref['shot_id']],directory/f'shot-{i}.mp4')
        references.append(Reference(f'reference_{i+1}',path,ref['kind']))
    return references, task['prompt']


def edit_tool_call(task, resolved, destination):
    """The actual Python function and arguments for deterministic assembly."""
    if task['operation']=='concatenate':
        return dict(name='tools.video_editing.concatenate', arguments=dict(
            clips=[str(resolved[clip['placeholder']]) for clip in task['clips']],
            destination=str(destination),size=task.get('output_size')))
    if task['operation']=='copy':
        dependencies=task.get('dependencies',task.get('reference_materials',[]))
        return dict(name='shutil.copy2',arguments=dict(src=str(resolved[dependencies[0]]),dst=str(destination)))
    raise ValueError('Not an assembly/copy operation: '+task['operation'])


def execute_edit(task, resolved, observation, destination, *, shot_map=None):
    """Execute source/copy/extract/assemble contracts without a generative model."""
    operation = task['operation']
    dependencies = task.get('dependencies', task.get('reference_materials', []))
    if operation == 'source':
        return Path(task['source_video']).expanduser().resolve(strict=True)
    if operation == 'copy':
        source = resolved[dependencies[0]]
        dest = _destination(destination)
        shutil.copy2(source,dest)
        return dest
    if operation == 'extract_frame':
        return extract_frame(resolved[dependencies[0]],task['frame_seconds'],destination)
    if operation == 'concatenate':
        return concatenate(**edit_tool_call(task,resolved,destination)['arguments'])
    if operation == 'extract_shot':
        source = resolved[dependencies[0]]
        mapping = shot_map or refine_shots(source,observation)
        return extract_shot(source,mapping['shots'][task['shot_id']],destination)
    raise ValueError('This contract requires a generation expert: ' + operation)
