"""Choose the best available view of a known identity and crop its bounding box."""
from __future__ import annotations
import argparse
import bisect
import json
import math
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.video_frames import frame_timestamps, sample_indices, extract_frames


def _json(text):
    lines = text.strip().splitlines()
    if lines and lines[0].startswith('```') and lines[-1] == '```':
        text = '\n'.join(lines[1:-1])
    return json.loads(text)


def find_crop_in_video(video_path, description, *, qwen=None, output_dir=None,
                       crop_output=None, frame_output=None, video_fps=4.,
                       coarse_frames=32, refine_frames=16, batch_frames=16,
                       max_side=768, hint_timestamps=None):
    """Rank EVERY sample batch, compare winners globally, refine, then locate.

    This is best-match retrieval for a planner-provided identity, not a presence
    classifier: there is no NOT_FOUND branch. Malformed model coordinates remain
    genuine technical errors. Cropping uses the original decoded resolution.
    """
    from PIL import Image
    from cocreation import Qwen, VLLMQwen, Reference
    if not isinstance(description, str) or not description.strip():
        raise ValueError('description must be nonempty')
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError('video_fps must be positive')
    if any(type(v) is not int or v < 2 or v > 128 for v in (coarse_frames, refine_frames)):
        raise ValueError('Frame budgets must be integers in [2,128]')
    if type(batch_frames) is not int or not 2 <= batch_frames <= 32:
        raise ValueError('batch_frames must be in [2,32]')
    if type(max_side) is not int or not 128 <= max_side <= 2048:
        raise ValueError('max_side must be in [128,2048]')
    source = Path(video_path).expanduser().resolve(strict=True)
    for output in (crop_output, frame_output):
        if output is not None and Path(output).exists():
            raise FileExistsError(output)
    base = Path(output_dir) if output_dir else source.parent/'_crop_search'
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='crop-', dir=base))
    timestamps = frame_timestamps(source)
    hints = list(hint_timestamps or [])
    if any(type(t) not in (int,float) or not math.isfinite(t) or not 0 <= t <= timestamps[-1]+.1 for t in hints):
        raise ValueError('Invalid hint timestamp')
    count = min(len(timestamps), max(coarse_frames, math.ceil(timestamps[-1]*video_fps)+1))
    coarse = set(sample_indices(timestamps, max(2,count)))
    coarse.update(min(range(len(timestamps)),key=lambda i:abs(timestamps[i]-t)) for t in hints)
    images, rounds = {}, []
    model = qwen
    owned = model is None
    if owned:
        model = (VLLMQwen if sys.platform.startswith('linux') else Qwen)(output_dir=directory/'qwen')

    def candidates(indices):
        needed = sorted(set(indices)-set(images))
        if needed:
            frames = extract_frames(source, needed, directory/f'decode-{len(images)}')
            for index, frame in zip(needed, frames):
                path = directory/f'preview-{index}.jpg'
                with Image.open(frame) as image:
                    image.thumbnail((max_side,max_side))
                    image.convert('RGB').save(path, quality=92)
                images[index] = path
                frame.unlink()
            frames[0].parent.rmdir()

    def choose(indices, stage):
        if len(indices) == 1:
            return indices[0]
        candidates(indices)
        refs = [Reference(f'candidate_{i}', images[index], 'image') for i,index in enumerate(indices)]
        prompt = ('Choose the single best frame showing this identity: '+description+
            '. Prefer the clearest, largest, least occluded complete view; match its identity, not nearby objects. '
            'The supplied images are numbered from 0 in order. Always choose the best available frame. '
            'Return only JSON {"index": N}.')
        schema = dict(type='object',properties=dict(index=dict(type='integer',minimum=0,maximum=len(indices)-1)),
            required=['index'],additionalProperties=False)
        response = model(prompt,refs,max_new_tokens=64,temperature=.1,response_schema=schema,max_pixels=max_side*max_side)
        selected = _json(response['text'])['index']
        if type(selected) is not int or not 0 <= selected < len(indices):
            raise ValueError('Invalid selected frame index')
        rounds.append(dict(stage=stage,indices=indices,selected_source_index=indices[selected]))
        return indices[selected]

    def tournament(indices, stage):
        values = sorted(set(indices))
        while len(values) > batch_frames:
            values = [choose(values[i:i+batch_frames],stage) for i in range(0,len(values),batch_frames)]
        return choose(values,stage+'-global')

    try:
        anchor = tournament(coarse,'coarse')
        radius = max(1/video_fps,.1)
        lower = max(0,bisect.bisect_left(timestamps,timestamps[anchor]-radius))
        upper = min(len(timestamps)-1,bisect.bisect_right(timestamps,timestamps[anchor]+radius)-1)
        fine = set(sample_indices(timestamps,refine_frames,lower,upper)) | {anchor}
        selected = tournament(fine,'refine')
        frame = extract_frames(source,[selected],directory/'selected_source')[0]
        with Image.open(frame) as original:
            width,height = original.size
            # A square localization view avoids aspect-ratio coordinate drift.
            # Only the VLM view is resized; the output always crops the original.
            preview = original.resize((1000,1000))
            view = directory/'bbox-input.png';preview.save(view)
            schema = dict(type='object',properties=dict(bbox_2d=dict(type='array',minItems=4,maxItems=4,
                items=dict(type='integer',minimum=0,maximum=1000))),required=['bbox_2d'],additionalProperties=False)
            locator = description
            if any(word in description.lower() for word in ('bag', 'satchel', 'purse', 'backpack')):
                locator += ', specifically the main bag body and flap, excluding the wearer and long shoulder strap'
            prompt = ('Locate '+locator+'. Output one tight bounding box using JSON format: '
                '{"bbox_2d": [left, top, right, bottom]}. Coordinates are normalized to 0..1000 '
                'relative to the entire square image. Include the complete main subject. Always return the best available box.')
            box = None
            for attempt in range(3):
                response = model(prompt,[Reference('selected',view,'image')],max_new_tokens=128,
                    temperature=.1,response_schema=schema,seed=attempt,max_pixels=1024*1024)
                box = _json(response['text']).get('bbox_2d')
                if (isinstance(box,list) and len(box)==4 and all(type(v) is int and 0<=v<=1000 for v in box)
                        and box[0]<box[2] and box[1]<box[3]):
                    break
            else:
                raise ValueError('Model returned malformed bounding box coordinates')
            pixels = [max(0,math.floor(box[0]*width/1000)),max(0,math.floor(box[1]*height/1000)),
                      min(width,math.ceil(box[2]*width/1000)),min(height,math.ceil(box[3]*height/1000))]
            cropped = original.crop(pixels)
            output = Path(crop_output) if crop_output else directory/'crop.png'
            output.parent.mkdir(parents=True,exist_ok=True);cropped.save(output,format='PNG')
            if frame_output:
                target=Path(frame_output);target.parent.mkdir(parents=True,exist_ok=True);original.save(target,format='PNG')
        result = dict(version=1,video_path=str(source),query=description,
            timestamp_seconds=timestamps[selected],source_frame_index=selected,
            bbox_normalized=box,bbox_pixels=pixels,source_size=[width,height],bbox_view_size=[1000,1000],
            frame_path=str(frame),crop_path=str(output.resolve()),rounds=rounds)
        (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return result
    finally:
        if owned:
            model.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video_path',type=Path)
    parser.add_argument('description')
    parser.add_argument('--crop-output',type=Path)
    parser.add_argument('--frame-output',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--video-fps',type=float,default=4.)
    parser.add_argument('--coarse-frames',type=int,default=32)
    parser.add_argument('--refine-frames',type=int,default=16)
    parser.add_argument('--batch-frames',type=int,default=16)
    parser.add_argument('--max-side',type=int,default=768)
    args=parser.parse_args()
    print(json.dumps(find_crop_in_video(**vars(args)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
