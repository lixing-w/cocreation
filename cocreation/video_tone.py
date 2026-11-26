"""Root-anchored, clip-wide joint shadow/midtone/highlight matching.

No generation models or case-specific regions. Stable spatial tiles and chroma
agreement exclude many changed objects. A single monotone Y-only curve preserves
black/white endpoints; held-out frames gate application. Audio is stream-copied.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np

POLICY = 'root-joint-y-v1'
TONE_VIDEO_CRF = 9


def edit_source(task):
    """Editing has exactly one base video; new generation may have several refs."""
    refs = [r for r in task.get('references', []) if r.get('kind') == 'video']
    if len(refs) != 1 or not refs[0].get('placeholder'):
        raise ValueError('video_edit requires exactly one base video: '+str(task.get('placeholder', 'clip edit')))
    return dict(refs[0])


def brightness_anchor(task, tasks):
    """Follow edit/copy edges only; generation and assembly start fresh chains."""
    if task.get('operation') != 'video_edit':
        return None
    ref = edit_source(task)
    visited = set()
    while True:
        token = ref['placeholder']
        if token in visited:
            raise ValueError('Cycle in video brightness lineage: '+token)
        visited.add(token)
        parent = tasks[token]
        # A selection is a new clip boundary. An earlier edit's selection is
        # retained when walking from its descendants (legacy source-shot runs).
        if any(key in ref for key in ('shot_id', 'frame_seconds', 'crop_query')):
            break
        if parent['operation'] == 'video_edit':
            ref = edit_source(parent)
        elif parent['operation'] == 'copy' and not parent.get('clip_edits'):
            ref = dict(placeholder=parent['dependencies'][0], kind='video')
        else:
            break
    return {k: ref[k] for k in ('placeholder', 'kind', 'shot_id') if k in ref}


def _probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', str(path)]))


def _stream(path):
    info = _probe(path)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    return stream, float(stream.get('duration') or info['format']['duration'])


def _limits(stream):
    return (0, 255) if stream.get('color_range') == 'pc' or stream['pix_fmt'].startswith('yuvj') else (16, 235)


def _supported(stream):
    return (stream.get('pix_fmt') in ('yuv420p', 'yuv422p', 'yuv444p', 'yuvj420p', 'yuvj422p', 'yuvj444p')
            and stream.get('color_transfer') not in ('smpte2084', 'arib-std-b67'))


def sample_planes(path, *, size=None, count=24):
    """Bounded-memory samples at normalized clip times, in native YUV range."""
    stream, seconds = _stream(path)
    if size is None:
        width = min(256, stream['width']) // 2 * 2
        height = max(2, round(stream['height'] * width / stream['width'] / 2) * 2)
    else:
        width, height = size
    raw = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
        '-map', '0:v:0', '-vf', f'tpad=stop_mode=clone:stop_duration=1,fps={count/seconds:.12f}:start_time=0:eof_action=pass,scale={width}:{height}:flags=area,format=yuv444p',
        '-frames:v', str(count), '-f', 'rawvideo', '-'])
    planes = np.frombuffer(raw, np.uint8).reshape(-1, 3, height, width).astype(np.float32)
    lo, hi = _limits(stream)
    planes[:, 0] = (planes[:, 0] - lo) * (255 / (hi - lo))
    # Chroma is compared only to reject changed content, never corrected.
    if lo == 16:
        planes[:, 1:] = (planes[:, 1:] - 128) * (255 / 224) + 128
    return planes, stream


def stable_mask(anchor, target):
    """Keep tone-consistent tiles, then reject changed silhouettes inside them."""
    a, t = anchor[:, 0], target[:, 0]
    chroma = np.max(np.abs(anchor[:, 1:] - target[:, 1:]), axis=1)
    result = np.zeros(a.shape, bool)
    for y in range(0, a.shape[1], 24):
        for x in range(0, a.shape[2], 24):
            region = (slice(None), slice(y, y+24), slice(x, x+24))
            aa, tt = a[region], t[region]
            delta = tt - aa
            offset = np.median(delta)
            residual = np.abs(delta - offset)
            c = chroma[region]
            if np.median(c) > 14 or np.median(residual) > 12:
                continue
            # A flat background has no useful correlation. Textured regions
            # must retain structure; subtracting means ignores exposure shifts.
            av, tv = aa - aa.mean(), tt - tt.mean()
            denom = float(np.sqrt((av*av).sum() * (tv*tv).sum()))
            corr = float((av*tv).sum() / denom) if denom > 1 else 0
            if min(float(aa.std()), float(tt.std())) > 5 and corr < .45:
                continue
            result[region] = (c < 22) & (residual < 20)
    return result


def _groups(anchor_y, mask):
    # Equal tone-band weights prevent large grey backgrounds from dominating.
    return [mask & (anchor_y >= lo) & (anchor_y < hi)
            for lo, hi in ((-256, 85), (85, 170), (170, 512))]


def _isotonic(points, black, white):
    xs = sorted({x for x, y, w in points})
    blocks = []
    for i, x in enumerate(xs):
        group = [(y, w) for px, y, w in points if px == x]
        blocks.append([i, i, sum(y*w for y, w in group), sum(w for y, w in group)])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append([a[0], b[1], a[2]+b[2], a[3]+b[3]])
    ys = np.zeros(len(xs))
    for lo, hi, total, weight in blocks:
        ys[lo:hi+1] = total/weight
    values = np.arange(256)
    lut = np.rint(np.interp(values, xs, ys)).clip(black, white)
    lut[values <= black] = values[values <= black]
    lut[values >= white] = values[values >= white]
    return lut.astype(np.uint8)


def joint_curve(anchor, target, *, black=16, white=235):
    """Fit on even samples; validate independent odd samples before applying."""
    train_a, train_t = anchor[::2], target[::2]
    mask = stable_mask(train_a, train_t)
    report = dict(stable_fraction=float(mask.mean()), sample_count=len(anchor))
    if mask.sum() < 512 or mask.mean() < .02:
        return None, dict(report, reason='insufficient_stable_content')
    a, t = train_a[:, 0], train_t[:, 0]
    if np.percentile(a[mask], 95) - np.percentile(a[mask], 5) < 16:
        return None, dict(report, reason='insufficient_tonal_range')
    points = [(float(black), float(black), 1e6), (float(white), float(white), 1e6)]
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    groups = _groups(a, mask)
    for region, weight in [(g, 1.) for g in groups] + [(mask, .25)]:
        if region.sum() < 128:
            continue
        src, ref = np.percentile(t[region], qs), np.percentile(a[region], qs)
        for x, y in zip(src, ref):
            x, y = black+x*(white-black)/255, black+y*(white-black)/255
            if black < x < white:
                points.append((float(x), float(np.clip(y, black, white)), weight))
    lut = _isotonic(points, black, white)
    hold_a, hold_t = anchor[1::2], target[1::2]
    hold_mask = stable_mask(hold_a, hold_t)
    if hold_mask.sum() < 512:
        return None, dict(report, reason='insufficient_validation_content')
    values = np.rint(black+hold_t[:, 0]*(white-black)/255).clip(0, 255).astype(np.uint8)
    mapped = (lut[values].astype(float)-black)*255/(white-black)
    before, after = [], []
    for region in _groups(hold_a[:, 0], hold_mask):
        if region.sum() < 128:
            continue
        ref = np.percentile(hold_a[:, 0][region], qs)
        before.append(float(np.abs(np.percentile(hold_t[:, 0][region], qs)-ref).mean()))
        after.append(float(np.abs(np.percentile(mapped[region], qs)-ref).mean()))
    report.update(validation_before=before, validation_after=after,
                  balanced_mae_before=float(np.mean(before)), balanced_mae_after=float(np.mean(after)))
    if not before or np.mean(before) < 1 or np.mean(after) >= np.mean(before)*.98:
        return None, dict(report, reason='no_validated_improvement')
    if any(b > a+3 for a, b in zip(before, after)):
        return None, dict(report, reason='tone_band_regression')
    return lut, dict(report, reason='validated', lut=lut.tolist())


def render_curve(source, destination, lut):
    """Apply directly in YUV; preserve input timestamps, chroma and audio packets."""
    stream, _ = _stream(source)
    # LUT expressions are evaluated only 256 times by ffmpeg's lut filter.
    terms = ['val'] + [f'eq(val,{i})*({int(v)-i})' for i, v in enumerate(lut) if int(v) != i]
    # A balanced expression avoids FFmpeg's parser depth limit for 256 entries.
    while len(terms) > 1:
        terms = ['('+terms[i]+'+'+terms[i+1]+')' if i+1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    expression = terms[0]
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-copyts', '-i', str(source),
        '-map', '0:v:0', '-map', '0:a?', '-map_metadata', '0', '-vf', f"lut=y='{expression}'",
        '-c:v', 'libx264', '-crf', str(TONE_VIDEO_CRF), '-preset', 'medium', '-pix_fmt', stream['pix_fmt'],
        '-fps_mode', 'passthrough', '-enc_time_base:v', '-1', '-c:a', 'copy', '-avoid_negative_ts', 'disabled']
    for key, option in (('color_range', '-color_range'), ('color_space', '-colorspace'),
                        ('color_transfer', '-color_trc'), ('color_primaries', '-color_primaries')):
        value = stream.get(key)
        if value and value not in ('unknown', 'unspecified'):
            command += [option, value]
    command += ['-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)


def match_video(source, anchor, destination):
    """Write a corrected video only if validation succeeds; otherwise leave it alone."""
    stream, _ = _stream(source)
    anchor_stream, _ = _stream(anchor)
    base = dict(policy=POLICY, status='skipped')
    if not _supported(stream) or not _supported(anchor_stream):
        return dict(base, reason='unsupported_pixel_format_or_hdr')
    for key in ('color_space', 'color_transfer', 'color_primaries'):
        a, t = anchor_stream.get(key), stream.get(key)
        if a and t and a != 'unknown' and t != 'unknown' and a != t:
            return dict(base, reason='different_color_encoding')
    if abs((stream['width']/stream['height'])/(anchor_stream['width']/anchor_stream['height'])-1) > .02:
        return dict(base, reason='different_framing')
    target, _ = sample_planes(source)
    ref, _ = sample_planes(anchor, size=(target.shape[3], target.shape[2]))
    n = min(len(ref), len(target))
    if n < 4:
        return dict(base, reason='insufficient_frames')
    lut, report = joint_curve(ref[:n], target[:n], black=_limits(stream)[0], white=_limits(stream)[1])
    if lut is None:
        return dict(base, **report)
    render_curve(source, destination, lut)
    return dict(base, **report, status='matched', video_crf=TONE_VIDEO_CRF)
