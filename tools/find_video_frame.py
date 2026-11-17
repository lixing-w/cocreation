"""Compatibility alias; new callers should use find_crop_in_video."""
from tools.video_frames import frame_timestamps, sample_indices, extract_frames


def find_video_frame(video_path, description, **options):
    from tools.find_crop_in_video import find_crop_in_video
    options.pop('exhaustive', None)
    return find_crop_in_video(video_path,description,**options)['timestamp_seconds']
