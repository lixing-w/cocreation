"""Exact output ratios, separate from latent-model dimension constraints."""
from scripts.dialogue_generation.cross_turn_rule import output_aspect

DEFAULT_ASPECT_PRESET = 'square_assets_landscape_video'
SYSTEM_ASPECT_NOTE = 'The system default for aspect ratio is 1:1 for images and 16:9 landscape for videos.'


def aspect_ratio_rule(state, kind):
    """Describe the effective ratio and who set it for this output."""
    preset = state.get('aspect_ratio')
    return dict(ratio=output_aspect(preset or DEFAULT_ASPECT_PRESET, kind),
                source='user_override' if preset else 'system_default')


def geometry(state, kind, *, image=(1024, 1024), video=(1024, 576), output=(1280, 720), native_video_output=False):
    preset = state.get('aspect_ratio')
    if kind == 'video' and native_video_output:
        # Keep the model's native pixel grid, orienting it by the effective ratio.
        ratio = aspect_ratio_rule(state, kind)['ratio']
        landscape = (max(video), min(video))
        native = landscape[::-1] if ratio == '9:16' else landscape
        return native, native
    if preset is None:
        return tuple(image if kind == 'image' else video), tuple(image if kind == 'image' else output)
    ratio = output_aspect(preset, kind)
    # Native model sizes are exact ratios and divisible by 32 for both backends.
    native = {'1:1': (1024, 1024), '16:9': (1024, 576), '9:16': (576, 1024)}[ratio]
    final = native if kind == 'image' else {'1:1': (1024, 1024), '16:9': (1280, 720), '9:16': (720, 1280)}[ratio]
    return native, final
