"""Explicit H3 inference recipes by operation; no model imports."""
from .h3 import default_steps


def inference_recipe(operation, *, steps=None, turbo=False, new_turbo=False,
                     new_steps=None, edit_steps=None, edit_sampler='euler'):
    if type(turbo) is not bool or type(new_turbo) is not bool:
        raise ValueError('H3 Turbo switches must be booleans')
    if turbo and new_turbo:
        raise ValueError('Choose all-video Turbo or new-video-only Turbo, not both')
    for value in (steps, new_steps, edit_steps):
        if value is not None and (type(value) is not int or not 1 <= value <= 10000):
            raise ValueError('H3 steps must be integer values in 1–10000')
    if edit_sampler not in ('euler', 'res_multistep'):
        raise ValueError('H3 edit sampler must be euler or res_multistep')
    editing = operation == 'video_edit'
    accelerated = turbo or (new_turbo and not editing)
    override = edit_steps if editing else new_steps
    count = override if override is not None else steps if steps is not None else default_steps(accelerated)
    return dict(turbo=accelerated, steps=count,
                sampler='turbo_custom' if accelerated else edit_sampler if editing else 'euler',
                scheduler='simple', shift_video=12., shift_audio=3.)
