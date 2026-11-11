"""Shared stochastic decoding options for ordinary and prefix-cached Qwen calls."""
import math


def generation_options(options):
    temperature = options.get('temperature', 0.8)
    top_p = options.get('top_p', 0.95)
    top_k = options.get('top_k', 50)
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature <= 0):
        raise ValueError('Qwen temperature must be finite and positive')
    if type(top_p) not in (int, float) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('Qwen top_p must be in (0, 1]')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('Qwen top_k must be a nonnegative integer')
    return dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
