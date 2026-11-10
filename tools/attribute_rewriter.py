"""Rewrite sampled prose with media-grounded clarification and creative freedom."""
import sys
from tools._attribute_vlm import ask_text, reference
from tools._attribute_protocol import validate_sources
from tools.attribute_rewriter_prompts import SYSTEM


def attribute_rewriter(description, references=None, *, kind, brief='', qwen=None,
                       output_dir=None, seed=0, **options):
    if kind not in ('image', 'video'):
        raise ValueError('Expected image or video target')
    refs = [reference(r) for r in references or []]
    description = validate_sources(description, refs)
    task = f'Target: {kind}.\nCreative brief: {brief or "Open creative direction."}\nSampled description:\n{description}'
    result = ask_text(SYSTEM, task, refs, qwen=qwen, output_dir=output_dir, seed=seed, **options)
    if result is None:
        return None
    try:
        return validate_sources(result, refs)
    except ValueError as error:
        print(f'attribute_rewriter: {error}', file=sys.stderr)
        return None
