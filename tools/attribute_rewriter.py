"""Explore a new identity or minimally repair conflicts across references."""
import sys
import random
from pathlib import Path
from tools._attribute_vlm import ask_text, reference
from tools._attribute_protocol import validate_sources, SOURCE
from tools.attribute_rewriter_prompts import SYSTEM


def attribute_rewriter(description, references=None, *, kind, brief='', qwen=None,
                       output_dir=None, seed=None, **options):
    if kind not in ('image', 'video'):
        raise ValueError('Expected image or video target')
    refs = [reference(r) for r in references or []]
    if not isinstance(description, str):
        raise ValueError('Description must be text')
    if not description.strip():
        if refs:
            raise ValueError('Zero attributes must not attach references')
        # Vary the creative brief as well as stochastic decoding. An explicit
        # seed reproduces the chosen direction; None chooses a fresh direction.
        rng = random.Random(seed)
        identity = rng.choice(('person', 'garment', 'object', 'scene'))
        form = rng.choice(('flowing curves', 'crisp geometric planes', 'asymmetric balance',
                           'layered translucent surfaces', 'sculptural folds',
                           'delicate open structures', 'rounded volumes', 'precise pleats'))
        character = rng.choice(('warm and tactile', 'cool and luminous', 'quiet and monumental',
                                'airy and delicate', 'rich and restrained', 'playful and polished',
                                'minimal and poetic', 'organic and architectural'))
        task = (f'Target: {kind}.\nZERO sampled attributes; no source media.\n'
                f'Invent one new {identity} in one or two concise sentences.\n'
                f'Creative direction: {form}; {character}. Interpret appropriately for the identity.\n'
                f'Creative brief: {brief or "Open creative direction."}')
    elif not refs:
        description = validate_sources(description, [])
        task = (f'Target: {kind}.\nNEW SOURCE-FREE REQUIREMENT; no reference media.\n'
                'Treat this as a new visual requirement, not an observation of existing media.\n'
                'Preserve its stated content without inventing source labels or unrelated elements.\n'
                f'Creative brief: {brief or "None."}\nDescription:\n{description}')
    else:
        description = validate_sources(description, refs)
        used = set(SOURCE.findall(description))
        if not used:
            raise ValueError('Sampled attributes must identify their actual sources')
        if len(used) == 1:
            # Enforce exact preservation without spending a VLM call.
            if output_dir is not None:
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                (directory/'input.txt').write_text(description, encoding='utf-8')
                (directory/'response.txt').write_text(description, encoding='utf-8')
                (directory/'mode.txt').write_text('same-source passthrough; no VLM call\n')
            return description
        task = (f'Target: {kind}.\nMULTIPLE sources: preserve sampled attributes.\n'
                'Only repair strict cross-source conflicts, with the smallest necessary change.\n'
                f'Creative brief: {brief or "None."}\nSampled description:\n{description}')
    result = ask_text(SYSTEM, task, refs, qwen=qwen, output_dir=output_dir, seed=seed, **options)
    if result is None:
        return None
    try:
        return validate_sources(result, refs)
    except ValueError as error:
        print(f'attribute_rewriter: {error}', file=sys.stderr)
        return None
