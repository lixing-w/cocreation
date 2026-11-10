"""Deterministic fictional prose walkthrough; no model calls or media generation."""
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation import Reference
from tools._attribute_protocol import sample_selection


def walkthrough():
    pool = [
        ('A ceramic cup has a red glaze.\nThe cup rests on a wooden table.',
         Reference('artifact_0', '/fictional/cup.png', 'image')),
        ('White flowers fill a walled courtyard.\nSoft morning light fills the courtyard.',
         Reference('artifact_1', '/fictional/garden.png', 'image'))]
    sampled, refs = sample_selection(pool[:1], 2, random.Random(0), max_parents=2)
    # Same-source mode returns sampled text exactly; no model call is necessary.
    preserved = sampled
    empty, no_refs = sample_selection(pool, 0, random.Random(0))
    assert empty == '' and no_refs == []
    # Fictional source-free model response; no old media or captions are attached.
    invented = 'An ivory wool cape has a sculptural crescent collar.'
    observed = 'An ivory wool cape has a curved collar.\nThe cape has a charcoal lining.'
    generated = Reference('artifact_2', '/fictional/new_cape.png', 'image')
    pool.append((observed, generated))
    next_sample, _ = sample_selection([pool[-1]], 1, random.Random(0))
    return ('Fictional walkthrough: descriptions below are demonstrations, not model observations.\n\n'
            + 'Same-source sample, passed through unchanged:\n' + preserved
            + '\n\nZero-attribute exploration; no references attached:\n' + invented
            + '\n\nVerifier description for the new image artifact_2:\n' + observed
            + '\n\nLater sample, now referencing artifact_2 itself:\n' + next_sample)


if __name__ == '__main__':
    print(walkthrough())
