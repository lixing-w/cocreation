"""Deterministic fictional prose walkthrough; no model calls or media generation."""
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation import Reference
from tools._attribute_protocol import sample_selection, prune_references


def walkthrough():
    pool = [
        ('A ceramic cup has a red glaze.\nThe cup rests on a wooden table.',
         Reference('artifact_0', '/fictional/cup.png', 'image')),
        ('White flowers fill a walled courtyard.\nSoft morning light fills the courtyard.',
         Reference('artifact_1', '/fictional/garden.png', 'image'))]
    sampled, refs = sample_selection(pool, 4, random.Random(0), max_parents=2)
    # Fictional model responses, demonstrating reference retention and new content.
    labels = {ref.artifact_id: f'<Image {i}>' for i, ref in enumerate(refs, 1)}
    rewritten = (
        f'A ceramic cup has a red glaze. (from {labels["artifact_0"]})\n'
        'A single white flower rises from the cup like a delicate sail.\n'
        'The cup stands on pale stone in soft morning light.')
    planned, retained = prune_references(rewritten, refs)
    observed = ('A ceramic cup has a dark red glaze.\n'
                'A white flower rises from the cup.\n'
                'The cup stands on pale stone in soft morning light.')
    generated = Reference('artifact_2', '/fictional/generated.png', 'image')
    pool.append((observed, generated))
    next_sample, _ = sample_selection([pool[-1]], 1, random.Random(0))
    return ('Fictional walkthrough: descriptions below are demonstrations, not model observations.\n\n'
            + 'Sampled description:\n' + sampled
            + '\n\nRewritten description sent to the prompter:\n' + planned
            + '\n\nAttached sources: ' + ', '.join(r.artifact_id for r in retained)
            + '\n\nVerifier description for artifact_2:\n' + observed
            + '\n\nLater sample, now referencing artifact_2 itself:\n' + next_sample)


if __name__ == '__main__':
    print(walkthrough())
