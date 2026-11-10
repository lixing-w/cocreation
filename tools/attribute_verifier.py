"""Revise intended prose against the generated artifact, with no inheritance pass."""
from tools._attribute_vlm import ask_text, reference, observation_result
from tools._attribute_protocol import description_text
from tools.attribute_verifier_prompts import SYSTEM


def attribute_verifier(*, generated_media, description, qwen=None, output_dir=None, **options):
    ref = reference(generated_media)
    description = description_text(description, sourced=True)
    task = ('Inspect only the newly generated ' + ref.kind
            + '. Old-source markers below are historical plan text, not attached media.\n'
            + 'Intended description supplied to the generation prompter:\n' + description)
    return observation_result(ask_text(SYSTEM, task, [ref], qwen=qwen,
                                      output_dir=output_dir, **options))
