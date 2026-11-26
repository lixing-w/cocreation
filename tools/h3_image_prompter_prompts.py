"""Task-specific Qwen instructions derived from the pinned official H3 Image skill."""
from tools._prompter_prompts import SKILLS_ROOT

BASE = '''Write the final English MiniMax-H3-Image prompt as direct plain text.
Use the supplied current requirements as the complete content specification.
Keep each selected property bound to its named subject. Current structured
attributes determine that subject's current appearance. Quote requested visible
lettering exactly in its original language. Describe the finished image in
composition order with concrete visible details. A sparse request needs a short
prompt. Return only the finished prompt, or CANNOT_FULFILL if incoherent.
Canvas orientation is supplied separately for composition. The executor sets
resolution and sampling. Output only visual prose, with at most 1200 words.
'''


def h3_image_system_prompt(editing=False):
    # Keep upstream files intact on disk. Send the matching rules only: worked
    # examples introduce unrelated subjects, and the other mode adds no context.
    name = 'edit-en.txt' if editing else 't2i-en.txt'
    guide = (SKILLS_ROOT/'h3-image-prompt-writing/references'/name).read_text()
    guide = guide.split('## Complete examples', 1)[0].split('## Examples', 1)[0]
    if editing:
        # The guide's inline example has its own subject and color. Keep that
        # material in the vendored skill, outside the Qwen request.
        guide = guide.replace('Begin with concise, source-grounded definitions such as `The red cup at the left of <Picture 1> is <Subject 1>.` ',
                              'Begin with concise definitions of the selected source subjects. ')
    guide = guide.replace('Ask only when', 'Resolve only essential missing dependencies; ask only when')
    return guide + '\nLocal execution contract (takes priority):\n' + BASE


ASSET_RULES = """Apply these image rules to each output's own operation and references.
For layout product, present one complete object in a minimal studio product photograph, natural front three-quarter view, neutral backdrop and soft contact shadow.
For layout portrait, present the person in a natural photographic full-body portrait, neutral studio backdrop and simple framing.
When presentation specifies seasonal_accents, incorporate those exact accents.
When presentation specifies time_of_day, use compatible lighting on the requested
scene or subject while retaining the supplied layout and background.
For local_edit, begin by naming the edited subject and its exact supplied image label, such as <Picture 1>. State the target result and the exhaustive requested changes, then preserve every other source detail. The subject is partially preserved when one of its properties changes. Do not claim full preservation of an edited subject or its whole image. Earlier conversation scenes explain later use, not extra content to insert into this asset image.
For a generation with references, define the selected identity or property from each listed <Picture N>, then describe its role in the target image and the current presentation.
"""


def asset_system(reference_modes):
    if len(reference_modes) == 1:
        guide = h3_image_system_prompt(reference_modes[0])
    else:
        guide = BASE + (
            '\nFor each output in a mixed delivery, use only its own reference list. When it is empty, '
            'write a complete standalone image description. When references exist, use only the '
            'selected properties of each source, using exact labels to bind them to their subjects. '
            'For a local edit, describe the change and retain all compatible source content.'
        )
    return guide + '\n' + ASSET_RULES


def h3_image_user_prompt(brief, attributes, catalog):
    import re
    mapping = {r['observation_label']: r['generator_label'] for r in catalog}
    text = re.sub(r'<(?:Image|Video) [1-9]\d*>', lambda m: mapping.get(m.group(), m.group()), attributes)
    return ('Current visual requirements:\n' + text
            + ('\nCreative brief:\n'+brief if brief else '')
            + '\nReturn only the finished image prompt using the supplied source labels.')
