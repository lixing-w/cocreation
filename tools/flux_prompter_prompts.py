"""Fixed FLUX.2-klein system prompt and per-request user prompt."""
from tools._prompter_prompts import COMMON_INSTRUCTION, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'rules/core-principles.md', 'rules/flux2-models.md',
               'rules/i2i-prompting.md', 'rules/multi-reference-editing.md')
FLUX_INSTRUCTION = '''Target: local FLUX.2-klein-4B image generation, not any video model or BFL API.
Write one compact natural-language image prompt, preferably 40-100 words, at most
200 words. Front-load selected subjects and clearly assign each reference's role.
Use positive descriptions and a coherent still-image composition. No negative
prompt field, section headings, markdown, or JSON envelopes.
Use the EXACT <Image N> generator_labels provided in the task map.
ALL supplied reference generator_labels must appear in your output.
Source-free sentences are new desired properties, with no source claim.
Each video source will later be replaced by ONE matching still image in
its same reference slot. Refer to that future image using generator_label, never
<Video N> or <Picture N>. Describe only a state that one frame can show. Do not
claim a frame has already been selected. If required video states are incompatible
within one frame, or temporal change itself is required, output only CANNOT_FULFILL.
For source-free new attributes, describe the desired property without claiming
reference support. Do not select a different model or apply the guides' API advice.

Example:
Input: The woman wears a red coat. (from <Image 1>)
Output: An elegant portrait preserving the woman's red coat like from <Image 1>, 
against a softly lit ivory background.
'''


def flux_system_prompt():
    """Return fixed local constraints plus official FLUX skill and relevant rules."""
    return "You are the FLUX.2-klein image prompt expert, composing a refined still image.\n" + COMMON_INSTRUCTION + '\n' + FLUX_INSTRUCTION + '\n' + load_skill(
        'flux-image-best-practices', SKILL_FILES)


def flux_user_prompt(brief, attributes, catalog):
    """Build task text from creative brief, attribute records and ordered source map."""
    return user_prompt(brief, attributes, catalog)
