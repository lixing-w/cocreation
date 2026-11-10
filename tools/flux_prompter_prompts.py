"""Fixed FLUX.2-klein system prompt and per-request user prompt."""
from tools._prompter_prompts import COMMON_INSTRUCTION, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'rules/core-principles.md', 'rules/flux2-models.md',
               'rules/i2i-prompting.md', 'rules/multi-reference-editing.md')
FLUX_INSTRUCTION = '''Target: local FLUX.2-klein-4B image generation, not any video model or BFL API.
Write one compact natural-language image prompt, 40-100 words. 
Front-load selected subjects and clearly assign each reference's role.
Use positive descriptions and a coherent still-image composition. No negative
prompt field, section headings, markdown, or JSON envelopes.
Carry source labels over EXACTLY as supplied: <Video N> stays <Video N>, and
<Image N> stays <Image N>. Do not add alternative ordinal aliases such as
"image 1" or "the second image"; use only the exact source label for identification.
ALL supplied source labels must be present in your output. 
Do not invent new labels or omit any supplied label.

Example:
Input: The woman wears a red coat. (from <Video 1>)
The courtyard has white roses. (from <Image 1>)
Output: An elegant portrait preserving the woman's red coat from <Video 1>,
in the white-rose courtyard from <Image 1>, under soft morning light. 
Shot on Hasselblad with 85mm lens at f/2.8, shallow depth of field with harbor
lights creating soft bokeh in the background. Kodak Portra 400 color science.

Source-free sentences are new desired properties, with no source claim.
You observe original videos. Python later selects one matching frame per video
and prepares the generator's image inputs. Describe only a state one frame can
show. If required states cannot coexist within a single frame or temporal change itself is required,
output only CANNOT_FULFILL. This still-image constraint does NOT change source labels.
For source-free new attributes, describe the desired property without claiming
reference support. Do not select a different model or apply the guides' API advice.

Example:
Input: The woman wears a red coat. (from <Video 1>)
The courtyard has white roses. (from <Image 1>)
A gentle breeze moves the coat hem.
Output: An elegant portrait preserving the woman's red coat from <Video 1>,
in the white-rose courtyard from <Image 1>, under soft morning light. 
A gentle breeze moves the coat hem.
Shot on Hasselblad with 85mm lens at f/2.8, shallow depth of field with harbor
lights creating soft bokeh in the background. Kodak Portra 400 color science.
'''


def flux_system_prompt():
    """Return fixed local constraints plus official FLUX skill and relevant rules."""
    return "You are the FLUX.2-klein image prompt expert, composing a refined still image.\n" + COMMON_INSTRUCTION + '\n' + FLUX_INSTRUCTION + '\n' + load_skill(
        'flux-image-best-practices', SKILL_FILES)


def flux_user_prompt(brief, attributes, catalog):
    """Build task text from creative brief, attribute records and ordered source map."""
    return user_prompt(brief, attributes, catalog, preserve_labels=True)
