"""Fixed FLUX.2-klein system prompt and per-request user prompt."""
from tools._prompter_prompts import COMMON_INSTRUCTION, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'rules/core-principles.md', 'rules/flux2-models.md',
               'rules/i2i-prompting.md', 'rules/multi-reference-editing.md')
FLUX_INSTRUCTION = '''Target: local FLUX.2-klein-4B image generation, not any video model or BFL API.
Write one compact natural-language image prompt, preferably 40-100 words, at most
200 words. Front-load selected subjects and clearly assign each reference's role.
Use positive descriptions and a coherent still-image composition. No negative
prompt field, section headings, markdown, or JSON inside the prompt string.
Use the EXACT <Image N> generator_labels provided in the task map, preferably
phrased as "the coat from <Image 1> (image 1)" to make image ordering explicit.
Binding example: {"text":"Red coat.","artifact_ids":["<Image 1>"]}
=> "Preserve the red coat from <Image 1>." Never substitute an artifact/node ID.
Empty artifact_ids means a new desired property, with no source claim.
Observation labels may differ: Qwen can see videos, but the generation expert
cannot. Each video source will later be replaced by ONE matching still image in
its same reference slot. Refer to that future image using generator_label, never
<Video N> or <Picture N>. Describe only a state that one frame can show. Do not
claim a frame has already been selected. If required video states are incompatible
within one frame, or temporal change itself is required, return {"prompt":null,"reason":"brief specific reason"}.
For source-free new attributes, describe the desired property without claiming
reference support. Do not select a different model or apply the guides' API advice.
'''


def flux_system_prompt():
    """Return fixed local constraints plus official FLUX skill and relevant rules."""
    return COMMON_INSTRUCTION + '\n' + FLUX_INSTRUCTION + '\n' + load_skill(
        'flux-image-best-practices', SKILL_FILES)


def flux_user_prompt(brief, attributes, catalog):
    """Build task text from creative brief, attribute records and ordered source map."""
    return user_prompt(brief, attributes, catalog)
