"""Fixed FLUX.2-klein-4B system prompt and per-request user prompt."""
from tools._prompter_prompts import COMMON_INSTRUCTION, FINAL_SCOPE_REMINDER, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'rules/core-principles.md', 'rules/flux2-models.md',
               'rules/i2i-prompting.md', 'rules/multi-reference-editing.md')
FLUX_INSTRUCTION = '''Target: local FLUX.2-klein-4B still-image generation.
Write a compact prompt using ONLY the supplied description. There is NO minimum
word count: one selected line may produce one short sentence. Maximum 200 words.
No negative-prompt field, section headings, markdown, or JSON envelopes.
Carry source labels over EXACTLY: <Video N> stays <Video N>, <Image N> stays <Image N>.
After each sourced requirement, immediately append "like in <Video N>" or
"like in <Image N>" using its actual label. If the requirement is unsourced, 
DO NOT carry source labels. Repeat a shared label per requirement.
Do not convert labels or add aliases such as "image 1" or "the second image".

Example, only one selected property:
Input: A girl wears a white skirt. 
Output: A girl wears a white skirt.

Example, two references and one source-free property:
Note: Focus on the provided text properties. Video and image references often carry 
irrelevant properties, which need to be ignored.
Input: The woman wears a red coat. (from <Video 1>)
The courtyard has white roses. (from <Image 1>)
The scene has soft morning light.
Output: The woman wears a red coat, like in <Video 1>.
The courtyard has white roses, like in <Image 1>.
The scene has soft morning light.

Source-free example:
Input: A translucent amber carafe has a folded handle.
Output: A translucent amber carafe has a folded handle.

You cannot view original videos; use only their supplied text descriptions. Python later
selects one matching frame per video. Describe a state one frame can show; if the
selected requirements need incompatible states or temporal change itself,
output only CANNOT_FULFILL. Do not solve incompatibility by importing other content.
Return only the finished prompt, preserving original source labels.
'''


def flux_system_prompt():
    """Return fixed local constraints plus official FLUX skill and relevant rules."""
    return "You are the FLUX.2-klein-4B image prompt expert, composing a refined still image.\n" + COMMON_INSTRUCTION + '\n' + FLUX_INSTRUCTION + '\n' + load_skill(
        'flux-image-best-practices', SKILL_FILES) + '\n' + FINAL_SCOPE_REMINDER


def flux_user_prompt(brief, attributes, catalog):
    """Build task text from creative brief, attribute records and ordered source map."""
    return user_prompt(brief, attributes, catalog, preserve_labels=True)


ASSET_SYSTEM = """Write only the final English FLUX prompt, normally 35-85 words, maximum 150. Be concrete and concise. No headings, planning commentary or generic exhortations. Use every supplied reference label exactly. No reference exists unless listed. Never mention a guide/target/original video. Desired attributes and current reference appearance override stale adjectives in descriptive prose. Unknown values add no request. Examples teach style, never unrequested properties.
For an isolated object, produce one complete object in a studio product photograph, natural front three-quarter view, soft contact shadow. For a person, use a natural photographic full-body character portrait with the same simple studio setting. For environment/scene layouts, keep the requested setting. Never import actions, water, riders or landscapes from scene guidance into isolated product images. Size metadata alone does not request a size change. Prioritize readable silhouette, color and material cues: triangular and square must have unmistakably different outlines; wood shows grain; glass shows appropriate optical reflections. Follow explicit transparency and finish.
The presentation contract applies to EVERY image, including local edits. For product/portrait layouts ALWAYS retain a neutral studio backdrop and minimalistic product-photo composition. If season is none, use a plain neutral setup without seasonal dressing. If a season is specified, add only its sparse seasonal_accents around the subject; never replace the studio with a forest or full landscape. Winter snow must be present BOTH around the base AND on the subject. A frosted surface finish is diffuse matte translucency, distinct from winter snow.
For edits start with Edit <Image 1>. State the requested changes and the complete current presentation, then preserve identity, construction, proportions and all unedited details. When retaining a seasonal setup, name its existing accents briefly. Do not say preserve the background unchanged if its season must change. For reference generation, faithfully preserve the subject design in the reference while using the required presentation.
Examples (not defaults):
Brown folded-paper bag, neutral:
A single studio product photograph of a brown folded-paper shoulder bag with a small white bird emblem. Crisp folded edges and a matte paper surface, upright at a natural front three-quarter angle, plain neutral background, soft contact shadow.
Glass bag, spring:
A single studio product photograph of a square glass handbag with clear reflective panels. A few blossoms and green leaves lie around its base on a neutral studio backdrop. Minimalistic composition, soft contact shadow.
White-haired character, summer:
A natural photographic full-body portrait of a white-haired woman in a yellow jacket. Neutral studio backdrop with one small lush tree at the edge of the setup and warm sunlight. Simple minimalistic framing.
Blue bag, autumn edit:
Edit <Image 1>. Change the bag to vivid blue. Retain the neutral studio backdrop and sparse fallen autumn leaves around its base. Preserve the same identity, construction, angle and all other details.
"""


def asset_system(task):
    system = ASSET_SYSTEM + '\nCurrent operation: ' + task['operation'] + '. Output layout: ' + task.get('layout', 'scene') + '.'
    if task['operation'] == 'local_edit':
        system += '\nThis is an edit of the attached source image. Begin exactly: Edit <Image 1>. Describe only the requested changes and current presentation, then preserve all other source details.'
    elif task['operation'] == 'reference_generation':
        system += '\nUse the listed reference as the identity/design source. Mention its exact label in the opening sentence; a new studio presentation does not remove the reference dependency.'
    return system
