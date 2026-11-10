"""Fixed FLUX.2-klein system prompt and per-request user prompt."""
from tools._prompter_prompts import COMMON_INSTRUCTION, FINAL_SCOPE_REMINDER, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'rules/core-principles.md', 'rules/flux2-models.md',
               'rules/i2i-prompting.md', 'rules/multi-reference-editing.md')
FLUX_INSTRUCTION = '''Target: local FLUX.2-klein-4B still-image generation.
Write a compact prompt using ONLY the supplied description. There is NO minimum
word count: one selected line may produce one short sentence. Maximum 200 words.
No negative-prompt field, section headings, markdown, or JSON envelopes.
Carry source labels over EXACTLY: <Video N> stays <Video N>, <Image N> stays <Image N>.
After EACH sourced requirement, immediately append "like in <Video N>" or
"like in <Image N>" using its actual label. Repeat a shared label per requirement.
Do not convert labels or add aliases such as "image 1" or "the second image".
Python converts labels to generator slots after you return.

Example, only one selected property despite a complex reference video:
Input: Green foliage is visible on the right. (from <Video 1>)
Output: Green foliage is visible on the right, like in <Video 1>.
Do not include the bear, rocks, wall or sunlight seen elsewhere in that reference.

Example, two references and one source-free property:
Input: The woman wears a red coat. (from <Video 1>)
The courtyard has white roses. (from <Image 1>)
The scene has soft morning light.
Output: The woman wears a red coat, like in <Video 1>.
The courtyard has white roses, like in <Image 1>.
The scene has soft morning light.
Do not add a camera brand, lens, film stock, harbor lights or any other detail.

Source-free example:
Input: A translucent amber carafe has a folded handle.
Output: A translucent amber carafe has a folded handle.

You observe original videos only to clarify selected properties. Python later
selects one matching frame per video. Describe a state one frame can show; if the
selected requirements need incompatible states or temporal change itself,
output only CANNOT_FULFILL. Do not solve incompatibility by importing other content.
Return only the finished prompt, preserving original source labels.
'''


def flux_system_prompt():
    """Return fixed local constraints plus official FLUX skill and relevant rules."""
    return "You are the FLUX.2-klein image prompt expert, composing a refined still image.\n" + COMMON_INSTRUCTION + '\n' + FLUX_INSTRUCTION + '\n' + load_skill(
        'flux-image-best-practices', SKILL_FILES) + '\n' + FINAL_SCOPE_REMINDER


def flux_user_prompt(brief, attributes, catalog):
    """Build task text from creative brief, attribute records and ordered source map."""
    return user_prompt(brief, attributes, catalog, preserve_labels=True)
