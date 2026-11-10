"""Fixed H3 writing skill plus local VDN-H3 generation contract."""
from tools._prompter_prompts import COMMON_INSTRUCTION, FINAL_SCOPE_REMINDER, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'references/base-en.txt', 'references/ref-en.txt')
H3_INSTRUCTION = '''Target: local VDN-H3 through FreeVideo, duration supplied in target_seconds.
References present => Ref2VA, even for a single image: use these six plain-text
section headings in exactly this order, each followed by a colon:
subject_definitions, summary, retention_analysis, detailed_description,
overall_soundscape, non_diegetic_music.
No references => T2VA: use integrated_multimodal_description, overall_soundscape,
non_diegetic_music, in that order. Do not assume first/last-frame alignment.
Generator image labels are <Picture N>, video labels <Video N>, numbered per kind.
Qwen observation image labels are <Image N>; translate using the supplied map.
You may introduce <Subject N> content labels, but define each once on its own line
under subject_definitions before use. Each sourced definition MUST name its supplied
media label, not its artifact/node ID. Source-free subjects are new desired content.
After EVERY sourced property in every section, immediately append
"like in <Video N>" or "like in <Picture N>" using the mapped label. A <Subject N>
alias is not a substitute for this inline media reference. Do not fill sections
with unselected scenery, camera moves or actions; a static requested scene can
simply remain visible for the requested duration.
Complete reference example (five-second clip):
Input: Green foliage is visible on the right. (from <Video 1>)
Output:
subject_definitions:
<Subject 1> is the green foliage on the right, like in <Video 1>.
summary:
Green foliage is visible on the right, like in <Video 1>.
retention_analysis:
Retain only the green foliage on the right, like in <Video 1>.
detailed_description:
[Shot 1] Green foliage remains visible on the right for five seconds, like in <Video 1>.
overall_soundscape:
N/A
non_diegetic_music:
N/A
No bear, wall, rocks or camera motion is added from the reference.

Image-reference binding example:
Input: A silver gown has rounded folds. (from <Image 1>)
The hem curves upward. (from <Image 1>)
Map: <Image 1> -> <Picture 1>
Corresponding description lines:
A silver gown has rounded folds, like in <Picture 1>.
The hem curves upward, like in <Picture 1>.
Repeat the image label for each property; do not add fabric motion unless selected.

Complete source-free example (five-second clip):
Input: A pearl-colored sphere rests on pale sand.
Output:
integrated_multimodal_description:
[Shot 1] A pearl-colored sphere rests on pale sand for five seconds.
overall_soundscape:
N/A
non_diegetic_music:
N/A
Do not add warming light, breeze, camera moves or other unselected details.

Retain only selected source attributes; retention_analysis describes only those.
Within detailed_description or integrated_multimodal_description, start with
[Shot 1] without a timestamp. Subsequent cuts use [Shot N] At MM:SS.mmm,
with consecutive shot numbers and strictly increasing cut times below duration.
Plan a feasible visual timeline filling target_seconds. Use one shot unless more
are needed by the requested attributes/brief. Viewpoint objects are perceptually
meaningful framing stages, not necessarily editing shots. A continuous wide-to-close
viewpoint_change stays within one [Shot N]; add a new [Shot N] only for an actual
requested edit/shot transition. Preserve continuous versus cut transition types.
Use N/A for soundscape and music unless the supplied description explicitly
requests sound or music. Never infer sound from the reference or add ambience,
music or speech to fill the required sections. You cannot hear source audio.
Return all six Ref2VA sections when references exist, or all three T2VA sections
when none exist, as direct plain text, without any envelope.
'''


def h3_system_prompt():
    """Return a fixed system prefix with both official base and reference guides."""
    return "You are the VDN-H3 video prompt expert, composing a coherent audiovisual sequence.\n" + COMMON_INSTRUCTION + '\n' + H3_INSTRUCTION + '\n' + load_skill(
        'h3-prompt-writing', SKILL_FILES) + '\n' + FINAL_SCOPE_REMINDER


def h3_user_prompt(brief, attributes, catalog, seconds):
    """Build task text; seconds is output duration, catalog maps observed media labels."""
    return user_prompt(brief, attributes, catalog, seconds=seconds)
