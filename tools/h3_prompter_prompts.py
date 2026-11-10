"""Fixed H3 writing skill plus local VDN-H3 generation contract."""
from tools._prompter_prompts import COMMON_INSTRUCTION, load_skill, user_prompt

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
Binding example (snippet only):
Input {"text":"Large brown bear.","artifact_ids":["<Video 1>"]}
=> <Subject 1> is the large brown bear from <Video 1>.
Input {"text":"Rocky enclosure.","artifact_ids":["<Video 1>"]}
=> <Subject 2> is the rocky enclosure from <Video 1>.
Both subjects share one source label. Empty artifact_ids means no source claim.
Retain only selected source attributes; retention_analysis describes intended
inheritance.
Within detailed_description or integrated_multimodal_description, start with
[Shot 1] without a timestamp. Subsequent cuts use [Shot N] At MM:SS.mmm,
with consecutive shot numbers and strictly increasing cut times below duration.
Plan a feasible visual timeline filling target_seconds. Use one shot unless more
are needed by the requested attributes/brief. Viewpoint objects are perceptually
meaningful framing stages, not necessarily editing shots. A continuous wide-to-close
viewpoint_change stays within one [Shot N]; add a new [Shot N] only for an actual
requested edit/shot transition. Preserve continuous versus cut transition types.
Design plausible NEW sound for the target scene: restrained ambience and physical
sounds matching visible materials, actions and timing. Add subtle background music
only when it suits the scene and brief; otherwise use N/A for music. Respect explicit
silence requests. Avoid implausible, abrupt, exaggerated or unrelated sound events.
You cannot hear the references: do not claim observed audio, source voice inheritance,
or introduce <Audio N> references. Do not infer dialogue from lip movements or add
unrequested speech. Sound descriptions are generation choices, not observations.
Return all six Ref2VA sections when references exist, or all three T2VA sections
when none exist, as ONE text string in the JSON object's "prompt" field.
'''


def h3_system_prompt():
    """Return a fixed system prefix with both official base and reference guides."""
    return COMMON_INSTRUCTION + '\n' + H3_INSTRUCTION + '\n' + load_skill(
        'h3-prompt-writing', SKILL_FILES)


def h3_user_prompt(brief, attributes, catalog, seconds):
    """Build task text; seconds is output duration, catalog maps observed media labels."""
    return user_prompt(brief, attributes, catalog, seconds=seconds)
