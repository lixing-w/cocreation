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
After every sourced property in every section, immediately append
"like in <Video N>" or "like in <Picture N>" using the mapped label. If the property 
is unsourced, DO NOT add source labels. A <Subject N>
alias is not a substitute for this inline media reference. Do not fill sections
with unselected scenery, camera moves or actions; a static requested scene can
simply remain visible for the requested duration.

Complete reference example:
Note: DO NOT add properties not listed in the input text, visual reference images 
and videos often carry irrelevant properties.
Input: A girl wears a white skirt. (from <Video 1>)
A photo studio with cinematic lighting. (from <Video 2>)
Output:
subject_definitions:
<Subject 1> is the girl wearing a white skirt, like in <Video 1>.
summary:
A girl wears a white skirt, like in <Video 1>, stands 
in a photo studio with cinematic lighting, like in <Video 2>.
retention_analysis:
Retain the girl with white skirt, like in <Video 1>, and 
only the studio setting with cinematic lighting, like in <Video 2>
detailed_description:
[Shot 1] The girl wears a white skirt stands in a photo studio with cinematic lighting.
overall_soundscape:
Soft studio ambient humming, gentle click of camera shutters, and subtle rustle of the skirt fabric.
non_diegetic_music:
N/A


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
Gentle ocean waves rolling ashore, soft wind whispering across the sand, 
and subtle crunch of tiny sand grains settling around the sphere.
non_diegetic_music:
N/A

Retain only selected source attributes; retention_analysis describes only those.
Within detailed_description or integrated_multimodal_description, start with
[Shot 1] without a timestamp. Subsequent cuts use [Shot N] At MM:SS.mmm,
with consecutive shot numbers and strictly increasing cut times below duration.
Plan a feasible visual timeline filling target_seconds. Use one shot unless more
are needed by the requested attributes/brief. Viewpoint objects are perceptually
meaningful framing stages, not necessarily editing shots. A continuous wide-to-close
viewpoint_change stays within one [Shot N]; add a new [Shot N] only for an actual
requested edit/shot transition. Preserve continuous versus cut transition types.
You may infer sound from the reference or add ambience,
music or speech to fill the required sections. 
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


VIDEO_SYSTEM = """Write only the finished English H3 generation prompt. Keep it concise, normally 100-190 words. Use simple visible actions and natural camera motion. Every supplied reference label must occur literally. Do not invent media inputs or mention a guide/target/original dataset video.
Current reference appearance and explicit desired attributes are authoritative. Scene/action prose supplies choreography, never an obsolete material, color or hairstyle. Do not reintroduce a guide appearance that conflicts with a current reference. Define each pictured subject using its CURRENT salient design: for a paper bag say folded paper, crisp creases and matte paper panels; never call that reference leather. For wood emphasize visible grain, for glass use the requested transparency and optical reflections. Shape changes must change the visible silhouette. Carry these same properties consistently through subject_definitions and detailed_description. Do not recite irrelevant inventories or change size from metadata.
Bind identity references to existing subjects: <Video 1> supplies the source action and composition when editing. An additional image of the same bag/person supplies that SAME subject's identity/design; it does not introduce another subject, insert a still image, or append a product shot. State this binding explicitly in subject_definitions. Preserve action, composition, camera movement and timing except for specified edits. Detailed_description describes the source action with the requested changes, rather than an appended reference showcase. Apply only listed changes; retain everything else. In an edit, source_action is a textual description of the source clip. Preserve that action exactly: riding must never become walking or carrying the mount. If source_action is absent, simply say retain the source action rather than inventing one. Examples teach format and binding; never copy their unrequested actions or properties.
Use one continuous shot unless cuts are requested. Start with [Shot 1] without a timestamp; further shots use [Shot 2] At 00:02.500, etc. Use N/A for soundscape and music. References are media already generated earlier in this conversation.
Examples (example identities, actions and styles are not defaults):
IMAGE REFERENCES / NEW VIDEO:
subject_definitions:
<Subject 1>: The traveler from <Picture 1>.
<Subject 2>: The shoulder bag from <Picture 2>.
summary: [reference generation] A traveler walks beside a canal carrying the bag.
retention_analysis: Preserve each subject's identity and the bag's construction.
detailed_description: [Shot 1] Side tracking view of <Subject 1> walking along the canal with <Subject 2> hanging naturally at their hip. Their steps gently swing the bag while the camera follows at walking speed.
overall_soundscape: N/A
non_diegetic_music: N/A
CURRENT PAPER BAG / NEW VIDEO:
subject_definitions:
<Subject 1>: The folded-paper bag in <Picture 1>, with crisp creases, matte paper panels and its bird emblem.
summary: [reference generation] A traveler carries the paper bag while riding through the canal.
retention_analysis: Keep the exact bag design and folded-paper material of <Picture 1> throughout.
detailed_description: [Shot 1] Side tracking view of a traveler riding a woolly sheep along the canal. <Subject 1> hangs at the traveler's hip; its rigid folded edges and paper panels remain clearly visible as it gently sways.
overall_soundscape: N/A
non_diegetic_music: N/A
SOURCE VIDEO PLUS UPDATED SAME BAG / EDIT:
subject_definitions:
<Subject 1>: The existing carried bag in <Video 1>, now matching the exact bag design in <Picture 1>. Both references describe the same bag.
summary: [video editing] Edit <Video 1>: make the scene winter and black and white; update its existing bag to <Picture 1>.
retention_analysis: Preserve the source people, motion, camera, shot structure and timing; replace only the existing bag's appearance, with no inserted shot.
detailed_description: [Shot 1] Retain the source action and framing. Apply winter snow and black-and-white rendering throughout, with <Subject 1> following the same motion and remaining attached in its original position.
overall_soundscape: N/A
non_diegetic_music: N/A
TEXT ONLY / NO REFERENCES:
integrated_multimodal_description: [Shot 1] A pink woolly sheep walks down a shallow flooded street at sunset, carrying a traveler in a yellow jacket with a shoulder bag. A wide rear view gently tracks them towards the distant buildings.
overall_soundscape: N/A
non_diegetic_music: N/A
"""

VIDEO_SYSTEM += '\nResolve conflicting instructions before writing. Priority: current output rendering rules > compatible explicit edit > current design attributes > reference preservation > guide prose. A black-and-white output rule covers EVERY subject and background. In that case do not mention pink wool, brown bags, multicolored clothing, colored lights, hues or saturation anywhere, even in subject names. Keep texture, silhouette and achromatic tonal details. Never invent selective color or an exception. For a local edit, a preserve-everything sentence must explicitly allow the higher-priority output treatment. A season changes studio accents while preserving the neutral product composition.\nConflict examples (style only):\nInput: pink sheep, medium saturation; output black and white.\nOutput wording: a woolly sheep, photographed entirely in black and white on a neutral studio background.\nInput: source video and blue bag reference; output black and white.\nOutput wording: replace the existing bag with the exact design in <Picture 1>; render every subject and the entire scene in grayscale. Preserve the source action and camera.\nInput: reference has autumn leaves; current season winter.\nOutput wording: keep the neutral studio setup, with snow around the base and on the product surfaces.\n'


def video_system(task):
    sections = ('subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music'
        if task.get('references') else 'integrated_multimodal_description, overall_soundscape, non_diegetic_music')
    operation = ('For this video edit, summary must start with [video editing], identify <Video 1> as the source being edited, and explicitly preserve all unedited content.'
        if task['operation'] == 'video_edit' else 'Generate the requested action. When references exist, summary starts with [reference generation].')
    return VIDEO_SYSTEM + '\nUse exactly these headings in order: ' + sections + '.\n' + operation
