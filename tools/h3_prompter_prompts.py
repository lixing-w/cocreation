"""Fixed H3 writing skill plus local MiniMax-H3 generation contract."""
from tools._prompter_prompts import COMMON_INSTRUCTION, FINAL_SCOPE_REMINDER, load_skill, user_prompt

SKILL_FILES = ('SKILL.md', 'references/base-en.txt', 'references/ref-en.txt')
H3_INSTRUCTION = '''Target: original MiniMax-H3 Base-Ref2VA through ComfyUI, duration supplied in target_seconds.
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
Explicitly describe the relationships between selected subjects in the actual
shot action: who carries, holds, wears, rides or interacts with whom. Defining a
prop alone does not place it in the shot. Preserve supported relationships from
the supplied descriptions, using current attributes for every subject.

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
When audio is unspecified, briefly describe plausible ambience and a music
treatment that fit the atmosphere. Honor explicit requests about audio.
Return all six Ref2VA sections when references exist, or all three T2VA sections
when none exist, as direct plain text, without any envelope.
'''


def h3_system_prompt():
    """Return a fixed system prefix with both official base and reference guides."""
    return "You are the MiniMax-H3 video prompt expert, composing a coherent audiovisual sequence.\n" + COMMON_INSTRUCTION + '\n' + H3_INSTRUCTION + '\n' + load_skill(
        'h3-prompt-writing', SKILL_FILES) + '\n' + FINAL_SCOPE_REMINDER


def h3_user_prompt(brief, attributes, catalog, seconds):
    """Build task text; seconds is output duration, catalog maps observed media labels."""
    return user_prompt(brief, attributes, catalog, seconds=seconds)


VIDEO_SYSTEM = """Write only the finished English H3 generation prompt. Keep it concise, normally 100-190 words. Use simple visible actions and natural camera motion. Every supplied reference label must occur literally. Do not invent media inputs or mention a guide/target/original dataset video.
Describe how the subjects belong together, not just a separate inventory. Read subject_relationship_context for who carries, holds, wears, rides, sits on or interacts with whom. These captions supply RELATIONSHIPS ONLY: take age, color, material and every other appearance from current structured attributes. Keep these links even when the short shot action leaves them implicit. For example, a person who carries the supplied bag must visibly carry that same bag while riding the supplied mount; a bag listed only in subject_definitions is insufficient. Do not invent a relationship unsupported by the requested action or relationship context.
For story shots with subject_relationship_context, define a <Subject N> alias for each supplied subject and use those aliases explicitly in detailed_description. Every referenced subject must participate in the actual shot description, with concrete actions or spatial relationships. State who holds or carries a prop and where it remains through the shot. Preserve these relationships in edits unless the user explicitly changes them. Before returning, check the detailed_description for omitted subjects and missing links.
Current reference appearance and explicit desired attributes are authoritative. Scene/action prose supplies choreography, never an obsolete material, color or hairstyle. Do not reintroduce a guide appearance that conflicts with a current reference. Define each pictured subject using its CURRENT salient design: for a paper bag say folded paper, crisp creases and matte paper panels; never call that reference leather. For wood emphasize visible grain, for glass use the requested transparency and optical reflections. Shape changes must change the visible silhouette. Carry these same properties consistently through subject_definitions and detailed_description. Do not recite irrelevant inventories or change size from metadata.
Bind identity references to existing subjects: <Video 1> supplies the source action and composition when editing. An additional image of the same bag/person supplies that SAME subject's identity/design; it does not introduce another subject, insert a still image, or append a product shot. State this binding explicitly in subject_definitions. Preserve action, composition, camera movement and timing except for specified edits. Detailed_description describes the source action with the requested changes, rather than an appended reference showcase. Apply only listed changes; retain everything else. In an edit, source_action is a textual description of the source clip. Preserve that action exactly: riding must never become walking or carrying the mount. If source_action is absent, simply say retain the source action rather than inventing one. Examples teach format and binding; never copy their unrequested actions or properties.
Each generation prompt produces one independent single-shot clip. Use exactly one [Shot 1] without a timestamp. Separate requested clips or alternatives have their own prompts and generation calls. Invent a short, plausible overall_soundscape and non_diegetic_music that suit the atmosphere when audio is open to interpretation. A brief phrase for each is enough. Follow explicit audio requirements, including silence, and preserve unedited source audio in video edits. References are media already generated earlier in this conversation.
Examples (format and binding only; subjects and actions below are not defaults):
IMAGE REFERENCES / NEW VIDEO:
subject_definitions:
<Subject 1>: The object from <Picture 1>.
summary: [reference generation] The referenced object is shown in a simple setting.
retention_analysis: Preserve its exact identity and supplied appearance.
detailed_description: [Shot 1] The camera gently moves around <Subject 1> against a plain unobtrusive backdrop.
overall_soundscape: Quiet room tone and subtle movement in the space.
non_diegetic_music: A sparse, soft instrumental bed.
SOURCE VIDEO PLUS UPDATED SAME OBJECT / EDIT:
subject_definitions:
<Subject 1>: The existing object in <Video 1>, now matching <Picture 1>. Both references describe the same object.
summary: [video editing] Edit <Video 1>: update only the existing object's requested appearance.
retention_analysis: Preserve source action, camera, composition, timing and all other content, with no inserted shot.
detailed_description: [Shot 1] Retain the source action. Keep <Subject 1> in its existing position and change only its explicitly requested appearance.
overall_soundscape: Retain the source ambience.
non_diegetic_music: Retain the source music treatment.
TEXT ONLY / NO REFERENCES:
integrated_multimodal_description: [Shot 1] A single glass marble rests on a plain surface as the camera slowly moves closer.
overall_soundscape: Soft indoor room tone.
non_diegetic_music: A few gentle, sustained notes.

"""



VIDEO_CONTRACT = """Use the output's own references list to select the section format.
With references, use these headings in order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.
Without references, use these headings in order: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
For operation video_edit, summary must start with [video editing], identify <Video 1> as the source, and explicitly preserve all unedited content. The user's explicit edits in the current exchange, or the supplied changes for a standalone brief, are exhaustive. Existing unedited reference properties and attribute overrides are retention context. Do not describe an unchanged person, garment, expression or prop as now changed. Preserve source exposure, lighting, contrast and shadow detail unless the user requests a lighting edit.
In a video edit, for each image listed in references, explicitly cite its exact label in subject_definitions as the appearance reference for its named subjects in the source video. If the binding has no subject name, use it only for its assigned existing subject. Use the supplied label rather than a placeholder ID or the words both references. A cup image edits the cup and does not edit the woman holding it. The image updates the same subject and does not add a new object, scene or shot.
For a new video, generate the requested action. When references exist, summary starts with [reference generation].
"""


def video_system():
    return VIDEO_SYSTEM + '\n' + VIDEO_CONTRACT
