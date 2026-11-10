# CPG visual attributes and sampling (discussion draft)

This file proposes the next CPG layer; it is not an implemented graph builder.
Audio is outside scope because the observing Qwen3-VL interface cannot hear it.

An attribute denotes a property grounded in a particular artifact/version. Its
language description is an approximation, not a replacement for the source media.
An observed association and a requested inheritance are distinct. Only comparison
of the generated artifact against its source can support a shared-attribute edge.

## Definitions

**Need flow in time?** means temporal change is necessary to express the attribute.
`Optional` means an entry may describe either a static relation or its evolution.
**Allow multiple?** is scoped to the parent: dotted entries are counted per identity,
environment or viewpoint; undotted entries are counted per artifact. `No` means at
most one, not that a value must always be present. Unknown/inapplicable values are
omitted rather than invented.

**short name** is unique across all categories and is the literal category suffix
in an entry ID; dots are retained. A child ID is
`<full parent_id>_<self>_<short name>`; a parentless ID is `<self>_<short name>`.
`self` is a meaningful name or a sibling-distinguishing label, using letters,
digits or hyphens. For example: `woman0_id`, `woman0_id_d1_id.desc`,
`canal0_env`, `canal0_env_s0_env.space`, `canal0_env_s0_env.season`.
Every ID must remain unique across the entire artifact. Use distinct self labels
for multiple entries of the same category under one parent. Reuse the full IDs in
parent_id, links and textual references; artifact_id is prefixed only downstream.
Season means a visibly supported season, not time of day; late afternoon belongs
to environment.time_of_day. Omit uncertain seasons.

### Identities

| Category | short name | Definition and scope | Need flow in time? | Allow multiple? |
|---|---|---|---|---|
| identity | id | A perceptually coherent subject or meaningful object group | No | Yes |
| &nbsp;&nbsp;identity.description | id.desc | Overall identity/design description | No | No |
| &nbsp;&nbsp;identity.appearance | id.app | Local appearance: clothing, color, material, shape, texture | No | Yes |
| &nbsp;&nbsp;identity.body_pose | id.pose | A body's configuration and orientation at a given state | No | Yes |
| &nbsp;&nbsp;identity.expression | id.expr | A visible facial/expressive state | No | Yes |
| &nbsp;&nbsp;identity.appearance_change | id.appc | Timeline connecting this identity's appearance states | Yes | No |
| &nbsp;&nbsp;identity.expression_change | id.exprc | Timeline connecting this identity's expression states | Yes | No |
| &nbsp;&nbsp;identity.pose_change | id.posec | Timeline connecting this identity's body-pose states | Yes | No |
| &nbsp;&nbsp;identity.motion | id.mot | A subject's activity and phases, including trajectory, speed, rhythm or deformation | Yes | Yes |

**Fictional video example (6 seconds).** A woman in an open red coat stands facing
left, buttons her coat, turns toward the camera, smiles, and sits on a stool.

- `identity`: `woman0_id` (woman), `stool0_id` (stool). The following children belong
  to `woman0_id`; the stool can have its own description and appearance.
- `identity.description`: an adult woman with short curly hair.
- `identity.appearance`: `woman0_id_a1_id.app` open red wool coat (0–2 s), `woman0_id_a2_id.app` buttoned red wool
  coat (2–6 s), `woman0_id_a3_id.app` short curly black hair (0–6 s).
- `identity.body_pose`: `woman0_id_p1_id.pose` upright, facing left (0–2 s); `woman0_id_p2_id.pose` upright, facing
  camera (3–4 s); `woman0_id_p3_id.pose` seated, knees bent (5–6 s).
- `identity.expression`: `woman0_id_e1_id.expr` neutral mouth (0–3 s); `woman0_id_e2_id.expr` visible smile (4–6 s).
- `identity.appearance_change`: "As the coat closes, woman0_id_a1_id.app changes to woman0_id_a2_id.app by 2 s.
  woman0_id_a3_id.app is present throughout and unchanged."
- `identity.expression_change`: "The person starts with the neutral expression
  woman0_id_e1_id.expr, then changes to the smile woman0_id_e2_id.expr during 3–4 s and keeps smiling through 6 s."
- `identity.pose_change`: "The person starts with woman0_id_p1_id.pose, then changes to woman0_id_p2_id.pose as she
  turns toward the camera during 2–3 s, then changes to woman0_id_p3_id.pose as she lowers herself
  onto the stool during 4–5 s. She remains seated through the end."
- `identity.motion`: three entries: "The person buttons her coat during 0–2 s,
  bringing its front panels together"; "She turns toward the camera during 2–3 s,
  rotating her torso from a left-facing to a front-facing orientation"; "She sits
  down during 4–5 s, bending her knees and smoothly lowering her hips onto the
  stool." Each combines the meaningful activity with its visible movement.

**Fictional image example.** A woman with short curly hair sits on a wooden stool,
wearing a buttoned red wool coat and smiling at the camera.

- `identity`: `woman1_id` and `stool1_id`, two coherent subjects.
- `identity.description`: for `woman1_id`, an adult woman with short curly hair;
  for `stool1_id`, a four-legged wooden stool.
- `identity.appearance`: for `woman1_id`, red wool coat and curly black hair as
  separate entries; for `stool1_id`, pale wood grain.
- `identity.body_pose`: for `woman1_id`, seated upright with bent knees.
- `identity.expression`: for `woman1_id`, a visible smile.
- `identity.appearance_change`, `identity.expression_change`, `identity.pose_change`,
  `identity.motion`: absent. The still supports a seated state,
  not a claim that she sat down, turned, or changed expression.


Appearance, body pose and expression are concepts under an identity, not independent
unbound subjects. Every entry binds exactly one `identity_id`. Multiple body-pose or
expression entries can describe the same identity at different times; each has its
own state ID and temporal scope. Each change entry references the corresponding
state IDs in a natural-language account of their order and transitions, including
unchanged states when relevant. Examples above are actual attribute contents, not
just instructions to connect IDs. Meaningful activities and their phases belong to
identity.motion together with their visible kinematics.
For example, one identity.pose_change can connect standing, turning and sitting.
There may be many pose_change entries across a video, but at most one per identity.
The former pose_expression category is split into body_pose and expression.

Identity granularity should follow human perception and the task: a lawn can be one
coherent unit, not millions of grass identities. Do not split every texture or part
into a subject; split only independently meaningful entities. Exact granularity is
still an open design choice.

### Environments / backgrounds

`background` in relation endpoints refers to an `environment` ID; it is not a second
unconnected object type. An environment is a coherent setting, physical or imagined.

| Category | short name | Definition and scope | Need flow in time? | Allow multiple? |
|---|---|---|---|---|
| environment | env | A distinct setting/background, independent of shot boundaries | No | Yes |
| &nbsp;&nbsp;environment.description | env.desc | Overall setting, including abstract or virtual worlds | No | No |
| &nbsp;&nbsp;environment.location | env.loc | Place, spatial layout or imagined spatial context | No | No |
| &nbsp;&nbsp;environment.space_type | env.space | Indoor, outdoor, mixed, or non-spatial/abstract | No | No |
| &nbsp;&nbsp;environment.season | env.season | Visible seasonal setting; optional for nonliteral worlds | No | No |
| &nbsp;&nbsp;environment.weather | env.weather | Weather or analogous atmospheric phenomena | No | Yes |
| &nbsp;&nbsp;environment.time_of_day | env.tod | Visible day/night or other temporal setting cues | No | No |
| &nbsp;&nbsp;environment.lighting | env.light | Illumination, direction, contrast and visible light sources | No | Yes |
| &nbsp;&nbsp;environment.structure | env.struct | Architecture, terrain, layers or abstract geometric fields | No | Yes |
| &nbsp;&nbsp;environment.atmosphere | env.atm | Visible atmospheric qualities: fog, openness, density, haze | No | Yes |
| &nbsp;&nbsp;environment.mood | env.mood | Perceived scene tone, such as tranquil, tense or surreal | No | Yes |
| &nbsp;&nbsp;environment.motion | env.mot | Within-setting changes, each bound to one environment | Yes | Yes |
| environment_change | envc | Global timeline connecting distinct environment IDs | Yes | No |

**Fictional video example (9 seconds).** Two shots show the same office: daylight
enters a window and a desk lamp switches on. A cut reveals a snowy courtyard in
winter daylight, with falling snow and mist drifting between bare trees.

- `environment`: `office0_env` (0–6 s, two shots), `courtyard0_env` (6–9 s, one shot).
- `environment.description`: a compact furnished office; a snow-covered courtyard.
- `environment.location`: desk beside a window in an office; courtyard enclosed
  by brick walls. No city or other unseen location is inferred.
- `environment.space_type`: indoor for `office0_env`; outdoor for `courtyard0_env`.
- `environment.season`: visibly wintry for `courtyard0_env`; omitted for `office0_env`.
- `environment.weather`: falling snow for `courtyard0_env`; omitted for `office0_env`.
- `environment.time_of_day`: daylight for both environments.
- `environment.lighting`: office window daylight and warm desk-lamp light
  (after 3 s), separate entries; diffuse overcast light for the courtyard.
- `environment.structure`: office walls and desk arrangement; courtyard brick
  enclosure and bare-tree row, each attached to its own environment.
- `environment.atmosphere`: clear office air; thin courtyard mist.
- `environment.mood`: orderly and quiet office; hushed, bleak courtyard. These
  are visual impressions, not inferred emotions of a person.
- `environment.motion`: for `office0_env`, "At 3 s, the desk lamp switches on,
  adding warm light to the desk while window daylight remains present." For
  `courtyard0_env`, two entries: "Snow falls through the courtyard during 6–9 s";
  "Thin mist drifts between the bare trees during 6–9 s."
- `environment_change`: one global entry: "The video begins in office0_env and
  stays in that environment across the first two shots. At 6 s, the setting
  changes to courtyard0_env, which remains visible through the end."

**Fictional image example.** An abstract background contains violet planar bands,
a gold circular glow, and a translucent haze, with no horizon or recognizable place.

- `environment`: `abstract0_env`, one coherent nonliteral background.
- `environment.description`: violet geometric fields surrounding a gold glow.
- `environment.location`: omitted; no spatial location can be established.
- `environment.space_type`: non-spatial/abstract.
- `environment.season`, `environment.weather`, `environment.time_of_day`: omitted;
  the gold glow alone does not establish sunshine, sunset, or weather.
- `environment.lighting`: a bright gold center and softly illuminated violet edges.
- `environment.structure`: layered planar bands and a circular central field,
  as separate entries.
- `environment.atmosphere`: translucent haze over the geometric fields.
- `environment.mood`: serene and otherworldly, as separate visual-tone entries.
- `environment.motion`, `environment_change`: absent; haze in a still does not
  establish drifting or any transition between settings.


Mood describes perceived visual tone, not an assertion of a person's inner emotion.
For abstract, artistic or virtual backgrounds, use visible geometry, color fields,
spatial organization, light and atmosphere; do not force a real place, season or
weather where none is evidenced. Global rendering style is recorded separately.

A new setting is a new environment: an office followed by outdoors has two.
Multiple shots inside the same office may share one environment. Within-setting
changes such as lights switching on/off, moving foliage or shifting haze belong to
environment.motion, even when the visual change is conspicuous. Thus "changed
setting" means a change of environment identity, not every changed pixel or lighting
state. environment.motion entries are scoped to one environment and may be multiple
per environment; environment_change remains a single global attribute per artifact.
Attribute states may carry time ranges without themselves requiring flow.

### Relations, viewpoints and global attributes

| Category | short name | Definition and scope | Need flow in time? | Allow multiple? |
|---|---|---|---|---|
| identity_relations | idr | Spatial/contact/interaction relation between exactly two identities | Optional | Yes |
| identity_background_relations | idbr | Placement/scale/occlusion relation between one identity and one background | Optional | Yes |
| viewpoint | vp | One shot's viewing/framing configuration, including its internal stages | No | Yes |
| &nbsp;&nbsp;viewpoint.viewpoint_motion | vp.mot | Continuous viewing/framing evolution within one shot | Yes | No |
| viewpoint_change | vpc | Global timeline connecting viewpoints across shot transitions | Yes | No |
| style | st | Artifact-wide rendering medium and visual treatment | No | No |
| visible_text | txt | Actually legible text, its presentation and placement | No | Yes |

**Fictional video example (8 seconds).** In one office, a woman passes a red mug
across a desk to a man. The first shot begins wide, pushes toward her hands, then
arcs to her right. At 5 s, a cut shows a static close-up of the man holding the mug.
A wall sign reads “STUDIO”. The whole clip uses a watercolor rendering treatment.

- `identity_relations`: (`woman0_id`, `man0_id`) face each other across the desk;
  (`woman0_id`, `mug0_id`) holds then releases the mug;
  (`man0_id`, `mug0_id`) receives then holds it. Each entry has exactly two endpoints.
- `identity_background_relations`: (`woman0_id`, `office0_env`) occupies the left
  foreground of shot 1; (`man0_id`, `office0_env`) moves from background placement
  in shot 1 to filling most of shot 2. Each binds one identity and one background;
  the latter describes changing depicted placement, not necessarily walking.
- `viewpoint`: `shot0_vp` (0–5 s), the continuous wide-to-close arcing shot;
  `shot1_vp` (5–8 s), the static frontal close-up. Both depict `office0_env`.
- `viewpoint.viewpoint_motion`: one entry under `shot0_vp`: "The camera holds a wide
  view during 0–1 s, pushes forward toward the woman's hands during 1–3 s, then
  arcs to her right during 3–5 s without a cut." Absent under static `shot1_vp`.
- `viewpoint_change`: one global entry: "At 5 s, a hard cut changes shot0_vp's moving
  view of the handover to shot1_vp's static frontal close-up of the man holding the mug."
- `style`: one global watercolor treatment with soft pigment edges.
- `visible_text`: “STUDIO”, dark uppercase letters on the office wall sign,
  legible in the wide portion of `shot0_vp`.

**Fictional image example.** A watercolor image shows a woman to the left of a man
at an office desk. She holds a red mug in the foreground; a wall sign reads “STUDIO”.

- `identity_relations`: (`woman1_id`, `man1_id`) woman left of man;
  (`woman1_id`, `mug1_id`) hand in contact with mug handle. Each is a static relation.
- `identity_background_relations`: (`woman1_id`, `office1_env`) left foreground;
  (`mug1_id`, `office1_env`) small object near the lower center of the depicted scene.
- `viewpoint`: `shot2_vp`, one eye-level medium-wide view of the desk and people.
- `viewpoint.viewpoint_motion`, `viewpoint_change`: absent; there is no observed
  camera movement or transition in the image.
- `style`: one global watercolor treatment with soft pigment edges.
- `visible_text`: “STUDIO”, dark uppercase letters on the wall sign.
  The still does not establish a mug-passing action or who receives it next.


Each identity_relations entry binds exactly two distinct identity IDs; retain roles
when the relation is directional. Each identity_background_relations entry binds
exactly one identity ID and one environment/background ID. Multiple entries can
share the same endpoints. Relations may be static or evolve over a time interval.

Composition is absorbed into these relations: pairwise placement, overlap and
relative scale belong to identity_relations; foreground/background position,
subject placement, negative space and scale relative to the scene belong to
identity_background_relations. Framing and viewing direction belong to viewpoint.

Each shot has one viewpoint object; a single continuous take has one even if its
camera moves through many actual viewing angles. Qwen should describe every stage
of that continuous evolution inside viewpoint.viewpoint_motion. A static viewpoint
needs no motion entry. A cut creates a new viewpoint, even if the environment is
unchanged. viewpoint_change records cuts/transitions between viewpoints, not the
camera's movement within a shot. Separate temporal_structure and shot_structure
categories are omitted for now.

Attach source artifact/version, parent or endpoint IDs, applicable time ranges and
spatial scope, and observation confidence/unknown status. A timestamp on a static
state does not make it flow-dependent. Identical descriptions alone do not imply
shared provenance. A generation selects specific carrier artifacts for its selected
attributes; requested inheritance must be checked against actual generated media.

## Image/video selection

Proposed `p_video = 0.5` for balanced experiments, `0.25` for cheap initial smoke
runs. The image probability is `1 - p_video`. This is an experimental sampling
choice, not an estimated optimal value. Record the RNG seed and actual generated
counts. Small graphs need not exhibit the configured ratio exactly.

Sample target modality BEFORE selecting its compatible attributes. For image
outputs, sample frame-compatible properties from image and video artifacts. For
video outputs, both static and temporal properties are eligible. Sampling arbitrary
temporal attributes first and then forcing video would bias the requested ratio.
Changing a temporal attribute into a snapshot is a deliberate projection that needs
a new pose/state attribute; it is not preservation of the original motion.

For image output, group compatible properties from the same video and ask
`find_video_frame(video_path, description, qwen=shared_qwen)` for a frame satisfying
all of them. On None, retry with another candidate attribute combination or skip
the proposal; never silently use frame zero. Properties that occur at incompatible
times need separate frames/reference roles, not an invented common frame.

Export the selected PNG, retain the original artifact ID, frame index and timestamp
in source metadata, and pass it with the other image references to Flux. These
transport frames need not become standalone creative artifact nodes. For video
output, pass source videos directly to VDNH3. Generated audio is not observed or
asserted as a graph attribute in this phase.

Prompts and frame-search execution records remain in runs, outside the formal CPG.
No user dialogue, reasoning trace or global-constraint language is generated here.
