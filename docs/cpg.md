# CPG visual attributes and sampling (discussion draft)

This specifies the nested visual-attribute protocol; it is not a graph builder.

## Definitions

An attribute is grounded in one artifact/version. Its description approximates the
source media; identical descriptions do not imply shared provenance. Observed
associations and requested inheritance are distinct. Verify inheritance against
actual generated media. Audio is outside scope for the observing Qwen3-VL interface.

### JSON contract

The response has exactly `artifact_id`, `kind` (`image` or `video`), and `attributes`.
Persisted `attributes` is a dictionary with only the nine keys listed below.
Extractor and verifier pass 1 first emit a temporary first key inside `attributes`:
`"all identities and all environments draft"`, containing `identity` and
`environment` lists of `{uid, description}` for all clearly observed major objects
used in the response. Normal attributes follow, reusing these UIDs. The draft is
a closed inventory: subsequent identity/environment entries must come from their
respective draft lists, with no new objects or replacement UIDs. identity_relations
can reference only draft identities; identity_background_relations can reference
only a draft identity and draft environment. Relation descriptions cannot introduce
undeclared objects either. Omit an attribute/relation needing an undeclared object.
This is a prompt constraint, not a new Python rejection rule. The draft is an
object inventory, not reasoning. Python unconditionally removes this draft before
repair/validation and before assigning atom IDs; it never counts toward K or
provenance. A missing draft is tolerated. Verifier pass 2 returns matches only. This is a nested
protocol: dotted paths in the table describe nesting, never literal JSON keys.
Unknown, absent, unclear or inapplicable fields/categories are omitted. Never emit
null, empty strings, empty arrays or placeholder strings such as "unknown"/"None".
An artifact without clear observations has `attributes: {}`.

Every object in a top-level list requires `uid` and `description`; relation objects
also require `links`. Other fields are optional and included only when observed.
Every uid is unique across the entire artifact, using letters, digits, underscores,
dots or hyphens. Use meaningful local identifiers (for example `woman0_id`,
`office0_env`, `shot0_vp`); no category suffix format is required. Do not prefix the
artifact_id. All relation links resolve to existing objects in this artifact.
Nested string properties have no individual uid or parent_id; nesting binds them
to their identity/environment. Only relation objects have links. No time_range or
other metadata fields belong in this JSON; express grounded timing/spatial scope
in descriptions when useful. Never invent precise timestamps.

All strings except uid/link identifiers are concise natural-language descriptions
(< 10 words each, prompt guidance only; no length enforcement). Change descriptions name states
in natural language because states no longer have IDs. Global changes can mention
existing environment/viewpoint UIDs in their descriptions; they have no links array.

**Need flow in time?** means temporal change is necessary to express the attribute.
Optional means a relation may be static or evolve. A timestamp on a static state
does not make it flow-dependent. Images cannot establish motion or changes.
**Allow multiple?** follows the JSON type: lists allow multiple entries; strings
allow one value per parent (or per artifact for top-level strings). Multiple visual
qualities of a scalar field must be summarized together in its one description.

### Structure

```text
attributes: {
  identity: [{
    uid: str,
    description: str,
    appearance: list[str],
    body_pose: list[str],
    expression: list[str],
    appearance_change: str,
    expression_change: str,
    pose_change: str,
    motion: list[str],
  }],
  identity_relations: [{
    uid: str,
    description: str,
    links: list[str],
  }],
  environment: [{
    uid: str,
    description: str,
    location: str,
    season: str,
    weather: str,
    time_of_day: str,
    lighting: str,
    atmosphere: str,
    motion: list[str],
  }],
  environment_change: str,
  identity_background_relations: [{
    uid: str,
    description: str,
    links: list[str],
  }],
  viewpoint: [{
    uid: str,
    description: str,
  }],
  viewpoint_change: str,
  style: str,
  visible_text: [{
    uid: str,
    description: str,
  }],
}
```

### Temporal and multiplicity rules

| Field path | Need flow in time? | Allow multiple? |
|---|---|---|
| identity | No | Yes |
| identity.description | No | No |
| identity.appearance | No | Yes |
| identity.body_pose | No | Yes |
| identity.expression | No | Yes |
| identity.appearance_change | Yes | No |
| identity.expression_change | Yes | No |
| identity.pose_change | Yes | No |
| identity.motion | Yes | Yes |
| identity_relations | Optional | Yes |
| identity_relations.description | Optional | No |
| environment | No | Yes |
| environment.description | No | No |
| environment.location | No | No |
| environment.season | No | No |
| environment.weather | No | No |
| environment.time_of_day | No | No |
| environment.lighting | No | No |
| environment.atmosphere | No | No |
| environment.motion | Yes | Yes |
| environment_change | Yes | No |
| identity_background_relations | Optional | Yes |
| identity_background_relations.description | Optional | No |
| viewpoint | No | Yes |
| viewpoint.description | No | No |
| viewpoint_change | Yes | No |
| style | No | No |
| visible_text | No | Yes |
| visible_text.description | No | No |

### Scope and interpretation

- An identity is a perceptually coherent subject or meaningful object group.
  Do not split every texture or part into an identity. Description gives the overall
  identity/design; appearance covers clothing, color, material, shape and texture.
  Body_pose describes configuration/orientation; expression describes a visible
  facial/expressive state. Their lists may describe states at different times.
- Appearance_change, expression_change and pose_change describe observed transitions
  between the same identity's states, in order. Motion describes meaningful activity
  (action) together with its visible trajectory, speed, rhythm, phases or deformation.
  Interpret an action when the observed interaction supports it: for example,
  "She hands a package through the service window, extending both arms." This is
  more useful than only "Her arms move forward." Describe the action, its object/
  target and relevant movement together; no separate action field is needed.
  Do not infer hidden motives, an unseen recipient or events between sampled frames.
  A still image supports a pose/contact state, not a completed handover. Emit change
  fields only for actual observed changes, never merely "unchanged throughout".
- An environment is a coherent physical, imagined or abstract setting, independent
  of shot boundaries. Description summarizes it; location describes visible place
  and layout. Season must be visibly supported; late afternoon is time_of_day.
  Weather describes atmospheric phenomena; lighting describes illumination and
  sources; atmosphere describes visible haze, openness or density. Do not infer
  hidden places, seasons or weather. Abstract settings need not have these fields.
  The old space_type, structure and mood fields are removed.
- Environment motion describes changes within one setting (falling snow, drifting
  mist, lights switching on). Environment_change is a single global account of
  transitions between distinct environments. Multiple shots may share a setting.
- Identity_relations connect exactly two distinct identities; describe spatial,
  contact or interaction relations and preserve directional roles in description.
  Identity_background_relations connect one identity then one environment (background
  is an environment, not another object type). They describe placement, relative
  scale, occlusion or negative space. Multiple relations may share endpoints.
  Both kinds of relation can be static or temporal.
- Viewpoint segmentation follows human perception of meaningful framing/viewing
  stages, not a strict one-object-per-editing-shot rule. A continuous move from a
  wide view to a close-up has two viewpoints when those stages are perceptually
  distinct. Minor camera drift or every intermediate frame does not need a new
  object. Each description includes framing scale and viewing direction relative
  to the subject, plus relevant camera height/angle and composition. For example,
  "A wide shot of the woman from behind." Distinguish front, rear, side/profile,
  three-quarter and over-the-shoulder views when supported. Viewing direction is
  not the subject's body_pose: a woman facing left may still be viewed from behind.
  Do not invent an uncertain direction; retain the framing information that is clear.

  A cut creates a new occurrence even when returning to the same framing; keep
  viewpoints in temporal order. Do not merge distinct occurrences by similarity.
- Viewpoint_change connects the viewpoint UIDs in order and explicitly distinguishes
  continuous transitions (push-in, pan, orbit, etc., without a cut) from editing
  transitions (hard cut, dissolve, etc.). One continuous shot may contain multiple
  viewpoints. There is no separate viewpoint_motion field. A continuous move that
  does not create distinct stages can stay in one viewpoint description. Do not
  invent cuts, frame-by-frame segmentation or unseen transitions in sampled video.
  Viewpoint itself does not require time flow; image descriptions remain static.
- Style is one artifact-wide rendering-medium/visual-treatment description.
  Visible_text entries describe actually legible words plus presentation/placement;
  they have only uid and description, without carrier links.

Write only clearly observed, unambiguous information; omit anything uncertain.
Keep observations grounded in the media. Video sampling does not justify inventing
unseen events between frames. Unknowns are omitted, not inferred. This JSON is an
observation representation, not proof of shared-attribute edges in a CPG.

### Attribute matching across media

Verifier pass 2 uses human-intuitive conceptual and visual equivalence within each
attribute's described scope. Matching descriptions and supported visible traits
count as the same attribute; exact pixels, whole-object identity and incidental
details need not match. For example, large brown bears with shaggy fur can share
that description despite different poses or backgrounds. Unclear or contradictory
properties are not matched. Input media and atom IDs are temporarily relabeled
by media order; Python restores returned IDs before provenance updates.

### Conservative observation repair

Models should still omit unknown/inapplicable values. Before accepting a fresh
extractor or verifier-pass-1 observation, Python repairs harmless output issues
in place: remove null/None and whole-string "none", "null", "unknown" (case
insensitive), blank strings, empty containers and placeholder description-list
items; trim description whitespace; wrap a scalar string in an expected string
list or unwrap a single string in an expected scalar field. Never join multiple
scalar alternatives, truncate descriptions or invent missing values.
Objects missing required uid/description (or relation links) are omitted. Remove
relations referencing explicitly removed objects, and omit a global environment/
viewpoint timeline if its participating object category was pruned, rather than
rewrite it speculatively. Identifiers/endpoints are not coerced or guessed.
Unknown nonempty fields, wrong metadata, meaningful wrong types, duplicate UIDs,
unexplained dangling links, wrong endpoint types and image temporal fields still
fail validation. Invalid JSON syntax still fails; no speculative text repair.
The strict validator stays non-mutating; only fresh model responses use the repair
step before atom IDs/provenance are assigned. Verifier-pass-2 matches stay strict.
Description length is prompt guidance, not a reason to truncate or discard evidence.

### Viewpoint example: continuous movement followed by a cut

A rear wide office view continuously pushes into a close-up of a woman's hands. A hard
cut then reveals a courtyard. The first two viewpoints belong to the same uncut
shot; the third starts after an edit. A useful representation is:

```json
{
  "viewpoint": [
    {"uid": "view0_vp", "description": "Wide shot of the woman from behind."},
    {"uid": "view1_vp", "description": "Close-up of her hands from behind."},
    {"uid": "view2_vp", "description": "Static high-angle wide view looking down into the courtyard."}
  ],
  "viewpoint_change": "Continuous push-in view0_vp→view1_vp; hard cut view1_vp→view2_vp."
}
```

The scale change justifies two stages without implying a cut. A slow, slight drift
that stays a medium view can remain one viewpoint with no viewpoint_change.
Do not copy this example unless supported by the observed media.

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
