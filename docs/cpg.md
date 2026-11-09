# CPG visual attributes and sampling (discussion draft)

This file proposes the next CPG layer; it is not an implemented graph builder.
Audio is outside scope because the observing Qwen3-VL interface cannot hear it.

An attribute denotes a property grounded in a particular artifact/version. Its
language description is an approximation, not a replacement for the source media.
An observed association and a requested inheritance are distinct. Only comparison
of the generated artifact against its source can support a shared-attribute edge.

## Definitions

| Category | Definition and scope | Single-frame reference? |
|---|---|---|
| identity | Recognizable visual identity/design of a particular subject | Yes, if sufficiently visible; one view is incomplete evidence |
| appearance | Local hair, clothing, color, material, shape, texture of a specified subject/part | Yes |
| pose_expression | A visible body configuration, orientation or facial expression at an instant | Yes |
| environment | Visible setting, background structure, weather/environment state | Yes |
| relations | Spatial/contact relationships between named subjects | Yes for a visible state; a changing interaction is temporal |
| composition | Placement, relative scale, occlusion, foreground/background arrangement in the image | Yes |
| viewpoint_framing | Visible viewing direction, height and framing scale; not guessed physical camera specs | Yes |
| lighting | Visible illumination direction, hardness, contrast, day/night appearance | Yes |
| style | Rendering medium, stylization, palette and global visual treatment | Yes |
| visible_text | Actually legible text and its presentation/placement | Yes; do not infer illegible words |
| subject_action | What a subject does over an interval, including action phases | No; a representative pose is a separate attribute |
| subject_motion | Trajectory, speed, acceleration, rhythm or deformation over time | No |
| camera_motion | Change of viewpoint over time: translation, rotation, zoom, tracking | No |
| environment_motion | Motion of background/environment elements | No |
| temporal_structure | Ordering, synchronization and duration of events | No |
| shot_structure | Cuts, shot sequence and transitions | No |

Attach entity/part scope, source artifact/version, optional time range/frame and
spatial region, observation confidence/unknown status. Avoid merging two attributes
merely because their descriptions are identical. Shared attributes can have multiple
observed carrier artifacts; an individual generation selects a specific carrier.

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
