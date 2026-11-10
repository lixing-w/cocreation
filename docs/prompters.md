# Attribute-conditioned generation prompt writers

`flux_prompter` and `h3_prompter` return prompt text (`str`), or `None` when Qwen
reports incompatible/unobservable requirements or returns an invalid response.
Invalid inputs and operational errors raise. Neither tool generates media or
changes the CPG. Both accept original images AND videos, observed directly by Qwen.

```python
from cocreation import Qwen, Reference, VDNH3
from tools.flux_prompter import flux_prompter
from tools.h3_prompter import h3_prompter

references = [Reference('root_video', '/data/coat.mp4', 'video'),
              Reference('garden', '/data/garden.jpg', 'image')]
attributes = [
    {'id': 'root_video_coat', 'category': 'identity.appearance',
     'text': 'Red wool coat.', 'artifact_ids': ['root_video']},
    {'id': 'garden_setting', 'category': 'environment.description',
     'text': 'A courtyard with white roses.', 'artifact_ids': ['garden']},
    {'id': 'new_light', 'category': 'environment.lighting',
     'text': 'Soft warm morning sunlight.'},  # New attribute, no source media
]
with Qwen() as qwen:
    image_prompt = flux_prompter(attributes, references, 'A seated portrait.', qwen=qwen)
    video_prompt = h3_prompter(attributes, references, 'A slow turn toward camera.',
                               seconds=5, qwen=qwen)
# If video_prompt is not None:
# with VDNH3() as expert:
#     result = expert(video_prompt, references, seconds=5)
```

Input is a list of flat JSON objects, not the extractor’s artifact envelope.
Each selected attribute requires unique global `id`, known CPG `category`, and nonempty
`text`; `artifact_ids` may be omitted/empty for NEW properties. A sourced attribute
may list several carriers. Optional `parent_id`, `links`, and `time_range` retain
selection context. Selected subsets may refer to parent/state IDs outside the subset;
include enough context in captions or the brief to make those bindings intelligible.
Every supplied reference must be associated with an input attribute; dangling source
IDs, duplicate source IDs, unused media and audio-only references are rejected.
The caller supplies already-flattened entries: prefix each local `id`, `parent_id`,
and `links` endpoint with its origin artifact ID plus `_` (e.g. `video_0_bear`).
New source-free entries can use `new_lighting`. IDs remain opaque; explicit
`artifact_ids` identify media carriers. No automatic conversion is performed.

Both tools have fixed system prompts built from local official skill snapshots.
The brief, attributes, source map and H3 duration are user-message data, never part
of the reusable prefix. The original media are interleaved after their observation
labels in Qwen's user content. Owned Qwen instances close in finally; shared instances
stay open. Cache files, under the configured scratch root, are:

- `kvcache/flux_prompter/system_prompt.kvcache`
- `kvcache/h3_prompter/system_prompt.kvcache`

Use `use_system_cache=False` to disable caching. Qwen run logs retain the raw JSON
response. The caller receives only validated expert text. Validation checks envelope,
reference labels and H3 sections/shot cut times; it cannot prove semantic fidelity.

## Observation labels and generator labels

| Source order | Qwen observation | Flux output | H3 output |
|---|---|---|---|
| video root_video | `<Video 1>` | `<Image 1>` (future matching frame) | `<Video 1>` |
| image garden | `<Image 1>` | `<Image 2>` | `<Picture 1>` |

H3: pass the prompt and the original ordered references to VDNH3; no manual visual
token placement is exposed or needed. Its pinned Comfy tokenizer builds reference
label + vision blocks FIRST (video includes timestamps), then appends prompt text.
Mentions inside the text do not trigger visual-token insertion at those locations.

Flux: the prompter can observe videos, but Flux's existing generator accepts only
images. Before generation, replace each video slot with one frame selected to satisfy
its associated attributes, preserving the full source order. Do not send a video
to Flux or reorder the resulting image list. The prompter performs no extraction
and cannot guarantee that one suitable frame exists; a failed search must abort or
resample attributes. Inputs with temporal requirements may return None rather than
silently converting motion into a pose. More than four planned image slots are rejected
for this klein prompt contract, following BFL's published guidance.

FLUX.2-klein does NOT use Qwen3-VL's interleaved text/vision sequence: its Qwen3 text
encoder encodes prompt text; reference images are encoded into VAE latents and supplied
to the diffusion transformer separately. `<Image N>` is our textual reference convention,
not a special visual-token substitution. Prompts pair the label with natural-language
image numbering. The existing Flux wrapper already passes `prompt` and ordered `image`
inputs independently; callers need not assemble tokens.

Official/code sources:
- [BFL skill](https://github.com/black-forest-labs/skills/tree/8907d515b0ac270a988ec7a239add81ee13d6cba/skills/flux-image-best-practices)
- [MiniMax H3 skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/199b7d0e1fd95e7b3eafa8c15eca692534142427/skills/h3-prompt-writing)
- [Diffusers Flux2KleinPipeline v0.37.0](https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [Pinned H3 tokenizer](https://github.com/comfyanonymous/ComfyUI/blob/421a1c245c682c04d4325ba365f40c834c66f5b0/comfy/text_encoders/minimax.py)

## CLI

```bash
python tools/flux_prompter.py /path/to/task.json
python tools/h3_prompter.py /path/to/task.json --seconds 5
```

Task JSON contains attributes, optional references and optional user_prompt. Reference
paths in CLI manifests resolve relative to the manifest. Output is expert text, or
`null`; neither CLI invokes a generation expert.

## H3 sound and response format

H3 may add restrained, plausible new ambience and action sounds matching the scene,
and subtle music when appropriate. It respects explicit silence requests and avoids
abrupt or unrelated events, unrequested speech, and claims of hearing source audio.
Flux produces a still image and has no sound fields.

Qwen returns one JSON object with a single `prompt` string. With references, that
string contains six Ref2VA sections: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`. Without references, it contains three T2VA sections:
`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`.
The tool validates this envelope and returns the string itself.

## Viewpoint stages versus editing shots

A viewpoint is a perceptually meaningful framing stage. Continuous movement from a
wide view into a close-up may yield two viewpoint objects, linked by a continuous
viewpoint_change without a cut. Small drift need not split the stage. H3 must keep
these stages within one [Shot N]; new shot headings represent actual editing/shot
transitions. Preserve the transition type supplied by viewpoint_change.

Viewpoint descriptions include framing scale and supported viewing direction relative
to the subject, e.g. "A wide shot of the woman from behind." This is distinct from
body pose. Identity.motion includes interpretable actions with their visible movement,
e.g. "She hands a package through the service window, extending both arms." Action
interpretation must be supported by the interaction, not guessed hidden intentions;
no additional action field is introduced. Still images do not establish a handover.

Attribute descriptions are prompted to stay < 10 words; Python does not enforce
this length. Observations must be clearly visible and unambiguous; uncertainty
means omission. Imaginer's creative changes are desired properties, not source
observations. Extractor/verifier-pass-1 JSON starts attributes with a temporary
"all identities and all environments draft" inventory. Python removes it before
validation, atomization or provenance. Pass 2 remains match-only. Observation
prompts use a compact shared contract/schema instead of the full CPG document.

The draft is a closed object inventory: later identity/environment entries and
identity_relations/identity_background_relations endpoints must use draft-listed
objects and UIDs only. No new objects, replacement UIDs or undeclared relation
participants may appear after the draft. Omit attributes needing undeclared objects.
This is a prompt instruction; draft removal and Python validation remain unchanged.

On rejection, both tools print the exact validation failure and raw response to
stderr. With output_dir, they also append diagnostics.jsonl. Explicit null replies
may use {"prompt":null,"reason":"brief specific reason"}; successful replies still
have only prompt. A model-declined request is distinguished from parser/format
failures. The orchestrator includes these diagnostics in its attempt/debug records.
