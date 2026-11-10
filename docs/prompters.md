# Natural-language generation prompt writers

`flux_prompter` and `h3_prompter` consume a rewritten description (`str`) and ordered
image/video references. They return the finished prompt as `str`, or `None` on an
invalid/declined response. There is no VLM JSON envelope on either side.

```python
from cocreation import Qwen, Reference
from tools.flux_prompter import flux_prompter
from tools.h3_prompter import h3_prompter

references = [Reference('coat', '/data/coat.mp4', 'video'),
              Reference('garden', '/data/garden.jpg', 'image')]
description = '''A woman wears a red wool coat. (from <Video 1>)
The courtyard has white roses. (from <Image 1>)
Soft warm morning light fills the scene.'''
with Qwen() as qwen:
    image_prompt = flux_prompter(description, references, qwen=qwen)
    video_prompt = h3_prompter(description, references, seconds=5, qwen=qwen)
```

For compatibility the first Python parameter is still named `attributes`, but its
value must now be natural-language text, not a list of mappings. Each referenced
line has `(from <Image N>)` or `(from <Video N>)`; unreferenced lines describe new
content. Every supplied reference must be used. Rewriter output is pruned and
renumbered by `prune_references` before these tools run. This can yield no references.

## Selected-description scope

Both prompters use only the supplied description as content. This is not image or
video captioning: every unselected subject, background, action, lighting detail,
style or camera property in the reference must be ignored. Media may only clarify
the selected property, not expand the scene. A brief guides wording, not additional
content. Do not add guide-example details, camera brands, lenses or film stocks.
Sparse input can yield a single short sentence; Flux has no minimum word count.

Each sourced requirement must immediately end with `like in <its source label>`;
repeat this for every requirement even when they share a source. No detached label
list, and H3 subject aliases do not replace inline source bindings. The parser
rejects missing labels and media-label mentions without the `like in ` prefix.
This checks label formatting, not semantic completeness or whether every clause
stays within the selected content; those remain model instructions.

For input `Green foliage is visible on the right. (from <Video 1>)`, the Flux output
should be `Green foliage is visible on the right, like in <Video 1>.` The bear,
rocks, wall or sunlight visible elsewhere in the reference must not be included.

## Media mapping

| Source order | Qwen observation / Flux prompter output | Python-prepared Flux input | H3 output |
|---|---|---|---|
| video coat | `<Video 1>` | `<Image 1>` | `<Video 1>` |
| image garden | `<Image 1>` | `<Image 2>` | `<Picture 1>` |

The plain-text Flux user prompt places observation media first and includes the
unchanged rewritten description. It does not expose a generator-label map. The
VLM must copy `<Video N>` / `<Image N>` source labels exactly, without renumbering
or adding ordinal aliases. The H3 prompt continues to explain its own label map. Flux uses at most four references;
H3 uses at most 32. Unsupported, unused or invented references are rejected.

Flux prompter returns one compact still-image prompt, up to 200 words, with
original `<Video N>` / `<Image N>` bindings. It observes original videos. The
caller replaces each video slot with a matching still, then calls
`flux_generation_prompt(prompt, original_references, generation_references)` from
`tools.flux_prompter` immediately before Flux generation. This Python helper
checks artifact order and all-image inputs, then simultaneously maps the original
labels to `<Image N>` slots across both source kinds. It never calls a VLM.
For `[video, image]`, `<Video 1>` becomes `<Image 1>` and original `<Image 1>`
becomes `<Image 2>` in one pass, avoiding cascading replacement collisions.
If no single frame matches the referenced properties, skip the proposal. Flux
receives its prompt and ordered reference images separately, as before.

H3 receives six plain-text sections with references: `subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`. Without references, it receives three sections:
`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`.
`retention_analysis` is the generator's desired reference-use specification, not a
CPG inheritance test. Optional `<Subject N>` labels must be defined once.

The first shot is `[Shot 1]`, with no timestamp. Later cuts use
`[Shot N] At MM:SS.mmm,` with consecutive numbers and increasing times below the
requested duration (5–15 seconds). A continuous camera move remains within one
shot. H3 uses `N/A` for soundscape and music unless the supplied description explicitly
requests them. It must not add unselected ambience, music, speech, camera movements
or actions just to fill its required sections. It cannot hear reference audio.

Both tool prompts begin with their expert role and purpose, require direct output,
include local task-specific examples, and reuse pinned repository guide material.
Examples and media text are data, not instructions. Invalid output prints a precise
diagnostic and optionally saves `diagnostics.jsonl`; this JSON is local logging,
never VLM conversation. `input.txt` records the actual text request. Shared Qwen
instances remain open; owned instances close even on failure. Dedicated system
prefix caches remain optional via `use_system_cache=False`.

## CLI

```bash
python tools/flux_prompter.py task.json
python tools/h3_prompter.py task.json --seconds 5
```

The local task manifest has `attributes` (a prose string), optional `references`
(artifact_id/path/kind records), and optional `user_prompt`. Paths resolve relative
to the manifest. This configuration file is parsed by Python; it is never sent as
JSON to the VLM. CLI output is direct prompter text or `CANNOT_FULFILL`. Flux CLI output still uses
original observation labels; callers must run the Python helper before generation.
In pipeline logs, `prompt.txt` is the original prompter output and
`generation_prompt.txt` is the final generator input. Both are also logged in
`debug.log`; the individual files survive normal completion with `--no-cleanup`.
