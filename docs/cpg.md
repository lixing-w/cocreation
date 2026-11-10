# Natural-language CPG pipeline (version 2)

The pool is a list of `(description, artifact)` tuples. A description contains
short natural-language sentences, one intuitive attribute or coherent element per
line; blank lines may group them. The artifact is the actual image/video node.
Pool descriptions have no media labels, atom IDs, schema fields or inheritance
edges. The association is implicit in the tuple. Identical descriptions on two
artifacts remain two separate observations.

## Flow

1. **extractor** observes each root with the CPG definitions as selective guidance.
   It mentions essential, clearly visible attributes and omits uncertain,
   unobserved or nonessential details. It returns direct prose.
2. Sample **0–3 lines by default**, with probabilities **40%, 30%, 20%, 10%**.
   `--sample MIN MAX` changes the inclusive range; descending linear weights
   favor lower counts within that range. `--parent-count` caps distinct carriers, and each actual reference may contribute
   **at most two lines**. Draw without replacement and reject selections that exceed
   either cap or cannot supply K lines. Retry **inside the same attempt**, redrawing
   both K and attributes; `--max-sampling-retries` defaults to 128 extra draws after
   the initial draw. Only exhausted retries skip the attempt. The probabilities above
   describe proposals; accepted counts are conditioned on feasible selections.
   Zero sampling succeeds even
   with an empty pool. Zero-attribute exploration always generates an **image first**;
   `--p-video` applies only to nonzero samples.
3. For nonzero samples, assign temporary `<Image N>` / `<Video N>` labels, numbered
   separately by kind in first-selected-carrier order. Append `(from <Kind N>)` to
   each sampled line. Zero samples attach no media or existing pool descriptions.
4. **attribute_rewriter** has three modes:
   - Zero attributes: invent one new identity (person, garment, object or scene),
     in one or two concise sentences. Aim for elegant, beautiful, refined design,
     avoiding grotesquerie. Seeded category/form/character directions vary the actual
     prompt because Qwen currently uses greedy decoding, so seed alone is insufficient.
   - One actual source: preserve every sampled line and annotation unchanged.
     Python enforces this by returning the text without a VLM call; `mode.txt`
     records the passthrough. Multiple lines from one source still use this mode.
   - Multiple sources: preserve the input unless attributes from different sources
     strictly conflict. Resolve only that conflict with the smallest necessary change,
     preserving as many attributes as possible. Do not add unrelated elements or
     restyle the scene. Compatible differences are not conflicts. Retained properties
     keep their source markers; changed values must not falsely claim source support.
5. Drop media no longer mentioned and simultaneously renumber surviving labels.
   Send this rewritten description and exactly its referenced media to the
   appropriate generation prompter: **flux_prompter** for images or **h3_prompter**
   for videos. Both consume prose and return the finished expert prompt directly.
6. Call Flux or VDNH3. Move each returned media file immediately to
   `generated/artifact_N.<extension>` and use that permanent path thereafter.
   Even outputs later rejected by verifier remain here, with a `generated_media`
   record in their attempt. A generated output does not enter the pool until verified. For Flux, original videos are observed by the prompter but
   replaced by matching still frames before generation, without changing slots.
   Frame search consumes only the prose associated with that source; if no one
   frame satisfies it, skip the attempt. Never silently use frame zero.
7. **attribute_verifier** sees only the generated artifact and the exact rewritten
   description passed to the prompter. In one pass it corrects the description to
   match actual output, removes unsupported details and all old-source markers,
   and adds essential observed details. There is no inheritance-matching pass.
8. Add `(verified description, generated artifact)` to the pool. Subsequent sampling
   attributes every line to this new artifact, regardless of how it was created.

All VLM task payloads and responses are plain text, including frame selection
(decimal seconds / zero-based index, or `NOT_FOUND`). Every prompt begins with a
role and purpose, includes task-specific examples and requests only its output.
`NO_OBSERVATION` and `CANNOT_FULFILL` are explicit text failure sentinels. Malformed
or unavailable descriptions skip attempts; inference/operational errors checkpoint
and propagate. There is no attempt to prove semantic fidelity with a text parser.

Python manifests, checkpoints, generator metadata and diagnostic logs may still
use JSON on disk. These are application bookkeeping, never serialized attribute
messages presented to a VLM. `graph.json` version 2 persists each pool pair as an
artifact record with `description`, `artifact_id`, `kind`, `path`, and optional
attempt/parent metadata. `parents` records generation inputs, not proven inheritance.
With `--no-cleanup`, per-attempt `selection.txt`, `plan.txt`, `prompt.txt`, and
`observation.txt` preserve the natural-language stages. By default, after all
attempts finish and models close, intermediate `attempt_*`, `flux`, `vdn_h3` and
`qwen` directories are removed. Always retain `debug.log`, `roots/`, `graph.json`
and `generated/`. Root media are never moved or deleted. Interrupted/failed runs
retain intermediate diagnostics, but already returned generated media still use
permanent paths. `--no-cleanup` disables deletion, not media collection.
Transport frame files are temporary: after cleanup their source/timestamp stay in
the graph with `retained: false`, and their removed `path` is no longer advertised.
Attempts log `selection_draws` and `selection_retries`; config records the limit. Before cleanup, each text tool logs `input.txt`; observation tools
also save `response.txt`. All tools share one Qwen instance in the pipeline; the
Flux and VDNH3 adapters are also reused. Model sampling itself is not deterministic.

## Usage and migration

Root manifest paths resolve relative to the manifest. Each root has `artifact_id`
(`artifact_0`, etc.), `kind` (`image` or `video`) and `path`. An optional plain-text
`description` or `description_file` bypasses extraction. Do not provide both.
Legacy `observation` JSON is rejected with migration instructions; remove it to
re-extract the actual media. Old run checkpoints are not silently converted.

```bash
python scripts/cpg/cpg_reverse.py roots.json --output-dir scratch/runs/cpg-v2 \
  --attempts 10 --sample 0 3 --parent-count 2 --p-video 0.5 --seed 0
```

`--keep` remains an alias for `--sample`, but now counts sampled lines, not retained
atoms or dependency closure. `--add` / `--variate` are removed: zero-attribute rounds introduce new identities. `extract_attribute_nodes` remains an import/CLI alias for the
new `extractor`, returning `str | None`. `imaginer`, nested schema validation,
atomization, provenance matching and the two-pass verifier API have been replaced.
Checkpoints record `sampling_weights` and `zero_attribute_kind`; attempts record
`mode` (`new_identity` or `reuse_attributes`). The new verifier takes `generated_media` and `description` only.

## CPG attribute definitions

These concepts guide attention; they are not required fields or category headings.
Use intuitive sentence grouping instead of forcing every concept into a response.

- **Identity:** a perceptually coherent subject or meaningful object group, not
  every texture or part. **Appearance:** clothing, color, material, shape, texture.
  **Body pose:** configuration and orientation. **Expression:** visible facial or
  expressive state. Changes describe actual ordered transitions of the same subject.
- **Motion:** meaningful activity with its target and visible trajectory, speed,
  rhythm, phases or deformation. A supported handover is more useful than vague
  limb movement. Do not infer hidden motives, unseen recipients or missing events.
- **Identity relations:** spatial, contact or interaction relationships between
  subjects, preserving participants and directional roles in the sentence.
- **Environment:** a coherent physical, imagined or abstract setting, independent
  of shot boundaries. Location means visible place/layout; season and time of day
  require evidence. Weather is atmospheric phenomena; lighting is illumination and
  its sources; atmosphere is visible haze, openness or density. Abstract settings
  need not have weather or season. Do not infer hidden places or mood categories.
- **Environment motion:** changes within a setting, such as drifting mist or lights
  switching on. **Environment change:** transitions between distinct settings;
  multiple shots can share one environment.
- **Identity-background relations:** placement, relative scale, occlusion and
  negative space between a subject and its setting.
- **Viewpoint:** meaningful framing/viewing stages, including framing scale and
  direction relative to the subject (front, rear, profile, three-quarter,
  over-the-shoulder), plus relevant camera height, angle and composition. Viewing
  direction differs from body pose. Retain clear framing when direction is unclear.
- **Viewpoint change:** distinguish continuous push-in, pan or orbit from editing
  cuts or dissolves. A continuous wide-to-close move has distinct framing stages
  but remains one shot. Minor drift does not need a new stage. A return after a cut
  is a new occurrence; preserve order without inventing unseen transitions.
- **Style:** overall rendering medium and visual treatment. **Visible text:** only
  legible words, with their presentation and placement, in the original language.

Images establish static states, not completed actions or motion. Video sampling
cannot establish unseen events between frames. Audio is outside Qwen's observing
scope. New H3 sound design is a generation choice, not a source observation.
