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
   Every successful nonempty draw is reserved in a Python set using an unordered
   combination of `(artifact_id, one-based description line number)`. Reordering
   lines or changing temporary reference labels cannot bypass deduplication; identical
   text at different line positions is still distinct. Previously used combinations
   retry inside the same attempt, even when their earlier generation failed. Zero
   draws are exempt because they sample no pool lines and invent fresh identities.
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
     prompt in addition to Qwen’s stochastic decoding. Explicit rewriter seeds
     reproduce both the chosen direction and sampling seed.
   - One actual source: preserve every sampled line and annotation unchanged.
     Python enforces this by returning the text without a VLM call; `mode.txt`
     records the passthrough. Multiple lines from one source still use this mode.
   - Multiple sources: preserve the input unless attributes from different sources
     strictly conflict. Resolve only that conflict with the smallest necessary change,
     preserving as many attributes as possible. Do not add unrelated elements or
     restyle the scene. Compatible differences are not conflicts. Retained properties
     keep their source markers; changed values must not falsely claim source support.
5. **Exactly one sampled line:** remove its source annotation and media **before rewriter**. Rewriter receives
   a nonempty new source-free requirement, not a zero-attribute invention request.
   Rewriter, prompter and generator receive no reference media or source labels, and no frame search is
   needed. The original carrier/line number stays in the attempt audit only.
   For multiple sampled lines, drop media no longer mentioned and simultaneously renumber surviving labels.
   Send this rewritten description and exactly its referenced media to the
   appropriate generation prompter: **flux_prompter** for images or **h3_prompter**
   for videos. Both consume prose and return prompt text directly. Flux prompter preserves
   all original `<Video N>` / `<Image N>` labels. After frame selection, Python
   converts those labels simultaneously to final `<Image N>` slots immediately
   before generation; no label conversion is delegated to the Flux VLM.
6. Call Flux or VDNH3. Move each returned media file immediately to
   `generated/artifact_N.<extension>` and use that permanent path thereafter.
   Even outputs later rejected by verifier remain here, with a `generated_media`
   record in their attempt. A generated output does not enter the pool until verified. For Flux, original videos are observed by the prompter but
   replaced by matching still frames before generation, without changing slots.
   Frame search consumes only the prose associated with that source; if no one
   frame satisfies it, skip the attempt. Never silently use frame zero.
7. **For every generated result**, run **attribute_verifier**, regardless of the
   sampled count (zero, one or multiple lines). It sees only the new media and the
   description sent to the prompter. Correct the description to match the result,
   remove all old-source labels, and retain essential observed details.
8. Add `(verified description, generated artifact)` to the pool, marked
   `in_pool: true`. Do not compare or deduplicate new observations against old
   attributes: identical wording on different artifacts remains separate evidence.
   Every line in this new description belongs to the newly generated artifact only;
   later samples bind it to that artifact, never to an earlier reference.

All VLM task payloads and responses are plain text, including frame selection
(decimal seconds / zero-based index, or `NOT_FOUND`). Every prompt begins with a
role and purpose, includes task-specific examples and requests only its output.
`NO_OBSERVATION` and `CANNOT_FULFILL` are explicit text failure sentinels. Malformed
or unavailable descriptions skip attempts; inference/operational errors checkpoint
and propagate. There is no attempt to prove semantic fidelity with a text parser.

Python manifests, checkpoints, generator metadata and diagnostic logs may still
use JSON on disk. These are application bookkeeping, never serialized attribute
messages presented to a VLM. `graph.json` version 2 records artifacts with `artifact_id`, `kind`, `path`,
`in_pool` and optional attempt/parent metadata. Every successfully verified result has an observed `description` and joins the
pool. Unverified outputs remain recorded in the attempt’s `generated_media` only.
Attempts include `sampled_lines` (artifact ID and stable line number),
`added_to_pool`, and whether verifier was requested/completed. `parents` records generation inputs, not proven inheritance.
With `--no-cleanup`, per-attempt `selection.txt`, `plan.txt`, `prompt.txt`, and
`observation.txt` preserve the natural-language stages. `generation_prompt.txt`
records the final Python-prepared generator input; `prompt.txt` keeps the original
VLM output. Both are also recorded in `debug.log`. By default, after all
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
scope. H3 sound or music requires an explicit request in the supplied description;
otherwise its audio sections use N/A.

Both generation prompters use only the supplied description, ignoring all other
media content. Each sourced requirement is immediately followed by `like in <its source label>`, repeated per requirement. Sparse selections stay sparse; no
camera/lighting/background embellishment or minimum word count is required.
