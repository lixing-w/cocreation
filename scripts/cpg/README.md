# Reverse CPG construction

`cpg_reverse.py` grows a graph from one or more image/video roots. The extractor
serves roots only. Each generated artifact goes through **two verifier passes**.
All choices and calls are made by Python; no model selects tools or runs an agent
loop. A single Qwen, Flux and VDNH3 instance is constructed before root processing
and reused until exit. Their existing wrappers load weights lazily.

## Parent budget and shared attributes

`--parent-count N` (alias `--max-parents N`, default 2) is an upper bound, not an
exact number. Every selected atom's actual carrier counts, including dependency
atoms. The sampler also enforces the budget directly, independently of the initial
parent draw. Effective limits are min(N, 4) for Flux and min(N, 32) for H3. Experts
receive only used retained-source media, so their reference count cannot exceed
this limit. Verifier pass 2 additionally sees the newly generated media.

Before sampling, atoms with the same confirmed origin (category, source_artifact_id,
source_attribute_uid) are deduplicated. Keep the representative from the numerically
earliest **candidate parent**, not an unselected historical ancestor. Same captions
alone are not evidence of sharing. Shared container descriptions redirect child and
relation bindings to the canonical owner; dependencies follow the representative IDs.
Unique attributes from later parents still require their own media and count toward
the budget. If deduplication leaves fewer than K atoms, skip this trial.

System prompts live in `tools/imaginer_prompts.py` and
`tools/attribute_verifier_prompts.py`, separate from execution/validation logic.

## Interfaces

```python
imaginer(selected_attributes, vary_ids, references, *, kind, qwen=None,
         output_dir=None, seed=0, ...)
# -> [{"id": "artifact_0_a0001", "text": "Violet glaze."}], [] for V=0, or None

attribute_verifier(mode="revise_observation", generated_media=new_ref,
                   target_attributes=target_json, qwen=shared_qwen, ...)
# -> complete observation JSON (artifact_id/kind/attributes), or None

attribute_verifier(mode="verify_inheritance", generated_media=new_ref,
                   observed_attributes=pass_1_json,
                   selected_sources=[{
                       "reference": old_ref,
                       "attributes": selected_original_atom_records
                   }], qwen=shared_qwen, ...)
# -> [{"observed_id": "artifact_2_a0000", "source_id": "artifact_0_a0000"}],
#    [] for no confirmed matches, or None for failed comparison
```

All tools accept optional observation/token budgets. They close only Qwen instances
that they create. Invalid model JSON, unknown/duplicate IDs, unexpected edits,
unrecoverable unsupported fields and invalid bindings return None. Extractor and
verifier pass 1 first repair harmless empty placeholders and scalar/list wrapping;
missing required objects and their dependent relations are omitted. Strict validators
and verifier pass 2 do not mutate existing observations or guess bindings. Operational/input errors
raise: missing files, inference failures and programming bugs are not disguised
as a creative failure. There are no internal retries.

The imaginer sees only the selected attributes, their source media and the exact
IDs it may edit. It returns text patches, never a reconstructed graph. A varying
attribute cannot change owner, category, relation endpoints or atom count. Its
new value is a desired property, not a claim that the source contains that value.
The prompt asks for creative, coherent changes while keeping other values fixed;
code checks the contract, not semantic creativity or coherence. Experts can reject
contradictory combinations. V=0 makes no imaginer inference call.

Pass 1 sees only new media and target JSON. It preserves supported wording/UIDs,
corrects inaccurate observations, omits uncertain/absent fields, and adds salient
observations. It does not write provenance. Its complete output is conservatively cleaned in place, then validated with
the existing observation protocol. No VLM retry is needed for harmless repairs. An empty observation is valid but will not be
used as a future parent.

Pass 2 sees new media, its revised observations, and **only selected original
attributes** with their old carrier media. For variated atoms this means the
pre-imaginer text, not the new requested value. It returns confident property-level
inheritance pairs, not a rewritten JSON. Empty matches is success, null is failure.
The code rejects unknown/unselected IDs, category mismatch, duplicate pairs and
inconsistent object/endpoint correspondence. Visual truth remains a VLM judgment;
there is no claim of perfect verification. No historical search for dropped or
previously unselected properties is implemented.

## Algorithm and data model

1. Read root media and observations, or call extractor once per root missing one.
2. For each bounded attempt, draw output modality, 1..max_parents parent nodes,
   a minimum retained count K and V variations from the inclusive configured ranges.
3. Flatten parents into a candidate pool. One atom is a scalar string, one string
   list item, or one relation/viewpoint/visible_text object. Container descriptions
   are atoms and count toward K; UIDs and link endpoints do not count separately.
4. Add atoms with dependency closure until the first legal selection has at least K
   atoms, then return immediately. Dependencies may take the total above K. Child fields require their owner
   description; relations require endpoint descriptions. Change fields conservatively
   require all corresponding source states and at least two states/environments/
   viewpoints. Single-valued global fields cannot be duplicated across parents.
   The bounded randomized search is dependency-biased, **not uniform** over subsets.
   It may skip a feasible but hard-to-find combination after 128 cheap attempts.
5. Pick V distinct selected atom IDs. If V exceeds the actual selected count, or
   no valid selection reaches K, skip. The log records requested `keep` and
   `actual_keep`; V is never silently clamped. No semantic projection.
6. Call imaginer once, validate exact edit coverage, and assemble target JSON.
7. Adapt to existing `flux_prompter` or `h3_prompter` flat records. Kept properties
   reference their current carrier media; variated properties are source-free desired
   values. Descriptions serve as container anchors without adding counted attributes.
   Include only references used by these expert records, in stable slot order.
8. For Flux, replace each selected video reference with one frame satisfying all
   retained properties from that source, using `find_video_frame` and shared Qwen.
   No match skips the attempt. Never use frame zero by default. H3 gets original
   ordered image/video references directly. Duration agrees with the expert input.
9. Generate once, run verifier pass 1, then pass 2. Commit only after both succeed.
10. Give each actual observed atom provenance and add the accepted media node to
    the candidate population for subsequent attempts.

Viewpoints follow meaningful human-perceived framing stages: a continuous wide-to-close
move may have two viewpoints connected by a continuous viewpoint_change. A cut is
also a transition, but must be labeled distinctly. Minor drift does not force a split.
H3 experts must not translate every viewpoint into a separate editing shot.

Image selection excludes flow-required categories. Static descriptions from video
can still describe incompatible times or include camera movement; the expert/frame
search must judge feasibility and may return null. Descriptive strings can overlap
semantically: K counts structural atoms, not independent bits of visual information.

Every frozen observation has deterministic atom IDs `artifact_N_a0000`, etc.,
assigned in schema order. Their JSON paths are stored in a sidecar; list positions
are never reused as IDs across different artifacts. Cross-artifact object UIDs
are namespaced before combination. Equal names never implicitly merge subjects.
The first version composes objects from multiple parents; it does not transplant
properties between arbitrarily chosen owners or let imaginer rewire relations.

`provenance[atom_id]` contains:

```json
{
  "path": ["attributes", "identity", 0, "appearance", 0],
  "source_artifact_id": "artifact_0",
  "source_attribute_uid": "artifact_0_a0001",
  "inherited_from": [
    {"artifact_id": "artifact_4", "attribute_uid": "artifact_4_a0003"}
  ]
}
```

`inherited_from` records the verified immediate carrier(s); `source_artifact_id`
propagates the numerically earliest confirmed origin. Select by numeric suffix,
not lexicographic order. Unmatched/new observations start at the new artifact.
Multiple confirmed sources are retained. Source media passed to generation and
verification are current carriers, not necessarily the earliest originating media.
The existing observation schema stays unchanged: provenance does not wrap strings.

## Fictional worked example

Run `python scripts/cpg/walkthrough.py` without any models. The full deterministic
fixture is `examples/walkthrough.json`, including every input/output and next pool.
This is a simulation of expected tool responses, not a claim about generated media.

Root 0 has a ceramic cup with red glaze and a curved handle, a tabletop, their
placement relation, and photographic style (6 atoms). Root 1 has a flower garden,
its location and morning light, and watercolor style (4 atoms).

Choose K=5, V=1 from N=10:

| Atom | Selected original | Action |
|---|---|---|
| artifact_0_a0000 | A ceramic cup. | Keep |
| artifact_0_a0001 | Red glaze. | Variate |
| artifact_1_a0000 | A garden filled with white flowers. | Keep |
| artifact_1_a0002 | Soft morning sunlight. | Keep |
| artifact_1_a0003 | Watercolor with soft pigment edges. | Keep |

Drop the curved-handle property, tabletop description, tabletop relation,
photographic style and garden location. Imaginer returns exactly one edit:

```json
{"edits":[{"id":"artifact_0_a0001","text":"Violet glaze flecked with tiny golden stars."}]}
```

Programmatically assembled target:

```json
{
  "artifact_id": "artifact_2",
  "kind": "image",
  "attributes": {
    "identity": [{
      "uid": "artifact_0_cup",
      "description": "A ceramic cup.",
      "appearance": ["Violet glaze flecked with tiny golden stars."]
    }],
    "environment": [{
      "uid": "artifact_1_garden",
      "description": "A garden filled with white flowers.",
      "lighting": "Soft morning sunlight."
    }],
    "style": "Watercolor with soft pigment edges."
  }
}
```

The Flux expert receives selected flat constraints; the violet-glaze atom has no
source claim. Suppose the generated image omits the stars and adds a “TEA” label.
Pass 1 returns the same structure except appearance becomes `["Violet glaze."]`
and it adds:

```json
"visible_text": [{"uid":"tea_label","description":"“TEA” appears in white letters on the cup."}]
```

Pass 2 compares these observations with the selected **red-glaze original**, cup,
garden, sunlight and watercolor values, plus both old media. Suppose it confirms
cup description, garden description, lighting and style, but not violet glaze or
the new text. Its four matches refer to observed atoms 0000, 0002, 0003 and 0004.
External code carries their original provenance forward. Violet glaze and “TEA”
start at artifact_2. The accepted six-atom observation enters the next pool.
The requested gold stars are retained only in the plan log, not as an observation.

## Running on a configured model host

Example root manifest (`roots.json`), with paths relative to this file:

```json
[
  {"artifact_id":"artifact_0","kind":"image","path":"cup.png","observation":"cup.json"},
  {"artifact_id":"artifact_1","kind":"video","path":"garden.mp4"}
]
```

Observation can be an inline object or JSON filename. Omit it to extract a root.
Artifact IDs must be `artifact_<nonnegative integer>`; numeric IDs must be unique.
Use the repository's configured Qwen/Flux/VDN model environment for real execution.

```sh
python scripts/cpg/cpg_reverse.py roots.json \
  --output-dir /tmp/cpg-run-001 --attempts 10 --seed 42 \
  --keep 3 8 --variate 0 2 --max-parents 2 --p-video 0.5 --seconds 5
```

The output directory must be new. `--attempts` limits trials, not successful nodes;
failed attempts consume IDs and computation and are never retried internally.
The seed determines sampling given the same sequence of tool outcomes, not bitwise
model reproducibility. Constructor/runtime options use existing model defaults;
`--vdn-python` optionally supplies the VDN bootstrap interpreter.

Each attempt records selection, edits/target, expert prompt, generation metadata,
revised observations and verified matches as it reaches those stages. `graph.json`
is checkpointed atomically after roots and every attempt. Skipped/generated-but-
unverified media remain in run logs but never enter the graph. Exceptions checkpoint
an error and propagate; all three model contexts are closed. This first version
records runs but does not resume an existing output directory.

Tests use fake media files and mocked model responses, including null at every
stage, invalid patches/matches, dependency closure, multi-parent bindings, model
reuse, original-versus-varied verification input and provenance propagation. They
do not require GPU inference. Run `python -m unittest tests.test_cpg_reverse`.

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
