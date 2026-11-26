# Sequence policy update, 2026-11-17

## Observed VLM timings

Measured from `scratch/runs/sequence-9230796-20261117T152413Z`:

| Phase | Wall time |
|---|---:|
| Video observation | 36.119 s |
| Dialogue writing and review | about 59.695 s |
| Expert prompt rewriting | 17.960 s |
| Combined VLM pipeline stages | 113.774 s |
| vLLM startup, separate | 156.190 s |

The dialogue/review model calls themselves account for 57.947 seconds over 152 requests (76 drafts and 76 reviews, 70 accepted turns). These are measurements with thinking disabled, not a prediction of thinking-enabled latency.

## New behavior

- Media kind is `image` or `video`. Logical placeholder `operation` stores create_reference/generate/generate_clip/assemble/export; nested generation contracts retain their executable operations. There is no trial/formal/final media category or rule selector.
- Person/object/environment remain subject content tags for attribute sampling. Neutral names are generated deterministically, preserving subject ids and disambiguating repeated names.
- Source descriptions remain available as evidence. Every writer explicitly analyzes conflicts and gives current `attribute_overrides` priority; it must omit conflicting source adjectives/clauses. Related subjects such as a carried bag use their own current state. Age-only edits explicitly preserve known hair and garment properties.
- Black-and-white rules and their dedicated rewrite/sampling/deferral/desaturation paths are removed. Black and white remain ordinary colors.
- Default layout is 1:1 images / 16:9 videos. Explicit observation constraints and the planner govern dialogue rule events.
- Persistent aspect settings can be replaced or given a one-request exception, but cannot be canceled. An explicit exception needs no extra approval turn. Its actual ratio must differ for the requested kind, and it expires after that request. An image-only exception does not affect videos in the same request.
- Thinking is enabled by `run_pipeline.sh` on the service and all VLM phases. Observation: 8192 tokens; dialogue/review: 4096; FLUX/H3 rewriting: 4096/6144. Reasoning is parsed separately from final text. Per-stage counters and latency are recorded for subsequent runs.

A fresh GPU generation run has not been started by this change. Existing run data is not migrated into the new schema.

## Verification

Applied to both the local and server repositories. The server's existing Slurm log paths are preserved.

- 179 relevant regression tests passed locally and on the server.
- All 19 server materialization tests passed, including real media assembly and resume/dependency handling.
- The final 62-test dialogue/reference subset passed locally and on the server, including all 5 reference-graph tests (203 distinct tests across the server checks).
- Shell syntax, whitespace checks and affected-file hashes were checked. No new model-generated media was produced as part of validation.
