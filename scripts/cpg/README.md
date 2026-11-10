# CPG reverse growth

This is the version 2 natural-language implementation. The pool contains
`(description, artifact)` tuples. Extractor observes roots; sampled sentences are
annotated with temporary media labels; attribute_rewriter preserves one-source attributes and minimally repairs only
strict cross-source conflicts; zero samples create one new identity; the target Flux/H3 prompter writes direct expert text;
the generator produces media; attribute_verifier revises the description against
the actual result and adds the new pair to the pool.

See [pipeline, preserved attribute definitions and migration](../../docs/cpg.md)
and [prompter API and source mapping](../../docs/prompters.md).

```bash
python scripts/cpg/cpg_reverse.py scripts/cpg/examples/video_root.json \
  --output-dir scratch/runs/cpg-v2 --attempts 10 --sample 0 3 \
  --parent-count 2 --p-video 0.5 --seconds 5 --seed 0
```

The example manifest expects a local video at `scratch/data/samples/video.mp4`.
Paths are relative to the manifest. Use your actual media path. Optional
`description` or `description_file` supplies observed prose; omit it to use extractor.
`--keep` aliases `--sample`; `--add` and `--variate` have been removed.
`--brief` guides new-identity design and the prompter; `--quiet` suppresses console
stage logs while retaining `debug.log` and checkpoints.

Final outputs by default:

```text
run/
  debug.log
  graph.json
  roots/
  generated/
    artifact_1.png
    artifact_2.mp4
```

Every returned generated media file moves to `generated/` immediately, named by
artifact ID, including results later rejected by verifier. Graph artifact and
reference paths use these permanent locations. Rejected outputs remain listed
under the attempt's `generated_media`, without entering the sampling pool.

`--cleanup` defaults on. After all attempts finish, it removes `attempt_*`, `flux`,
`vdn_h3`, and `qwen` intermediates; the four entries above are always preserved.
Use `--no-cleanup` to retain per-attempt prose, generation metadata, and model logs.
Media collection remains enabled either way. Operational errors checkpoint and
retain diagnostics; already generated media stay in `generated/`.

Each sampled reference contributes no more than two lines. Invalid selections
redraw both count and attributes inside the same attempt; `--max-sampling-retries 128` allows up to 128 retries after the initial draw. Only exhaustion skips the
attempt. `selection_retries` and `selection_draws` record this in `graph.json`.
The same Qwen, Flux and VDNH3 objects serve the whole run. JSON is local bookkeeping;
VLM task input/output remains plain text.

A deterministic fictional walkthrough requires no models or media files:

```bash
python scripts/cpg/walkthrough.py
```

Its output is checked in at [examples/walkthrough.txt](examples/walkthrough.txt).
It demonstrates exact same-source preservation, zero-attribute exploration, and
how a verified description becomes
bound to its generated artifact, without inheritance matching.

Tool APIs:

```python
extractor(reference, qwen=shared_qwen)  # str | None
attribute_rewriter(sampled_text, references, kind='image', brief='', qwen=shared_qwen)
flux_prompter(rewritten_text, referenced_media, qwen=shared_qwen)
h3_prompter(rewritten_text, referenced_media, seconds=5, qwen=shared_qwen)
attribute_verifier(generated_media=new_reference, description=rewritten_text, qwen=shared_qwen)
```

Use `prune_references` on a successful rewrite before passing text/media to either
prompter. Pass that exact pruned text to verifier after generation. Image generation
from a referenced video additionally requires a matching exported frame in the same
reference slot. A missing frame skips the attempt.

Default counts 0/1/2/3 have probabilities 40/30/20/10%. Custom ranges also use
linearly decreasing weights. Zero samples always produce a new image first, even
with `--p-video 1`; nonzero samples use the configured video probability.
One-source rewriter calls are deterministic passthroughs, with no VLM inference.
Multiple sources only authorize minimal repair of strict contradictions; they do
not authorize new elements or broad creative rewriting.

Flux prompter copies source labels verbatim (`<Video N>` stays `<Video N>`).
After video frame selection, Python's `flux_generation_prompt` simultaneously maps
labels to the actual ordered Flux image slots. `prompt.txt` records original VLM
output; `generation_prompt.txt` records the converted generator input, and both
are logged in the permanently retained `debug.log`.
