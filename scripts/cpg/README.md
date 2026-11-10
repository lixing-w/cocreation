# CPG reverse growth

This is the version 2 natural-language implementation. The pool contains
`(description, artifact)` tuples. Extractor observes roots; sampled sentences are
annotated with temporary media labels; attribute_rewriter resolves errors and
creates a new composition; the target Flux/H3 prompter writes direct expert text;
the generator produces media; attribute_verifier revises the description against
the actual result and adds the new pair to the pool.

See [pipeline, preserved attribute definitions and migration](../../docs/cpg.md)
and [prompter API and source mapping](../../docs/prompters.md).

```bash
python scripts/cpg/cpg_reverse.py scripts/cpg/examples/video_root.json \
  --output-dir scratch/runs/cpg-v2 --attempts 10 --sample 3 8 \
  --parent-count 2 --p-video 0.5 --seconds 5 --seed 0
```

The example manifest expects a local video at `scratch/data/samples/video.mp4`.
Paths are relative to the manifest. Use your actual media path. Optional
`description` or `description_file` supplies observed prose; omit it to use extractor.
`--keep` aliases `--sample`; `--add` and `--variate` have been removed.
`--brief` guides the creative rewrite and prompter; `--quiet` suppresses console
stage logs while retaining `debug.log` and checkpoints.

Outputs: `graph.json` version 2, `debug.log`, and per-attempt `selection.txt`,
`plan.txt`, `prompt.txt`, `observation.txt`, plus application metadata and model
logs. JSON files are local bookkeeping; no VLM is asked to consume or return JSON.
Only successfully verified results enter the pool. Skips count toward attempts.
Operational failures checkpoint before raising. The same Qwen, Flux and VDNH3
objects serve the whole run.

A deterministic fictional walkthrough requires no models or media files:

```bash
python scripts/cpg/walkthrough.py
```

Its output is checked in at [examples/walkthrough.txt](examples/walkthrough.txt).
It demonstrates source removal/renumbering and how a verified description becomes
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
