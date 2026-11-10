# Official prompting skill snapshots

- `flux-image-best-practices`: Black Forest Labs' official image prompting skill.
- `h3-prompt-writing`: MiniMax's official H3 base/reference prompt writing skill.

Original upstream files are preserved. Each `SOURCE.json` records the pinned commit,
source URLs and SHA-256 hashes. These are project reference materials, not changes
to the user's globally installed agent skills. The BFL MIT license is included;
H3 provenance is recorded without inventing a separate license for its documents.

Qwen system prompts embed a fixed selection of these files. Flux embeds SKILL.md,
core principles, FLUX.2 model guidance, I2I and multi-reference editing. H3 embeds
SKILL.md and both complete base/reference guides. No files are fetched at runtime.
Tool-specific constraints in `tools/*_prompter_prompts.py` adapt the guides to our
local model and visual-only CPG scope. Upstream examples are guidance, not requested
content. Changes to embedded files change system tokens and invalidate prefix KV.
