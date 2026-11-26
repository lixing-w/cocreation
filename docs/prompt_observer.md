# Prompt observer

`tools/prompt_observer.py` converts H3 structured prompts and OpenVid captions into the same catalog-bound `observation.json` consumed by dialogue generation. It uses the shared Qwen/VLLM/Ollama transport **with no media attachments**. The video observer remains available for checking what a generated video actually contains.

From the repository root:

```bash
# H3 JSONL: select a zero-based row. The released 124-frame samples use 24 fps.
python -m tools.prompt_observer metadata.jsonl --row 42 --source-format h3 --fps 24 \
  --backend vllm --base-url http://127.0.0.1:8000 --output-dir scratch/runs/h3-prompt-42

# OpenVid-HD CSV: duration comes from the row's seconds (or frame/fps) metadata.
python -m tools.prompt_observer OpenVidHD.csv --row 42 --source-format openvid \
  --backend vllm --output-dir scratch/runs/openvid-caption-42

# A plain caption needs an explicit duration. Width/height are optional metadata.
python -m tools.prompt_observer caption.txt --duration 8 --width 1920 --height 1080 \
  --output-dir scratch/runs/caption-example
```

Accepts a text file, one JSON object, a JSON array, JSONL, or CSV. H3 rows use `prompt`; OpenVid rows use `caption`. `auto` detects H3 from its section headings. Output directories must be new. With vLLM, text extraction defaults to thinking enabled and an 8192-token output budget for field coverage; `--no-enable-thinking` opts into the faster mode, whose recall should be checked separately. Reuse one `PromptObserver` context and call `observe(record, new_output_dir, ...)` repeatedly for a corpus, so the same model/client stays loaded.

The conversion identifies principal assets and their shot membership, extracts each asset's attributes, and reviews the proposed values against the named subject and target text. Assets contain id, kind, name and description. Attribute replies are plain objects mapping existing fields to string values, without evidence fields or source quotations. Missing values are filled as unknown. The listed values are suggestions: a suitable new value in an existing field is stored in that asset's attribute_extensions and can be used by the planner, without changing the global catalog or adding new fields. Only the final reviewed values enter the planner vocabulary.

H3 uses `detailed_description` (Ref) or `integrated_multimodal_description` (T2VA). Reference definitions, retention analysis, audio and music sections cannot supply attributes; properties only available in reference images remain unknown. The target description controls changed appearances. Explicit `[Shot N] At MM:SS.mmm` boundaries are parsed in Python. Repeated markers in summary/retention sections do not create extra shots. Missing internal timestamps, invalid order, or more than four shots fail instead of silently inventing or grouping cuts. A caption without explicit H3 structure stays one shot even when it describes successive actions or camera movement.

Duration must come from metadata, an explicit `--duration`, or a frame count with explicit fps. There is no guessed fps or image geometry. The output preserves any supplied geometry. `video` is a `prompt://` identifier, and `observation_source` marks the result as unverified intended content. Required `assets[].frames` values are nominal midpoints of their assigned shots, **not sampled or observed frames**. Only a separate video inspection can establish actual cuts, timing and generated appearance.

Each run saves `input.json`, `source.txt`, parsed `source.json`, inventory, extracted and reviewed `attributes.json`, the final `observation.json`, and per-stage prompts, schemas, model replies and retries. No source video or reference image is read or downloaded. The resulting JSON can be passed to the existing dialogue planner; materializing references from an actual source video still requires a real video and verified timestamps.
