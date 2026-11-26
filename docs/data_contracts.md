# 数据契约

本文按管线顺序列出各阶段的落盘文件和关键字段。只描述下游真正依赖的字段；完整内容以代码为准（定义位置写在每节标题后）。所有 JSON 都通过"写 `.tmp` 再 `replace`"原子写入。

一个 case 的 `dialogue/` 目录：

```text
dialogue/
  observation.json        planner 实际使用的 observation 副本
  plan.json               规划结果（无台词）
  rules.json              规则时间线
  placeholders.json       精简的可执行占位符清单
  dialogue.json           完整状态：plan + 台词 + 生成合约 + 统计 + 物化状态
  dialogue.md             对话完成后是占位符版本，物化后被正式导出覆盖
  generation_plan.json    按依赖排序的全部生成任务
  qwen/ | sonnet/         writer / verifier 调用日志（calls.jsonl 等）
  prompt_rewriter/  prompt_reasoner/  operation_reasoner/  rework_reasoner/  delivery_reasoner/
  media/                  已接受的最终媒体
  media_manifest.json     物化记录
  generation_attempts/<token>/attempt-NN/   每次生成尝试：output、inputs/、critique/、fixer/、attempt.json
  generation_records/     每次生成调用的原始回执
  turn_reviews/<leader>/<run>/              一轮多产物的顺序评审
  sequence.json           正式训练样本
  dialogue.annotated.md   带媒体链接的标注版
  dialogue.placeholders.md  物化前的占位符版本备份
```

---

## 1. observation.json（`tools/prompt_observer.py`，由 `planner.observation()` 校验）

```json
{
  "video": "prompt://<sha256>",
  "video_metadata": {"duration": 5.17, "width": 768, "height": 512},
  "observation_source": {"type": "prompt", "format": "h3", "source_id": "videos/000279.mp4",
    "sha256": "…", "video_verified": false, "frames_are_nominal": true,
    "timing_basis": "explicit_prompt", "duration_basis": "metadata"},
  "attribute_schema_version": 2,
  "assets": [{"id": "A1", "kind": "object", "name": "tote bag",
              "description": "…稳定的、目录之外的细节…",
              "frames": [1.2], "attributes": {"color": "blue", "material": "canvas", "…": "unknown"},
              "attribute_extensions": {"…": {"label": "…", "values": ["unknown", "…"]}}}],
  "shots": [{"id": "S1", "start": 0.0, "end": 2.5, "description": "…动作/构图/机位…", "assets": ["A1", "A2"]}],
  "target_order": ["S1", "S2"],
  "constraints": [],
  "description_policy": "…"
}
```

约束：

- `kind` ∈ `person | object | indoor_environment | outdoor_environment`；`attributes` 必须**恰好**包含该类别（加上扩展）的全部字段，每个字段一个字符串值；`unknown` / `not applicable` 表示未观测到。
- 1–4 个镜头；每个资产至少被一个镜头使用；镜头内资产不重复；`target_order` 是镜头 id 的排列。
- `frames`：1–2 个非负秒数（prompt 来源时只是名义值）。
- `constraints`：可选的初始用户规则，格式同[跨轮规则](planner.md)。
- observation 来自源输入（caption 或视频），但只作为语义和创意的引导。下游所有校验都针对 plan 和对话本身，生成的媒体不需要在画面上与源输入一致。
- 资产名要中立（只写类别）。planner 加载时会再次中立化名字和描述（`tools/attribute_authority.py`）。

## 2. plan.json / dialogue.json（`planner.Planner.run()`，`generate.run()`）

顶层字段（`schema_version: 6`）：

| 字段 | 含义 |
|---|---|
| `turns[]` | 对话轮次（见下） |
| `placeholders[]` | 全部媒体占位符，按创建顺序（见下） |
| `target_assets` | `{aid: {id, name, kind, description, attributes, attribute_extensions?}}`：终点属性 |
| `target_order` | 最终镜头顺序 |
| `final` | `{media, order, assets: {aid: token}, shots: {sid: {media, bindings}}}`：终点状态 |
| `rules` | 规则记录：`{id, scope, selector, effect, text, started_at, ended_at}` |
| `rework_plan` | 预留返工：`[{asset_id, field, initial_value, target_value}]` |
| `events` | 模块执行记录：`[{module, asset_ids, …}]` |
| `selections` | 采纳记录：`{asset_id, target_media, decision: accepted\|temporary, explicit, rollback, choice_required, after_message}` |
| `version_history` | `{aid: [token, …]}`：每个资产的全部图片版本 |
| `behavior_coverage`, `behavior_labels` | 覆盖到的行为 id，以及 1–15 的名称 |
| `seed`, `personas` | 规划 seed；`{user: {id, name, style, example}, assistant: {...}}` |
| `realization` | （dialogue.json）`{backend, model, model_mode, think_mode?, status, completed_turns, sampling, rewrite_seed, rewrite_seed_stride}` |
| `stats` | （dialogue.json）`stats.collect_stats()` 的覆盖统计 |
| `prompt_rewriting` | （dialogue.json）`{status, version, construction_order, completed, model, …}` |
| `materialization` | （物化后）`{status, manifest, official_sequence, official_dialogue, pruning?}` |

### turn

```json
{"id": 9, "role": "user", "content": "Can you make a reference image of the tote bag…",
 "intent": {
   "action": "request_asset", "module": "direct",
   "description": "Request the assistant to make a reference image of the tote bag. …",
   "reference_policy": "clear",
   "behavior_ids": [5], "asset_ids": ["A1"], "shot_ids": [], "asset_types": {"A1": "object"},
   "first_creation": true,
   "context_media": [], "display_media": [],
   "attribute_overrides": [{"subject": "tote bag", "attributes": {…}}],
   "media_rule_context": ["image: daytime as the time of day"],
   "requested_outputs": [{"kind": "image", "count": 1}]
 },
 "input_media": [], "output_media": []}
```

- `content`：realizer 写的台词。如果本轮重新展示旧媒体，末尾会追加一行 `References: [[...]]`；如果本轮交付媒体，末尾会追加交付的占位符。
- `intent.description`：给 writer 的意图草稿（英文），不是最终台词。
- `intent.action`：对话行为，完整列表见 [planner.md](planner.md)。
- `intent.context_media`：本轮语义上指向的历史媒体（监督信号，不作为附件展示）；`display_media`：需要作为附件重新展示的旧媒体。
- `input_media`：交付轮次中等于本轮产物的真实构造输入（`references.assign_reference_materials()` 重算），非交付轮次等于 `display_media`。
- `output_media`：本轮**新产生**的占位符，只有 assistant 轮次有，且每个占位符只出现一次。
- 其他可能出现的 intent 字段：`decision`、`scope`、`required_attribute`、`attribute_changes`、`variation`、`target_media`（歧义的真实所指，oracle，不展示）、`reference_correction`、`edits` / `resolved_edits` / `clarifications`（批量返工）、`compilation_order` / `render_updates` / `compound_request`、`numbering_operation` / `numbering_basis`、`conflict_type` / `generation_mode`、`shot_description`、`reference_updates`、`render_update`、`rule_event` / `rule_overlay` / `rule_request` / `requested_media_kind`、`asset_update_workflow`、`updated_clips`、`preview_media`。
- prompt 改写阶段会给 assistant 交付轮次加 `generation_opening`（真实生成前的一句开场）。资产+镜头返工轮次还有 `rework_transition`。

### placeholder

```json
{"placeholder": "[[image_0002]]", "kind": "image", "operation": "create_reference",
 "asset_ids": ["A1"], "shot_ids": [], "inputs": [], "edit_parent": null, "edit_depth": 0,
 "goal_match": false, "attributes": {…}, "effective_attributes": {…},
 "variation": {"type": "color", "phase": "alternative", "before": "brown", "after": "purple"},
 "base_description": "…", "description": "…最终生成指令…",
 "render_state": {"time_of_day": "daytime"}, "applied_rules": [{"id": "R2", "effect": {…}}],
 "reference_render_states": {}, "generation_requirements": {},
 "introduced_at": 10, "reference_materials": [], "reference_depth": 0,
 "generation": {…生成合约…}, "generation_prompt": "…"}
```

| `operation` | 含义 | 典型输入 |
|---|---|---|
| `create_reference` | 单资产图片（或候选视频）版本 | 无（首次）/ 被编辑的父版本 |
| `generate` | 非目标镜头的试拍视频（如 temporary_rollback 里的组合试拍） | 若干资产图 |
| `generate_clip` | 目标镜头 `shot_ids[0]` 的一个版本，`bindings` 记录每个资产绑定的图片版本 | 资产图；更新时加上旧镜头 |
| `assemble` | 拼接，`shot_versions` 记录每个镜头用的是哪个版本 | 各镜头视频 |
| `export` | 最终导出（内容等于输入） | 一个拼接 |

- `goal_match`：该媒体是否与终点一致（资产属性等于目标属性；镜头的全部绑定都 goal_match）。
- `edit_depth` ≤ 2：从首次提案起最多连续编辑两次。
- `description`：由 `cross_turn_rule.describe_media()` 根据 `base_description` 和 `render_state` 计算，带上当时生效的输出要求。
- `reference_materials`：真实因果依赖，全部早于本次交付。`reference_depth` 是 DAG 深度。镜头更新时依赖 = 旧镜头 + 发生变化的绑定。
- 源视频（guide）永远不会成为构造输入（`validate_reference_materials` 会检查）。

## 3. 生成合约：`placeholder.generation` / generation_plan.json（`generation.build_tasks()`，`rewrite_prompts.rewrite_run()`）

`generation_plan.json = {version: 11, status, construction_order: [token…], tasks: {token: task}}`。

| 字段 | 含义 |
|---|---|
| `operation` | `text_generation`（无参考）、`reference_generation`（有参考的新生成）、`local_edit`（图片编辑）、`video_edit`（视频编辑）、`copy`（属性与规则都没变时的复用）、`concatenate`（多段拼接） |
| `backend` / `model_id` | `h3_image` / `MiniMaxAI/MiniMax-H3-Image`，或 `minimax_h3` / `Comfy-Org/MiniMax-H3`；确定性操作为 `null` |
| `references[]` | 有序槽位 `{placeholder, kind, label, role, subjects?, asset_ids?, appearance?, crop_query?}`。图片在 H3 系列中用 `<Picture N>`，视频用 `<Video N>`，按类型分别编号；视频→图片的依赖带 `crop_query`，执行时从视频中裁出该主体 |
| `dependencies` | 执行依赖（等于 `reference_materials`） |
| `desired_description`, `desired_attributes`, `attribute_overrides` | 目标内容与当前属性（已叠加规则） |
| `changes[]` | `{field, before, after}`：编辑时的精确差异。`render.*` 是规则处理差异，`output.*` 是构图/姿态要求差异，`<subject>.<field>` 是镜头内主体属性差异 |
| `render_state` | 输出可见的规则效果（拼接只保留 `aspect_ratio`） |
| `shot_id`, `source_action`, `subject_bindings`, `subject_relationship_context` | 镜头任务的动作与主体绑定 |
| `scene_invention` | 非目标试拍：只允许用给定主体自行设计一个简单场景 |
| `layout`, `presentation` | 图片版式（product / portrait / environment / scene），以及影棚背景和季节点缀等呈现要求 |
| `seconds`, `output_seconds` | 视频固定 4 秒 |
| `clips`, `assembly_version` | 拼接的有序片段 |
| `sequence_context` | `formal_sequence.generation_context()`：真实的已说出口的历史、当前用户请求、`latest_rules`、`rule_applicability`、`output_position/count`、返工工作流阶段 |
| `prompt` | 最终生成 prompt（确定性操作为 `null`） |
| `reasoner` | `{think: {intent, rules, references}, generation_prompt, latest_rules, …}`：思考链。共享时，非 leader 只有 `reasoner_shared_with` |
| `operation_reasoner` | 拼接操作的思考链 |
| `workflow_leader` / `delivery_leader` / `writer_group_version` | 返工工作流、一轮多产物、联合候选的分组标记 |
| `fingerprint`, `writer_input_hash`, `writer_system_hash`, `reasoner_policy_hash` | 续跑缓存键 |
| `status` | `pending` / `ready` |
| `native_size`, `output_size` | 物化时写入的像素尺寸 |

`placeholders.json` 是这份合约的精简投影（`catalog.placeholder_catalog()`）：每项包含 `placeholder, kind, description, operation, asset_ids, shot_ids, reference_depth, reference_materials, backend, model_id, prompt, references`（去掉了属性和描述）等。当 `dependencies` 与 `reference_materials` 不同时，会额外列出 `dependencies`。`python scripts/dialogue_generation/catalog.py RUN/dialogue` 可以只刷新这个文件。

## 4. media_manifest.json（`materialize.materialization_session()`）

```json
{"version": 1, "status": "running|complete|partial|failed", "completed": 14, "elapsed_seconds": 1234.5,
 "construction_order": […], "operations": {…}, "output_sizes": {token: [w, h]},
 "settings": {"seed": 17, "native_video_output": true, "video_size": [1024, 576],
              "h3_inference_policy": {"new_video": {…}, "video_edit": {…}}, "media_serving": "sglang",
              "media_review": {"enabled": true, "max_generations": 3, "model": "claude-sonnet-5-5", "effort": "medium"}},
 "video_brightness_matching": {"enabled": true, "policy": "…", "anchors": {…}},
 "h3_request_recipes": {token: [recipe]},
 "media": {token: record},
 "pruning": {"exhausted": {token: error}, "omitted_media": […], "removed_turn_ids": […]}}
```

`record`：

| 字段 | 含义 |
|---|---|
| `status` | `running` / `pending_review` / `complete` / `failed` / `omitted` |
| `path`, `sha256`, `size_bytes`, `fingerprint`, `seed` | 最终文件与缓存键 |
| `attempts[]` | 每次生成：`{attempt, placeholder, kind, prompt, path, sha256, references, options, effective_parameters, inference_recipe, generation_seconds, model_reload_seconds, critique, retry_edit_prompt?, reasoner?}` |
| `review_status`, `accepted_placeholder` | 评审结论；被接受的尝试版本，如 `[[video_0004_01]]` |
| `superseded_attempts`, `critique_history` | 被后续结论取代的尝试和审查（审计用，不进入正式序列） |
| `turn_reviews`, `turn_review_leader` | 一轮多产物的顺序评审轮次（记录在 leader 上） |
| `tool_call`, `operation_reasoner` | 确定性操作实际执行的 Python 调用（`tools.video_editing.concatenate` / `shutil.copy2`） |
| `brightness_matching` | 视频编辑亮度对齐报告 |
| `generation_record` | `generation_records/*.json` 路径 |

尝试版本命名：第 1 次沿用原 token，第 n 次为 `[[kind_NNNN_{n-1:02d}]]`（`generation_retry.attempt_placeholder`）。

## 5. sequence.json（`formal_sequence.export_sequence()`）

```json
{"version": 4, "status": "complete|incomplete|partial|failed",
 "turns": [{"id": 10, "role": "assistant", "content": "…", "segments": [
    {"type": "text", "text": "…generation_opening…"},
    {"type": "thinking", "source": "prompt_reasoner", "text": "<think>\n…\n</think>"},
    {"type": "tool_call", "tool_calls": [{"name": "minimax-h3-image",
        "arguments": {"references": [{"kind": "image", "placeholder": "[[image_0001]]", "label": "<Picture 1>", "role": "…"}],
                      "prompt": "…", "parameters": {"ratio": "1:1", "res": "4096p"}}}],
     "text": "<tool_call>\n{…}\n</tool_call>"},
    {"type": "media", "kind": "image", "placeholder": "[[image_0002]]", "path": "generation_attempts/…/output.png"},
    {"type": "thinking", "source": "media_critique", "text": "<think>…</think>"},
    {"type": "text", "text": "…原计划的交付台词…"}]}],
 "media_aliases": {"[[image_0002]]": {"placeholder": "[[image_0002_01]]", "path": "…", "after_turn": 10}}}
```

- 工具名：`minimax-h3-image`、`minimax-h3`；确定性操作使用实际函数名。`parameters` 只保留 `seconds / ratio / res`；`references` 不含本地路径。
- `thinking.source`：`prompt_reasoner`（首次调用的规划）、`generation_transition`（无规划时的过渡句）、`operation_reasoner`、`media_critique`。
- 被拒绝的尝试会依次出现：调用 → 媒体 → 审查思考 → 拒绝说明（text）→ 修复调用（第 2 次起不再重复规划）→ …
- 交付的媒体全部被接受后，才追加该轮原计划的交付台词。一旦某个交付未完成，导出就在这一轮截止，不会发布还没生成的未来台词。
- `media_aliases`：被接受的尝试不是第 1 次时，后续轮次里该 token 指向的实际版本。
- `dialogue.md` 是同一内容的 Markdown 版，媒体用 `<img>` / `<video>` 标签。

## 6. VLM 阶段状态文件

- `<case>/vlm_state.json`：`{case, phase, stage, elapsed_seconds, phase_seconds, job_id, node, error?}`，`phase` ∈ `observing | dialogue_and_generation_prompts | validating_generation_plan | complete | failed | cancelled`。
- run 级别文件见 [native_batch.md](native_batch.md)。
