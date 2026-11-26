# Planner 与跨轮规则

代码：`scripts/dialogue_generation/planner.py`、`cross_turn_rule.py`、`asset_variations.py`、`references.py`、`generation_requirements.py`、`conflicts.py`。这一层**完全不调用模型、不读写媒体**，输入 observation，输出一个已通过校验的 plan。

```python
from scripts.dialogue_generation.planner import build_plan, validate_plan
plan = build_plan(observation, seed=0, reworks=1, p_compile=0.65,
                  modules=None, extra_behaviors=True, cross_turn_rules=None)
```

| 参数 | 含义 |
|---|---|
| `seed` | 决定全部结构随机性 |
| `reworks` | 0–2，预留返工的资产数 |
| `p_compile` | 每渲染出一个新镜头（且已有 ≥2 个镜头）后，触发一次拼接预览的概率 |
| `modules` | 限定准备模块的子集 |
| `extra_behaviors` | 关闭后不再出现指代歧义、单镜头冲突、随机规则事件和复合返工 |
| `cross_turn_rules` | 单独开关随机规则事件；observation 中的显式 `constraints` 始终会被建模 |

另有两个仅供续跑旧 checkpoint 使用的参数：`legacy_rule_sampling`、`sample_time_of_day`。`generate.py --resume` 会依次尝试它们，以复现旧 plan。

---

## 1. 状态

`Planner` 维护的核心状态：

| 属性 | 含义 |
|---|---|
| `assets`, `shots` | 来自 observation（已规范化） |
| `initial_specs[aid]` | 资产首次提案的 `AssetSpec`。预留返工资产的首次提案在一个字段上偏离目标 |
| `pending_reworks` | 尚未修正的预留返工资产 |
| `versions[aid]` | 该资产全部图片版本 token |
| `tried_designs[aid]` | 已尝试的 `design_key` 集合，同一设计不会再生成第二次 |
| `ready[aid]` | 当前采纳的版本 |
| `rendered[sid]`, `bindings[sid]` | 镜头当前版本及其资产绑定 |
| `overrides[sid][aid]` | 范围为 `shot` 的返工只覆盖这个镜头的绑定 |
| `dirty` | 绑定或规则状态已过期、需要重渲的镜头 |
| `render_order` | 镜头首次渲染的顺序（"按生成顺序编号"的依据） |
| `comp_order`, `comp_media`, `comp_snapshot` | 当前剪辑的顺序、token 和片段版本 |
| `media`, `visible` | 全部占位符；已在对话中出现过的占位符 |
| `turns`, `events`, `selections` | 输出记录 |
| `rules` | `CrossTurnRules` 实例 |

两个底层原语：

- `add_media(kind, description, *, assets, shots, inputs, parent, goal_match, operation, bindings, generation_requirements)`：创建占位符。它会先调用 `rules.before_generation()`（可能插入一段规则冲突对话），再记录规则快照 `render_state / applied_rules`。父版本必须已可见，编辑深度不能超过 2。
- `emit(role, action, description, *, assets, shots, inputs, outputs, references, behaviors, …)`：追加一轮。这里会检查不能引用未来的媒体、每个产物只能出现一次、用户不能交付媒体，并在交付后更新 `visible`。

`asset_media()` 在 `add_media` 之上处理资产版本：记录完整属性、`variation`（变化字段的 before/after）、相对父版本的 `attribute_changes`，以及描述中提到的其他已采纳主体的状态（`related_subject_states`）。

`adopt()` 记录采纳。50% 的采纳是隐式的（不额外生成一轮 "用这个"）；回退、临时采纳和多选一始终是显式的，会生成 user `adopt|rollback` 和 assistant `confirm_selection` 两轮。

## 2. 主循环（`Planner.run`）

```text
启动 observation.constraints 中的规则
while 还有资产未就绪:
    rules.advance()                          # 可能设立/取消/挂起冲突一条规则
    从合法模块中抽一个，再抽资产（跨资产模块抽两个）
    single(...) / cross(...)
    rules.advance()
    render_due()                             # 所需资产都就绪的镜头立即渲染
    若有已进入镜头的预留返工资产：60% 概率立即 rework()，然后再 render_due()
rework_remaining()                           # 剩余预留返工；≥2 个时合并成批量返工
render_due()
rules.finish()                               # 停止抽样规则事件；已接受的规则保留到成片
最终拼接 / 重排到 target_order（可能出现编号歧义片段）
必要时 request_final_export → deliver_final_export
accept_project → confirm_final
rules.decorate(); decorate_attributes(); requested_outputs; assign_reference_materials(); validate_plan()
```

合法模块的条件：该资产存在可用的变化（`compatible()`，`direct`/`clarify` 总是合法），跨资产模块要求至少还剩两个未就绪资产。都不合法时回退到 `direct`。

## 3. 准备模块

| 模块 | 对话骨架 | 行为 |
|---|---|---|
| `direct` | 请求参考图 → 交付 → 采纳（30% 临时采纳） | 5, 15 |
| `candidates` | 请求两个候选（一个是目标，另一个是变体，顺序随机；默认出图，偶尔出视频）→ 交付 → 明确选择 | 4, 15 |
| `trial_edit` | 请求一个试稿（变体）→ 交付 → 否决并指定改值 → 编辑交付 → 采纳 | 15 |
| `clarify` | 模糊请求 → 助手问 "想要什么样" → 用户给出设计 → 交付 → 采纳 | 2 |
| `relative_preview` | 初稿（相邻等级）→ "再浅/大一点" → 助手给预览并确认 → 临时采纳 | 12 |
| `task_switch` | A 的试稿 → 挂起 A，切到 B（B 再走一个单资产模块）→ 回到 A 的试稿并修改 → 采纳 | 5, 9 |
| `temporary_rollback` | A 直接做 → 先准备 B → A 换一个变体 → 用 A 变体 + B 拍一段试拍视频（`operation=generate`）→ 回退到 A 原版 | 1, 5, 9, 10, 15 |

## 4. 镜头、冲突与拼接

`render_due(automatic=False)` 会渲染所有资产都已就绪且未渲染（或 `dirty`）的镜头：

- 新镜头：`request_shot`（引用各资产当前版本）→ `deliver_shot`。用户台词必须把镜头的场景、机位、动作说清楚（writer 会收到 shot staging）。
- 已有镜头更新：`request_shot_update`，intent 中带 `reference_updates`；如果是规则变化导致的，还带 `render_update`。
- 每个 plan 最多一次（30% 概率），把渲染请求换成一段**不可能同时满足的需求**（`conflicts.py`：framing、camera_motion、depth_of_field、shot_structure、aspect_ratio，按镜头中的资产类型筛选）：`incompatible_*` → `clarify_conflict` → `resolve_conflict`。
- 新镜头使已渲染镜头数 ≥2 时，以 `p_compile` 的概率触发 `compile_preview()`：60% 由用户指定一个打乱后的顺序，否则追加到上次剪辑末尾。
- 剪辑包含的镜头被更新后，`refresh_compilation()` 在原顺序中替换为新片段。

最终顺序：只有一个镜头时，成片就是这个镜头；否则如果当前剪辑顺序不等于 `target_order`，就发起最终拼接或重排。如果已有剪辑与目标包含相同的镜头集合，有机会生成**编号歧义**片段：用户说 "Swap shot 1 and shot 2" 或 "Move shot 3 after shot 1"，助手追问是按生成顺序还是按当前剪辑编号，用户回答按当前剪辑；`apply_numbering()` 给出确定结果。

## 5. 返工

`rework(asset)`（单个）：

1. 计算 `revision_target()`：目标属性。如果被规则锁定或已满足，则跳过，必要时生成 `attribute_already_satisfied`。
2. 如果目标设计在历史版本中已经存在，则改为**版本恢复**（`request_version_restore`，行为 1），不重新生成。
3. 否则 70% 概率先插入一段指代歧义：`ambiguous_reference` → `clarify_reference`（故意展示一张错误的图）→ `correct_reference` → `confirm_reference`（行为 3, 8）。
4. 用户指定某个已渲染镜头提出修改（按当前剪辑或生成顺序称呼）。该资产被多个已渲染镜头使用时，先由助手追问范围（`clarify_update_scope` → `choose_update_scope`：`shot` 或 `all_uses`）。
5. `revision_preview()` 生成资产新版本，`accept_revision()` 更新绑定，`deliver_revision()` 自动重渲受影响的镜头并刷新剪辑。最后由 assistant 发出一轮 `deliver_reworked_clips`，里面同时包含新资产图、新镜头和新剪辑（intent 中带 `asset_update_workflow`，下游会按"先图后视频"的依赖工作流生成和审查）。

`rework_group(assets, compilation_order=None, render_updates=None)`（批量/复合）：

- 多个返工合并成一个 `request_reworks`，每个修改编号为 `Edit i`，可以各自带歧义和范围问题。助手在 `clarify_reworks` 中一次问完，用户在 `resolve_reworks` 中一次答完，然后统一交付。
- **复合请求**：当某次拼接因为规则变化需要重渲片段时（`compilation_render_updates`），可以把最多两个预留返工并入这次拼接请求。用户在回答第一个澄清时可以追加第二个修改（`additional_edits`），助手对追加的修改再单独澄清。两轮澄清之间不交付任何视频。最终只渲染受影响镜头的并集：身份修改只作用于各自确认的范围，规则带来的渲染修改可以作用于其他镜头，但不改变它们的身份绑定。

## 6. 属性与变化（`asset_variations.py`）

- `CATALOG`：从 `attribute_catalog.json` 读取（`schema_version: 2`，字段数：person 19、object 13、indoor_environment 21、outdoor_environment 20），包括可选值、`ordered_values`（相对修改用）、类别内的相容性约束和 `summary_fields`。
- `fields(row)`：目录字段加上该资产的 `attribute_extensions`。`register_attributes()` 把模型新增的值登记到扩展中。
- `AssetSpec(row, attributes, change=None, focus=(), max_fields=7)`：一个 `str` 子类，文本形式为"资产描述; 若干关键属性"，同时携带完整的 `attributes` 和 `change`，planner 用它作为"版本设计"在各处传递。
- `VARIATION_VALUES`：只在可见字段上变化（object: color/material；person: clothing_color/age_group；environment: season/time_of_day/weather）。color 和 clothing_color 的采样权重是 2。
- `alternatives(row, base, relative, locked, tried)`：列出每个可变字段的合法新值。会排除 unknown 基线、被规则锁定的字段、语义相同的值、已尝试过的设计和违反相容性约束的组合。
- `canonical_value` / `design_key` / `changed_attributes`：语义比较（别名、颜色列表忽略顺序）。
- `relative_variation()`：在有序字段上取相邻等级，并生成 "Make X a little lighter" 式的请求和确认问题。

## 7. 跨轮规则（`cross_turn_rule.py`）

规则结构（`normalize_rule`）：

```json
{"scope": "global" | "partial",
 "selector": {"kinds": ["image"|"video"], "asset_ids": ["A1"]},
 "effect": {"season": "winter", "time_of_day": "night",
            "aspect_ratio": "portrait|landscape|square_assets_landscape_video|square_assets_portrait_video",
            "asset_attributes": {"A1": {"color": "red"}}},
 "text": ""}
```

`global` 不能带 selector，`partial` 必须带。`aspect_ratio` 按媒体类型投影（`output_aspect`）；默认 `square_assets_landscape_video`，即图片 1:1、视频 16:9。

`CrossTurnRules.advance()` 在模块边界调用，每次独立掷骰：

- 20%：设立新规则（最多同时 3 条、累计 6 条；随机生成的规则是 global 或只针对 image/video 的 partial；`aspect_ratio` 总是 global）。如果和现有规则重叠且取值冲突，会进入 `propose_rule → clarify_rule_conflict → resolve_rule_conflict(keep|override)`。
- 20%：取消一条已有的非比例规则（`cancel_cross_turn_rule → confirm_rule_cancel`）。
- 30%：对一条已有规则发起 **challenge**：只设置 `pending`，不立即产生对话。

`before_generation()` 由 `add_media` 调用。当最近一轮是用户的生成请求，且新媒体命中 pending 规则时：

- 直接例外（`once`，常用于比例）：在这条用户请求上标记 `rule_event: exception`，并临时覆盖本次生成的效果。
- 一般情况：插入 `clarify_rule_conflict`（助手提醒现有规则，问保留、仅本次、还是以后替换）和 `resolve_rule_conflict`。选 `override` 时，确认语（`confirm_rule_resolution`）会被合并进随后的交付轮次（`take_confirmation`）。

这样冲突总是附着在一个真实的生成请求上，不需要专门为演示规则制造媒体。

`snapshot(row)` 计算新媒体的 `render_state` 和 `applied_rules`。`decorate()` 在规划结束后用 `describe_media()` 生成最终描述（包括 `effective_attributes`），并为每个交付轮次及其请求轮次写入 `media_rule_context`。

## 8. 校验器

`validate_plan(plan)` 会重放整段对话，检查：

- 指针不能指向未来（输入、展示、上下文、歧义 oracle），每个媒体恰好出现一次；
- 编辑深度和祖先关系正确；资产版本属性合法；`goal_match` 版本属性等于目标；`attribute_changes` 等于与父版本的差异；`variation` 合法且确实改变了值；
- 批量返工协议：请求 → 澄清（必须覆盖全部必需的歧义和范围问题）→ 解答（不能改动目标）→ 交付（覆盖全部修改、范围与确认一致、镜头集合准确、绑定只在确认范围内改变、复合拼接的顺序和片段准确）；
- 编号操作在确认前不能出图，确认后的结果等于 `apply_numbering`；
- 镜头更新必须能说明是哪些参考变了，或者是哪个规则状态变了；
- 终点：最终顺序等于 `target_order`，最终镜头绑定指向正确的身份，最终剪辑没有过期版本；
- 最后调用 `validate_rule_timeline()`（重放规则事件：start/end/challenge/resolve/replace/exception 必须挂在正确的 action 上；每个媒体的快照必须等于当时有效规则的计算结果；描述必须等于 `describe_media` 的输出；规则生命周期必须完整闭合）和 `validate_reference_materials()`（重新计算因果依赖并比对）。

`generate.py` 在规划完成、对话完成、续跑时都会调用 `validate_plan`。

## 9. 对话行为

用户：`request_asset, request_candidates, request_trial, specify_goal, vague_request, begin_task, suspend_task, resume_task, switch_for_trial, request_variant, temporary_adoption_trial, reject_and_edit, relative_request, adopt, rollback, ambiguous_reference, correct_reference, request_rework, request_reworks, resolve_reworks, request_version_restore, choose_update_scope, request_shot, request_shot_update, incompatible_framing, incompatible_requirements, resolve_conflict, request_compilation, reorder_shots, replace_compilation_shots, ambiguous_shot_number, resolve_shot_number, request_final_export, accept_project, set_cross_turn_rule, propose_rule, resolve_rule_conflict, cancel_cross_turn_rule`。

助手：`deliver_asset, deliver_candidates, deliver_trial, deliver_edit, deliver_variant, deliver_exploration, deliver_resumed_task, preview_for_confirmation, confirm_selection, clarify_goal, confirm_switch, save_task, clarify_reference, confirm_reference, clarify_update_scope, clarify_reworks, deliver_reworked_clips, attribute_already_satisfied, deliver_shot, clarify_conflict, deliver_compilation, deliver_refreshed_compilation, clarify_shot_number, deliver_final_export, confirm_final, confirm_cross_turn_rule, clarify_rule_conflict, acknowledge_rule_decision, confirm_rule_resolution, confirm_rule_cancel`。

每个 action 在 writer 中对应一条 `ACT_GUIDANCE`（`tools/dialogue_realizer.py`），助手的交付类 action 还对应一条 `ASSISTANT_CONTENT`。

### 行为 id → 代码位置

`intent.behavior_ids` 由 `emit(..., behaviors=[...])` 显式写入；覆盖统计（`stats.user_behavior_*`）只统计**用户轮**。每种行为的含义和对话示例见根目录 [README 的用户模型一节](../README.md#用户模型)。

| id | 行为 | 写入位置（action） |
|---|---|---|
| 1 | 版本回退 | `adopt(rollback=True)` → `rollback` / `confirm_selection`（`temporary_rollback` 模块结尾）；`rework()` 中命中历史设计时的 `request_version_restore` |
| 2 | 目标澄清 | `clarify` 模块：`vague_request → clarify_goal → specify_goal → deliver_asset` |
| 3 | 指代歧义 | `ambiguity()`：`ambiguous_reference`、`clarify_reference`；批量返工中带歧义的 `request_reworks` / `clarify_reworks` |
| 4 | 一轮多候选 | `candidates` 模块：`request_candidates`、`deliver_candidates` |
| 5 | 先准备后组装 | `direct` 的 `request_asset` / `deliver_asset`；`request_shot`；`suspend_task`、`switch_for_trial` |
| 6 | 试拍后拼接 | `compilation()` 中的 `request_compilation` / `reorder_shots`、`deliver_compilation`；复合返工（带 `compilation_order`）的请求和交付 |
| 7 | 修改范围澄清 | `request_rework`、`clarify_update_scope`、`choose_update_scope`、`request_reworks / clarify_reworks / resolve_reworks`、`deliver_reworked_clips`、`replace_compilation_shots` |
| 8 | 指代纠正 | `correct_reference`、`confirm_reference`、`clarify_reference`；批量返工中歧义的澄清和解答 |
| 9 | 任务挂起与恢复 | `task_switch`：`suspend_task → save_task → … → resume_task → deliver_resumed_task`；`temporary_rollback`：`switch_for_trial`、`confirm_switch` |
| 10 | 多参考组合 | `request_shot`、`deliver_shot`、`resolve_conflict`、`temporary_adoption_trial`、`deliver_reworked_clips` |
| 11 | 依赖上下文的镜头编号 | `ambiguous_shot_number → clarify_shot_number → resolve_shot_number`；最终的 `reorder_shots` |
| 12 | 相对修改与预览 | `relative_preview` 模块：`relative_request`、`preview_for_confirmation` |
| 13 | 持续规则与例外 | `CrossTurnRules.emit` 发出的全部规则轮；`before_generation` 给附带规则冲突或例外的生成请求追加 13 |
| 14 | 冲突需求 | `render_due()` 中的 `incompatible_framing / incompatible_requirements → clarify_conflict → resolve_conflict`；规则冲突轮（`clarify_rule_conflict`、`resolve_rule_conflict`；非直接例外的生成请求也会追加 14） |
| 15 | 采纳决策 | `adopt` / `confirm_selection`、`reject_and_edit`、`request_variant`、`temporary_adoption_trial`、`deliver_exploration`、`accept_project` / `confirm_final` |

## 10. 扩展指南

- **新增准备模块**：在 `single()` 或 `cross()` 中实现，并加入 `SINGLE_MODULES` / `CROSS_MODULES`；在 `compatible()` 中声明合法条件；所有媒体都必须通过 `asset_media()` / `add_media()` 创建，所有轮次都必须通过 `emit()` 产生；需要采纳时调用 `adopt()`。
- **新增 action**：在 `tools/dialogue_realizer.py` 的 `ACT_GUIDANCE`（以及交付类的 `ASSISTANT_CONTENT`）中加入写作指导。如果它请求产物，还要加入 `tools/dialogue_intent.PRODUCT_REQUESTS`；如果它是一个规则请求，加入 `cross_turn_rule.GENERATION_REQUESTS`。
- **新增可变字段**：修改 `attribute_catalog.json` 和 `VARIATION_VALUES`，然后运行 `python scripts/dialogue_generation/observation_contract.py` 重新导出 `observation.schema.json` 和 `observation_prompt.md`。
- **新增规则效果**：修改 `EFFECTS`、`effect_text`、`visible_effect`、`describe_media`，并在 `generation.treatment()` 和 prompt writer 中体现。
- 每次改动后，在本地用多个 seed 运行 `build_plan`（纯 Python），并运行 `tests/test_dialogue_generation.py`、`tests/test_cross_turn_rule.py` 等测试。注意：plan 的变化会让旧 run 无法 `--resume`。
