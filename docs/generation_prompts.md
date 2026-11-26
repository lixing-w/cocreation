# 生成合约、Prompt 与思考链

对话台词写完后，`rewrite_prompts.rewrite_run()` 为每个占位符准备：可执行的**生成合约**、最终的 **H3 / H3-Image prompt**、assistant 的**思考链**，以及交付前的**开场白**。这一阶段只处理文本，不读取任何媒体。

代码：`scripts/dialogue_generation/generation.py`、`rewrite_prompts.py`、`formal_sequence.generation_context`、`planner_rules.py`、`tools/placeholder_prompter.py`、`tools/candidate_prompter.py`、`tools/prompt_reasoner.py`、`tools/{h3,h3_image,flux}_prompter_prompts.py`、`tools/_prompter.py`、`skills/`。

```bash
# 通常由 generate.py 自动调用；单独补跑或强制重写：
python scripts/dialogue_generation/rewrite_prompts.py RUN/dialogue --backend sonnet --think-mode medium [--force]
```

## 1. 生成合约（`generation.build_tasks`）

占位符的描述全部来自 planner，不是模型写的：资产描述来自 observation 和 `AssetSpec`，镜头描述来自 observation 的 shot，试拍描述由对应模块生成，规则效果来自 `describe_media`。`build_tasks()` 把每个占位符转换成一个任务（字段见 [data_contracts.md §3](data_contracts.md)）：

| 占位符 | 任务 |
|---|---|
| `create_reference` 图片，无输入 | `h3_image / text_generation` |
| `create_reference` 图片，有图片父版本 | `h3_image / local_edit`，`changes` = 属性、规则处理（`render.*`）和构图要求（`output.*`）的差异；全无差异时为 `copy`（无 backend） |
| `create_reference` 图片，父版本是视频 | `h3_image / reference_generation`，参考槽位带 `crop_query`，执行时从视频中裁出该主体 |
| `generate_clip`，首次 | `minimax_h3 / reference_generation`（有资产图）或 `text_generation` |
| `generate_clip`，已有旧版本 | `minimax_h3 / video_edit`：`<Video 1>` 是旧镜头，再加上变化的资产图；`changes` 用 `<主体>.<字段>` 记录主体属性差异 |
| `generate`（试拍） | `minimax_h3`，带 `scene_invention`：只能使用给定主体，自行设计一个简单可行的场景 |
| `assemble` / `export` | `concatenate`（多段）或 `copy`，没有 prompt；只保留画幅规则 |

视频固定为 4 秒。图片按 `layout` 区分：单个物体或人物是影棚产品照或人像（中性背景），单个环境是环境图，多主体是场景图。规则中的季节会转换成少量季节点缀（例如冬天在主体底部和表面加雪）。

参考槽位按类型分别编号：图片为 `<Picture N>`，视频为 `<Video N>`（FLUX 遗留路径用 `<Image N>`）。prompt 中的标签必须和槽位一一对应。

`construction_order()` 对 `dependencies` 做拓扑排序；检测到环时报错。

## 2. 上下文：只用"已经说出口"的内容

每个任务的 `sequence_context = generation_context(result, token)`：

- `visible_sequence`：本次交付之前**真实的台词**（去掉占位符）和每轮出现的媒体 id；不包含未来的台词，也不包含 planner 的内部描述。
- `current_user_request`：最近一条用户台词；`current_exchange`：最近 3 轮。
- `latest_rules`、`rule_applicability`：根据该媒体的规则快照计算出的当前生效要求（`planner_rules.media_rules`）。画幅会注明是系统默认还是用户覆盖。
- `output_position` / `output_count`：本轮第几个产物、共几个。
- 资产+镜头返工时还有 `production_workflow`（`updated_asset_image → reworked_clip → assembled_video`，各阶段之间存在依赖）。

## 3. Prompt writer（`placeholder_prompter.rewrite_placeholder`）

- System：`h3_image` 使用 H3-Image 官方 skill 的规则（`skills/h3-image-prompt-writing`，区分有参考和无参考），`minimax_h3` 使用 H3 官方 base/ref 指南（`skills/h3-prompt-writing`），再加上 `TASK_GUIDANCE`（不在 prompt 里写尺寸和采样参数；只使用列出的参考；编辑时保留无关内容；`scene_invention` 的主体列表是封闭的；按工作流阶段写作）。官方指南是快照文件，修改它们会改变 system prompt，进而让缓存失效（见 `skills/README.md`）。
- User：`writer_contract(task)`，只包含当前产物的计划（`planned_output`、`output_format`）和 `sequence_context`，不包含完整的规划 schema。
- 校验（`_prompter.parse_prompt`，加上任务相关检查），失败最多重试 3 次，只把正向要求反馈给模型：
  - H3 视频 prompt 必须包含规定的章节顺序：有参考时为 `subject_definitions / summary / retention_analysis / detailed_description / overall_soundscape / non_diegetic_music`，无参考时为 `integrated_multimodal_description / overall_soundscape / non_diegetic_music`；只能有一个 `[Shot 1]`；
  - 每个参考标签都必须出现；不能出现媒体占位符或 `<think>`；
  - 编辑必须明确写出保留哪些内容；`video_edit` 必须标注 `[video editing]`，并把 `<Video 1>` 当作源视频；
  - `changes` 里的每个新值都必须写进 prompt；
  - 修复重试时不能原样复制被拒绝的 prompt。

**同一轮的多个候选**（同一 backend 和 kind 的独立产物）由 `candidate_prompter.rewrite_candidates()` 联合撰写，确保各候选只写自己的那一个选项、彼此区分，并按用户给出的顺序对应。

## 4. 思考链（`tools/prompt_reasoner.PromptReasoner`）

先有最终 prompt，再写思考。reasoner 拿到真实对话、当前规则、实际参考绑定和最终 prompt（只用于对齐，不能作为理由引用），以"还没写 prompt"的第一人称视角写出三段：

1. `intent`："Let's analyze what the user is asking for." 用户要什么产物、主体、属性、动作、编辑范围和需要保留的内容；多个候选时只针对当前这一个；
2. `rules`："Now let's look at which cross-turn rules are effective." 哪些规则生效、如何满足，哪些旧规则为什么不适用；没有用户设定的画幅时，说明系统默认（图片 1:1、视频 16:9）；
3. `references`："Now let's gather visual references for this generation." 为什么需要每个参考：是编辑、延续已有身份，还是新身份不需要参考。

全文约 150 词（只靠 prompt 引导，不设硬性长度门槛）。输出 `{"think": {...}}`；Sonnet 后端使用 `PUBLIC_BRIEF_SYSTEM`，输出可公开展示的 production brief（同样三段，字段名为 `production_brief`），避免让模型"扮演私有思考"。

变体：

| 方法 | 用途 | 第三段开头 |
|---|---|---|
| `write` | 单个产物 | gather visual references |
| `write_delivery` | 同一轮多个相互独立的产物，共享一份规划，后续调用之间只插入一句过渡 | 同上 |
| `write_rework` | 资产图 → 镜头的依赖工作流，共享一份规划 | 同上 |
| `write_operation` | 纯拼接或复制；输入是计划中的 Python 调用，没有 prompt | "Now let's gather the source clips for this assembly." |

正式导出时，Python 会在思考末尾追加一句 "I have analyzed the user request, cross-turn rules, and applicable references. Now, I'll make a tool call to generate the …"，然后闭合 `</think>`，再输出独立的 `tool_call` segment。

## 5. 开场白

每个包含生成的 assistant 交付轮次都有一句 `generation_opening`（`QwenRealizer.production_opening`）：一句第一人称的话，确认请求并说明要做什么，不声称媒体已经做好；如果上一轮是澄清，会引用澄清的内容。纯拼接轮次使用固定句式，说明本轮只是拼接已有片段、不生成新媒体（如果有季节规则，还会说明新媒体规则不适用）。

## 6. 续跑与缓存

每个任务都有 `fingerprint`（任务内容），以及 `writer_input_hash`、`writer_system_hash`、`reasoner_policy_hash`。已是 `ready` 且 fingerprint 未变的任务直接跳过。联合候选组里只要有一个成员变化，整组都会重写。每完成一个任务，都会写一次 `dialogue.json`、`placeholders.json`、`generation_plan.json`。中间产物在 `prompt_rewriter/<token>/`（`system.txt`、`input.txt`、`attempt-N.json`、`diagnostics.jsonl`）、`prompt_reasoner/`、`operation_reasoner/`、`delivery_reasoner/`、`rework_reasoner/`。

## 7. 独立的 prompter 工具（遗留）

`tools/flux_prompter.py`、`tools/h3_prompter.py`、`tools/h3_image_prompter.py` 是早期的独立封装：输入一段自然语言描述和有序参考，输出 prompt（`str` 或 `None`），也可以用 `python -m tools.h3_prompter …` 从命令行调用。管线不再调用这些函数，但它们的 `*_prompts.py` 模块（system prompt 和 skill 嵌入）仍被 `placeholder_prompter` 复用。FLUX 只在旧 catalog 中出现；新 plan 的图片一律使用 `h3_image`。
