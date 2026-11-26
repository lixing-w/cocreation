# Observation：从文本或视频得到目标描述

Observation 是 planner 唯一的输入，描述一个 1–4 镜头的目标视频：有哪些资产、每个资产的完整结构化属性、每个镜头的动作和机位，以及最终顺序。契约见 [data_contracts.md](data_contracts.md)。

| 来源 | 工具 | 生产使用 |
|---|---|---|
| H3 结构化 prompt / OpenVid caption（纯文本） | `tools/prompt_observer.py` | ✅ native batch |
| 真实视频（抽帧） | `tools/video_observer.py` | 可选，用于有真实视频的数据或检查生成结果 |
| 手写 | `scripts/dialogue_generation/examples/*.json` | 测试与示例 |

## 1. Prompt observer

```bash
python -m tools.prompt_observer INPUT --source-format h3|openvid|auto [--row N] [--fps 24] \
  [--duration S] [--width W --height H] --output-dir NEW_DIR \
  --backend sonnet|vllm|ollama|local [--think-mode low|medium] [--max-assets 8] [--max-tokens 8192]
```

输入可以是 text、单个 JSON 对象、JSON 数组、JSONL 或 CSV（`--row` 从 0 开始计数）。H3 记录读取 `prompt` 字段，OpenVid 记录读取 `caption` 字段。native batch 调用时会传 `--source-format`、`--seed`、`--max-tokens 8192`、`--context-tokens`、`--timeout 600`，H3 还会传 `--fps 24`。vllm 后端默认开启 thinking。

### Python 负责的解析（`prepare_source`）

- H3：只读取 `detailed_description`（Ref2VA）或 `integrated_multimodal_description`（T2VA）。`subject_definitions`、`retention_analysis`、音频、音乐等章节都不作为视觉证据。
- 镜头：只认显式的 `[Shot N] At MM:SS.mmm`。编号必须从 1 开始连续递增，第 2 个镜头起必须带时间戳，起点严格递增且小于总时长，最多 4 个镜头；不满足时直接报错，不会去猜切点。OpenVid caption 永远只算一个镜头，即使其中描述了多个连续动作或运镜。
- 时长：来自元数据（`duration_seconds / seconds / duration`）、`--duration`，或帧数加显式 fps。不猜测 fps。

### 模型负责的抽取（`PromptObserver.observe`）

1. **Inventory**：识别主要资产（`person / object / indoor_environment / outdoor_environment`，最多 `max_assets` 个），给出 id、中立的类别名、描述，以及每个镜头的完整描述和资产列表。有一些分类约定：动物、怪兽、载具都算 object；一片连贯的景观是一个环境，而不是冰、水、植物等多个物体；完全相同的多个实例只记为一个资产，在镜头描述里写明数量。
2. **逐资产属性**：只提供该资产相关的镜头原文。目录中的取值只是建议，可以在已有字段中填一个新值（会登记到 `attribute_extensions`），但不能新增字段。未明确的细节填 `unknown`，不能根据名字、国籍、风格推测外观。
3. **独立复核**：拿原文重新抽取每个字段，把第一遍结果当作"不可靠的候选"来核对：补回漏掉的、纠正错误的、保留有依据的。
4. Python 校验：`validate_attributes`（字段完整、取值合法、满足类别内相容约束），登记扩展值。
5. **中立化**（`attribute_authority.name_assets` / `neutral_asset_descriptions`）：资产名只描述类别（"teacup" 而不是 "porcelain teacup"；人物统一叫 person 类）。
6. **描述分离**（`observation_descriptions.separate_descriptions`）：在属性确定之后做一次纯文本整理，把可编辑的外观（颜色、材质等）从资产描述和镜头描述中移除，只保留在属性里；动作、机位、方向、速度、数量等仍保留在镜头描述中。这一步不改变属性、身份、绑定和时间。
7. 最后用 `planner.observation()` 校验，写出 `observation.json`。

输出目录必须是新目录。每个阶段的 prompt、schema、原始回复和重试都保留下来（`inventory/`、`asset_NN/`、`asset_NN_review/`、`description_separation/` 等），另外还有 `input.json`、`source.txt`、`source.json`、`inventory.json`。

`observation_source.video_verified=false`，`frames` 是各资产所在前两个镜头的名义中点，不是真实抽帧。

### 源输入只是引导

无论输入是 caption、H3 prompt 还是真实视频，它都只是**语义和创意的引导**：决定这条 sequence 讲什么、有哪些资产、拍哪些镜头。observation 定义的是 planner 层面的终点（采纳版本的属性、镜头、最终顺序）。最终 sequence 里生成的视频**不需要**在画面上还原源视频或 caption，只要与对话本身自洽就达到目的：生成的每个资产、镜头、修改都和对话中说的一致，并经过审查。因此不需要验证源视频（`video_verified=false`），源视频也从不作为生成输入，审查时只对照对话和实际参考，不对照源输入。

### 为什么要这样拆分

- 让编辑能真正落地：planner 只在属性上做变化。如果描述里还写着 "blue tote bag"，把包改成棕色之后，旧的镜头描述就会和新版本冲突。
- 让 prompt 不被源描述污染：下游所有写作环节都把当前属性当作权威（`ATTRIBUTE_AUTHORITY`），源描述只提供与属性不冲突的细节。

## 2. Video observer

```bash
python tools/video_observer.py VIDEO.mp4 --output-dir NEW_DIR --backend vllm [--frames 24 --max-side 768]
```

在整段视频的实际显示时间上均匀抽取最多 `--frames` 帧（含首尾），按时间顺序作为图片发送，不发送 MP4 或音轨。流程与 prompt observer 相同（inventory → 逐资产属性 → 复核 → 中立化 → 描述分离）。镜头边界取相邻抽样帧的中点，超过 4 个镜头时合并相邻片段。产出同一种 `observation.json`，`frames` 是真实的抽样时间。

## 3. 属性目录

- 唯一来源：`scripts/dialogue_generation/attribute_catalog.json`（`schema_version: 2`）。字段数：person 19、object 13、indoor_environment 21、outdoor_environment 20。每个字段有 `label`、可选值，部分字段有 `ordered_values`（相对修改用）；每个类别有 `summary_fields` 和相容性约束。
- `unknown` / `not applicable` 是保留值，不参与变化采样。
- 修改目录后运行 `python scripts/dialogue_generation/observation_contract.py`，重新导出 `observation.schema.json` 和 `observation_prompt.md`（一个可以直接交给外部 VLM 的完整观察 prompt，包含全部字段、取值和示例）。
- `attribute_schema_version` 不匹配的 observation 会被 planner 直接拒绝。
