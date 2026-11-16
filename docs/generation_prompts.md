# Dialogue placeholders → generation experts

## 描述从哪里来

`Planner.add_media()` 写入 `base_description`；资产描述来自 observation 的身份文字及 `AssetSpec` 的结构化属性，正式镜头来自 observation 的 shot description，试拍与剪辑由对应规划模块描述。`CrossTurnRules.decorate()` 在对话规划结束后计算各输出的 `effective_attributes`、`render_state`，再用 `describe_media()` 生成最终 `description`。这些都不是模型生成提示词。主对话模型只把意图改写成自然对白，不决定资产版本、镜头顺序或规则状态。

`assign_reference_materials()` 描述离线构建依赖，和对话出现顺序独立：目标视频为根；目标一致的资产从目标提取身份；其他资产沿实际版本依赖局部编辑。正式镜头和试拍使用隐藏 target shot 提供动作、构图和关系。对话双方不会提及这个目标视频；未来 reasoning trace 可解释为 assistant 的推理内容。

## 新的后处理

正常 `generate.py` 在所有主对话轮次完成后自动调用 `rewrite_run()`，逐个输出可供 expert 执行的合约与 prompt；完整合约在 dialogue / generation_plan，精简清单在 placeholders。真实 Qwen 流程复用驻留模型，rewriter 始终只接收文字；旧 `flux_prompter` / `h3_prompter` 也不再把媒体附给 Qwen。图片/视频只在执行阶段 `tools.video_editing.prepare_inputs()` 实际读取。

输出：

- `placeholders.json`：供人核对与脚本生成的精简清单，顶层 `prompt` 只保存一次；optional `references` 是实际 expert 槽位。
- `dialogue.json`：完整描述、状态、`generation` 合约及 `generation_prompt`，保留原有详细记录。
- `generation_plan.json`：`status`、按依赖排序的 `construction_order` 及全部任务。
- `prompt_rewriter/<placeholder>/input.txt` 与 attempt 日志：纯文字输入、模型原始输出、验证问题。

`prompt` 对源文件、直接提取、复制、拼接为 `null`；这些条目有可执行的确定性操作，不需要编造模型 prompt。生成任务路由为 FLUX.2 [dev] 或优先 VDN-H3（同一种 MiniMax-H3 prompt 格式，备用 MiniMaxH3）。每个 prompt 的 `<Image N>` / `<Picture N>` / `<Video N>` 已经对应清单 `references` 的真实槽位，不要再次按整个 dialogue 编号。

`generation_plan.tasks[placeholder].dependencies` 是执行依据；它可以在 target clip 之外增加绑定资产，不能用 turn 的 `input_media` 代替。清单顶层 `references` 记录抽帧秒数、shot ID、引用职责。所有输入在生成时才准备，frame 来源仍清楚标明为 target-video frame，而 FLUX 实际只收到图片。

失败保留已完成对话和逐 placeholder 检查点；重跑会跳过指纹一致且 ready 的条目。指纹包含引用状态及 rewriter system 的哈希；修改规则后自动重写。模型截断、缺源标签、格式错误、遗漏显式修改值会给出具体反馈，最多重试三次。只有 `generation_plan.status == complete` 才可消费。

## 精简 placeholders.json

每条记录只保留以下用途所需字段；不再重复完整 generation、属性字典、规则历史、哈希或参考媒体的长篇描述。

| 字段 | 用途 |
| --- | --- |
| `placeholder`, `kind`, `purpose`, `asset_ids` / `shot_ids` | 标识媒体及其资产、镜头；空 ID 列表省略 |
| `description` | 画面内容和视觉要求，去除反复出现的通用指令话术 |
| `reference_depth`, `reference_materials` | 原有深度和构建引用，值及顺序保持不变 |
| `operation`, optional `backend` / `model_id` | 确定性操作或 Flux / H3 路由 |
| `prompt` | 原生成提示词，字符串逐字保留，只存一份 |
| optional `references` | 按 expert 输入顺序排列的媒体槽位；保留 placeholder、kind、label、role，以及抽帧时间或 shot ID |
| optional `dependencies` | 仅当实际执行依赖不同于 reference_materials 时出现，否则直接使用 reference_materials |
| optional `seconds`, `output_seconds`, `shot_id`, `frame_seconds`, `preprocess`, `clips`, `source_video` | 对应任务真正需要的时长、提取和拼接参数 |
| optional `status` | 仅未就绪条目出现；未完成 prompt 的条目不可执行 |

`references` 是视觉输入槽位，`reference_materials` 是构建关系；二者不能互相替代。目标视频帧在 references 中仍指向原 video placeholder，kind 为 image，并带 frame_seconds。拼接 clip 顺序、shot ID、依赖边和 prompt 不会因为简化而改变。

`tools.video_editing.prepare_inputs()` 和 `execute_edit()` 同时支持精简清单条目与完整 generation 合约。只需排版、迁移既有输出时运行以下命令；它只改 placeholders.json，不请求模型、不重算引用，也不修改其他计划文件：

```bash
python -m scripts.dialogue_generation.catalog scratch/runs/tokyo-rule-manual-prompts-v2
```

## 图像编辑与三视图

角色/物体资产默认生成同一身份的 front / side / back 三视图设定图，统一比例和中性背景。环境保持环境参考图；observation 可用 `reference_layout` 覆盖默认布局。源帧只供身份依据，输出可重建成请求的初始设计状态。

局部修改比较实际 `reference_materials` 中的父版本与目标版本的完整有效属性，而不是沿对白的 edit_parent 猜测。只改颜色时只写颜色操作，并保留三视图布局、身份、材质、几何、光照和构图。若同时修改两项，必须写全两项；改变比例时不能要求保持该比例不变。整图黑白与单个物件改色是不同字段。

Tokyo 旧样例中黑白杯子还把 `aspect_ratio` 从 balanced 改为 wide and flat，所以它应有两项变化，不能误写为只去色。资产图的黑白处理不会自动污染后续要求 original_colors 的试拍/正式镜头。

## 试拍与正式镜头

用户请求试拍时不再默认要求 static / no character action。如果 observation 有覆盖这些资产的共同 shot，优先保留该 shot 的实际构图、动作、关系、时间线与未改变身份，只改当前版本或规则要求的属性；完全一致则直接提取。只重排顺序的镜头内容因此贴近目标视频。没有共同 shot 时才做多参考合成，并根据目标镜头文字选择合理动作关系。

H3 video editing 使用六段官方结构，summary 明确源视频，retention_analysis 区分修改与保留。对于已有源镜头，不凭文字杜撰手的左右、位置、镜头运动细节或背景角色。参考资产只传递指定设计，三视图不会作为视频里的三个人/物件出现。音频未知写 N/A；精确复用音轨由剪辑器实现。

现有 H3 接口限制 5–15 秒。较短 shot 的合约记录 `seconds >= 5` 与实际 `output_seconds`，动作按原速度完成后只在末尾补足，生成后需裁掉补足部分。超过 15 秒的生成编辑需先拆分，不能作为单个 H3 调用。纯拼接不受这个生成限制。

## 镜头交换 / 重排

纯顺序编辑用确定性代码，不重新采样 H3。`shot_ids` 决定顺序，`shot_versions` 决定每个位置的实际版本。只有明确要求改变镜头画面时才调用 H3。

`tools.video_editing.refine_shots()` 将 observation 的 start/end 作为粗切点，在每点两侧的局部窗口用 ffmpeg scene score 找实际切点，并映射回原视频的真实帧时间戳。窗口受相邻逻辑切点限制；返回 shot ID → 精确 frame interval 和候选/误差审计数据。找不到切点会明确报错，不能悄悄用粗时间切或增加逻辑镜头。检测能保证边界/编号策略，语义仍以 observation 为准；模糊转场可调整窗口与阈值并检查审计。

`extract_shot()` 按解码帧精确截取，并同步截取音轨；不使用会跳到关键帧的 stream copy。`concatenate()` 按给定片段顺序拼接音画；尺寸需一致，缺失音轨补静音。`execute_edit()` 执行合约中的确定性操作。

## 使用

```bash
# 正常主对话 + 自动 prompt 后处理
python scripts/dialogue_generation/generate.py observation.json \
  --output-dir scratch/runs/new-sequence --seed 68 --model qwen3-vl-30b-instruct

# 已完成轨迹补写 / 从失败条目继续
python -m scripts.dialogue_generation.rewrite_prompts scratch/runs/tokyo-rule-manual \
  --model qwen3-vl-30b-instruct

# 只合成对白，以后再补 prompt
python scripts/dialogue_generation/generate.py observation.json \
  --output-dir scratch/runs/dialogue-only --skip-generation-prompts
```

程序接口支持 `rewrite_run(..., writer=request)` 注入文字模型；request 接收 prompt、system、seed、max_tokens，返回 `(text, finish_reason)`。库调用注入纯对白 realizer 时，默认不额外启动模型；可显式提供 `prompt_rewriter` 或启用 `rewrite_prompts=True`。

```python
from tools.video_editing import prepare_inputs, execute_edit
from cocreation import Flux, VDNH3

# resolved 是已构建 placeholder -> 本地媒体路径的字典；按 construction_order 执行。
# task 可以直接取自 placeholders.json，也可使用 generation_plan 中的完整合约。
if not task.get('backend'):
    asset = execute_edit(task, resolved, observation, destination)
else:
    refs, prompt = prepare_inputs(task, resolved, observation, output_dir)
    if task['backend'] == 'flux':
        with Flux(model_id=task['model_id']) as expert:
            image = expert(prompt, refs)  # dev 默认 50 steps / guidance 4；klein 原默认保留
    else:
        with VDNH3(runtime=accepted_vdn_runtime) as expert:
            clip = expert(prompt, refs, seconds=task['seconds'])
# 对 video 输出按 output_seconds 去除补足部分；要求严格音轨复用时由确定性代码恢复。
```

本次验证为真实本地 Qwen prompt 重写、主对话重跑、专家 dry-run 及 ffmpeg 音画剪辑；不代表已在 NVIDIA 上实际生成 FLUX/H3 媒体。示例 target 为 fictional:// 路径，执行媒体任务时需要真实 target 文件。

## 参考依据

- [MiniMax 官方 H3 全参考模式指南](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)：引用职责、六段结构、视频编辑及 retention。
- [BFL 官方 FLUX.2 编辑文档](https://docs.bfl.ai/flux_2/flux2_image_editing)：指令编辑与多图引用。
- [FLUX.2 dev 官方模型卡](https://huggingface.co/black-forest-labs/FLUX.2-dev)：dev 的图像编辑接口。
- [BFL FLUX.2 提示词指南](https://docs.bfl.ai/guides/prompting_guide_flux2)：明确参考职责、正向描述与颜色控制。该页主要面向 pro/max；这里将通用图像编辑策略用于 dev，不把托管 API 参数当作本地能力。

三视图默认、隐藏 target-shot 继承和确定性重排是本项目的产品策略；官方指南没有为帧级镜头重排提供保证，未将 H3 的生成能力当成精确剪辑能力。
