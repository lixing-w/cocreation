# 媒体物化、审查与正式导出

代码：`scripts/dialogue_generation/materialize.py`、`media_queue.py`、`reviewed_turn.py`、`sglang_media.py`、`native_batch.py`、`generation_retry.py`、`formal_sequence.py`、`output_geometry.py`、`tools/media_review.py`、`tools/turn_media_review.py`、`tools/prompt_fixer.py`、`tools/video_editing.py`、`tools/find_crop_in_video.py`、`cocreation/video_tone.py`。

## 1. 物化会话：一个协程

`materialization_session(run_dir, **options)` 是一个 **generator 协程**。它把"一个 case 要生成什么、怎么生成"封装起来，而把"什么时候生成"交给外部调度器。这样单 case 的 `materialize()` 和多 case 的 `native_batch.run_media()` 可以共用同一套逻辑。

协议：

```python
session = materialization_session(dialogue_dir, resume=..., shared_generator=gen, review_model=m,
                                  verify_media=True, shared_vlm=True, defer_retries=True, async_reviews=True,
                                  native_video_output=True, video_width=w, video_height=h, seed=s)
units = next(session)                # {leader_token: {members, dependencies, model, plan_position}}
event = session.send(token)          # 推进一个已就绪单元
#   event 可能是：
#   - None / AssetEvent(status='accepted')  该单元完成
#   - AssetEvent(status='rejected')         评审拒绝，修复重试已排好，单元回到待办（defer_retries）
#   - ReviewRequest                         需要一次评审：在线程池里调用 .run()，再 session.send(ReviewResult)
#   - AssetEvent(status='exhausted', discarded=…)  3 次仍失败，相关媒体已剪枝
# StopIteration.value == (manifest, last_event)
```

会话打开时，先由 `load_contract()` 读取 `generation_plan.json` / `placeholders.json` 和 `dialogue.json`，并完成全部校验：亮度锚点必须唯一；尺寸合法；SGLang 模式下只允许 H3 系列 backend；H3 recipe 合法。dry-run 在这一步就返回计划摘要。

**单元**（`materialization_units` / `queued_units`）：

- 每个任务是一个节点，依赖等于 `dependencies`。
- 一轮多产物（≥2 个生成产物、不含 clip_edits）构成一个**交付组**。组内按依赖排序后串成链，下游只能在整组都通过后，才能使用组内的产物。
- `model` 字段是模型 lane 键：`(backend, model_id[, turbo][, 't2va'])`。确定性操作（复制、拼接）没有模型键，可以随时执行。

**调度**（`model_order`）：从每个可能的起始模型出发做一次贪心拓扑排序：先清空所有就绪的 CPU 单元，然后尽量连续执行当前模型的就绪单元，直到没有为止再切换。比较各起点的切换次数（包括与当前已加载模型的切换），取最少的那个。

## 2. 单次生成（`generate_media`）

1. **准备输入**（`tools.video_editing.prepare_inputs`）：按槽位顺序取依赖媒体，构造 `Reference`。视频→图片依赖调用 `crop_generated_reference()`：先用 `tools/find_crop_in_video.py` 让 VLM 在生成的视频里找到主体的最佳帧和 bbox，再裁剪（结果按视频哈希和 query 缓存在 `_inputs/generated_crops/`）。H3 的视频参考如果短于 2 秒会被补长。
2. **尺寸**（`output_geometry.geometry`）：图片使用 H3-Image 推荐画布（`cocreation.h3_image.size_for`：1:1 为 4096², 16:9 为 5472×3072, 9:16 为 3072×5472），要求边长是 32 的倍数、面积 4–20 MP。native 视频输出时，视频使用模型原生网格，按画幅决定横竖。
3. **调用生成器**：SGLang 模式调用 `SGLangMedia.generate(backend, prompt, references, recipe, dir, options)`；否则使用 `cocreation` 的 ComfyUI 封装（每次切换模型键都会重启 worker）。
   - 图片：H3-Image，6 步，guidance 1.0。
   - 新视频：H3 base，50 步，int8；有参考时为 `ref2va`，无参考时为 `t2va`（独立的 text-video 服务）。
   - 视频编辑（`video_edit`，包括所有修复重试）：同一个视频服务挂载 Turbo LoRA（`lightx2v-ref2va-v1.1`，8 步）。
   - 视频时长：`max(4, 输出时长, 各参考视频时长)`，上限 15 秒。
4. **后处理**：校验返回的尺寸；native 模式保留原生几何、音轨和帧网格时长；开启 `match_video_brightness` 时，对 `video_edit` 结果做亮度对齐（[video_brightness_matching.md](video_brightness_matching.md)；使用 H3-MT VAE 后已不再需要，生产命令传 `--no-match-video-brightness` 关闭，但代码默认仍开启）。
5. **记录**：在 record 中写入 `actual_prompt`、`references`、`options`、`effective_parameters`（SGLang 返回的实际参数，如 steps / precision / shift / fps / frames）、`inference_recipe`、`model_id`、`generation_seconds`、`model_reload_seconds`；原始回执写到 `generation_records/`。

## 3. 审查与修复循环（`generate_reviewed`，单产物）

```text
attempt 1: 用最终 prompt 生成 ─▶ VLM 评审 ─┬─ accepted ─▶ 完成
                                          └─ rejected ─▶ prompt fixer 写编辑指令
attempt 2: 以 attempt 1 的结果为编辑底图（原参考 + 被拒结果作为最后一个输入）─▶ 评审 ─▶ …
attempt 3: 同上；仍被拒 ─▶ AssetRejected ─▶ 剪枝
```

- **评审**（`tools/media_review.MediaReviewer.verify`）：输入是 `review_context`（真实对话、当前交换、当前用户请求、`latest_rules`、画幅来源、`requested_edits`、参考绑定、返工工作流阶段）、每个参考的真实图片或视频（带标签说明）以及生成结果。视频以原生视频块输入，Sonnet 后端则按帧输入。评审输出 JSON：`think{request_check, quality_check, decision}`，5 个分数（`request_fidelity`、`edit_preservation`、`reference_alignment`、`visual_integrity`、`cross_turn_rules`），逐条规则的 `rule_checks`，以及 `decision / reason / reply`。
- **判定标准**：0 表示明确的、可见的、实质性的失败；1 表示轻微偏差或无法确认；2 表示满足。**所有项 ≥1 即接受。** 系统 prompt 明确列出不该拒绝的情况：年龄观感、笑容、光线强弱、细微动作、季节在画面里无从判断、物体的小部件颜色不同等。视频必须看开头、中段和结尾再下结论。格式错误时，对同一媒体做最多 2 次技术性重试，不算作一次生成。
- **修复**（`tools/prompt_fixer.PromptFixer`）：看到被拒结果、用户请求、规则和评审理由后，输出 `{"edit_prompt": …}`，只要求修改可见缺陷，保留其他一切。`generation_retry.retry_edit_task` 把任务转换为 `local_edit` / `video_edit`：保留原参考槽位和标签，把最近一次被拒结果追加为最后一个输入（`role='edit_source'`）。
- 被接受的尝试的 token 记为 `accepted_placeholder`（如 `[[image_0003_01]]`）。之后才把文件发布到 `media/`。
- `defer_retries=True` 时，被拒绝后会话立即 yield `rejected` 事件，把 GPU 让给别的单元；修复后的重试在下一波执行。

### 一轮多产物（`reviewed_turn.iter_reviewed_turn`）

同一轮里的多个独立产物（例如两个候选）**按顺序生成、累积评审**（策略 `sequential-delivery-v1`）：先生成第 1 个并评审；通过后生成第 2 个，评审时把已通过的第 1 个一起提供给评审模型作为上下文，但只给当前产物打分（`tools/turn_media_review.verify`）……被拒的产物在同一位置走修复和重试，最多 3 次。全部轮次快照记录在 leader 的 `turn_reviews` 中。

### 资产图 → 镜头的返工工作流

`deliver_reworked_clips` 轮次里的资产图和镜头**不按候选处理**，而是作为有依赖的阶段：先生成并审查资产图，通过后再用它编辑镜头。每个阶段的评审都带 `workflow_stage`：资产图只看外观修改，镜头看动作和修改。

## 4. 剪枝（`dialogue_without_failed_media`）

一个媒体 3 次仍被拒绝时，不丢弃整个 case：

1. 把对话切成"交换"：以一个 assistant 交付轮、或非澄清类的 assistant 轮作为一个交换的结尾；
2. 把失败的媒体标记为 omitted，并迭代传播：依赖它的任务；产出或引用了 omitted 媒体的交换；依赖被删交换中创建的规则的交换；规则结束点落在被删轮次里的交换……
3. 去掉这些轮次和占位符；规则在被删轮次里的结束会被撤销。

被剪的媒体在 manifest 中记为 `failed`（耗尽）或 `omitted`。case 结束状态为 `partial`（还有完成的媒体时）或 `failed`；`pruning` 字段记录全部细节。正式导出基于剪枝后的对话。

## 5. 多 case 调度（`native_batch.run_media`）

- 一个 `GenerationQueue` 管理所有 case 的单元，键为 `(case_id, token)`；lane 映射为 `image / video-base / video-text / video-turbo`（`sglang_media.media_lane`）。
- **波次**：`promote()` 把 buffer 中的单元按 `model_order` 排进主队列（只考虑 buffer 内部的依赖），`round += 1`。主队列清空且没有在途评审时，这一波结束（`media_wave_drained`）。
- `pop_ready(in_flight)`：取主队列中第一个依赖都不在"主队列、在途评审、buffer"里的单元。
- `defer(key)`：被拒单元进入 buffer，主队列里依赖它的后代也随之移入 buffer（主队列本身是拓扑序，一次扫描即可）。被拒的 `video-base` / `video-text` 单元改到 `video-turbo` lane。
- `discard(...)` / `discard_case(...)`：剪枝或 case 失败时移除单元，并取消该 case 的在途评审。
- 评审在 4 线程的 `ThreadPoolExecutor` 中运行（`ReviewRequest.run`）。评审参数是快照，**只有主线程推进会话、写 manifest 和 results**。
- VLM 阶段在后台 asyncio 线程中按批运行；每开启一波媒体，就预取下一批 VLM，让 Claude API 或 vLLM 上的文本工作和 GPU 生成重叠。

## 6. SGLang 媒体服务（`sglang_media.SGLangMedia`）

- 服务通过 `scripts/serve_h3_sglang.sh {image|video|text-video} [--h3-vae native|mt]` 启动，监听随机本地端口。启动日志在 `RUN_DIR/sglang/<job>-<n>-<lane>/server.log`。
- `select(lane)`：`video-turbo` 和 `video-base` 共用同一个视频服务，通过 `/v1/set_lora`（dynamic merge，strength 1.0）和 `/v1/unmerge_lora_weights` 切换 Turbo。非常驻模式下，切换到另一个服务前会先停掉当前服务；常驻模式下所有服务保持运行。
- 图片：`/v1/images/generations` 或 `/v1/images/edits`（多张参考图以 `image[]` 上传），6 步，返回 base64 PNG。
- 视频：`POST /v1/videos`（`task=ref2va|t2va`，`conditions` 中每个参考为 image / video / video_audio，`target={short_edge, aspect_ratio, duration_seconds}`，flow shift 12/3，`output_compression=64`，约等于 x264 CRF 18），然后轮询任务状态并下载内容。
- 请求超时或连接异常时，会停掉当前服务，下次使用时重新启动。

## 7. 正式导出（`formal_sequence.export_sequence`）

每次有媒体完成、每次评审之后，以及会话结束时，都会重新导出 `sequence.json` 和 `dialogue.md`。结构见 [data_contracts.md](data_contracts.md)。要点：

- 只发布真实发生的事情：被拒的尝试、评审思考和拒绝说明都保留在序列中。
- 一个交付轮还没有全部被接受时，导出在这一轮截止，状态记为 `incomplete`。
- 规划思考只出现在该轮第一次调用之前；之后的调用只带一句过渡；修复重试不再重复规划。

`scripts/dialogue_generation/repair_sequence_layout.py RUN/dialogue …` 可以在**不调用模型、不修改审计记录**的前提下，用当前的导出代码重建 `sequence.json` 和 `dialogue.md`（会先备份原文件，并核验调用、媒体、台词和评审都没有变化）。

## 8. 续跑与缓存

- 每个单元的 `request_fingerprint` 由以下内容计算：任务本身、设置（不含 h3 推理策略）、依赖媒体的 sha256、亮度锚点哈希、H3 recipe。fingerprint 和文件哈希都匹配时直接复用。
- 已归档的尝试（`generation_attempts/…/attempt.json`）在续跑时按顺序复用：已有评审的不会重审，已有修复 prompt 的不会重写。旧策略版本的评审会被重新评审，旧结论保留在 `critique_history`。
- `.materialize.lock` 防止两个进程同时物化同一个目录。成功结束后会删除 `_work/`。

## 9. 单 case 命令行（调试）

```bash
python scripts/dialogue_generation/materialize.py RUN/dialogue --dry-run      # 只看计划，本地可用
python scripts/dialogue_generation/materialize.py RUN/dialogue --resume \
  --native-video-output --video-width 1024 --video-height 576 --shared-vlm --review-base-url http://127.0.0.1:8000
```

单 case 命令行走 ComfyUI 封装（`cocreation.MiniMaxH3` / `MiniMaxH3Image`），不走 SGLang。默认的 H3 recipe 是 50 步、无 Turbo（`--video-turbo-lora` / `--new-video-turbo-lora` / `--edit-video-steps` / `--edit-video-sampler` 可以覆盖）。评审默认连接 `--review-base-url` 上的 vLLM，必要时会自行启动一个。生产数据请使用 native batch。
