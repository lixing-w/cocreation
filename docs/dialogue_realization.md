# 台词实现：Writer、Verifier 与后端

代码：`scripts/dialogue_generation/generate.py`、`tools/dialogue_realizer.py`、`tools/dialogue_verifier.py`、`tools/dialogue_intent.py`、`tools/dialogue_personas.py`、`tools/dialogue_models.py`、`tools/attribute_authority.py`。

## 1. `generate.py` 的流程

```bash
python scripts/dialogue_generation/generate.py OBSERVATION.json --output-dir DIR \
  --seed 17 --rewrite-seed 20261135 --reworks 1 \
  --backend sonnet --think-mode medium --max-tokens 4096 --context-tokens 65536 --timeout 600 [--resume]
```

1. 读取 observation，`choose_personas(seed)`，`build_plan()`。输出目录必须不存在（`--resume` 除外）。
2. 写出 `plan.json`、`rules.json`、`observation.json`、`placeholders.json`。
3. 逐轮调用 `QwenRealizer(turn, media, completed, seed=rewrite_seed + turn_id * 5)`，每完成一轮就写一次 `dialogue.json`（`realization.completed_turns` 记录进度）。
4. 全部完成后再次 `validate_plan`，标记 `realization.status=complete`，写出 `dialogue.md`。
5. 复用同一个 realizer 运行 `rewrite_prompts.rewrite_run()`，见 [generation_prompts.md](generation_prompts.md)。`--skip-generation-prompts` 跳过这一步，之后可以用 `rewrite_prompts.py RUN_DIR` 补跑。

续跑会校验 plan（必要时尝试旧版规则采样方式复现）、人设、采样参数（temperature / top_p / top_k）、改写 seed、backend/model/model_mode/think_mode，以及已完成的台词前缀是否完整。

`--seed` 只控制结构；`--rewrite-seed` 只控制措辞（省略时每次随机，并在续跑时恢复）。native batch 固定使用 `20261118 + seed`。

## 2. Writer

`QwenRealizer.__call__(turn, media, previous, seed)` 为一轮构造 prompt：

**System**（`SYSTEM`）：`ATTRIBUTE_AUTHORITY` 加上通用写作规则：意图决定含义；保留全部细节、选项绑定、镜头顺序和范围；区分图片和视频、新版本和旧版本、待做和已做；保留有意设计的歧义和冲突；用户让助手做事，助手交付、确认或提问；不要捏造设计、历史或偏好。

**User prompt** 包括：

- 人设语气和语气示例（只模仿语气，不模仿内容）。用户有 5 种人设，按 `persona:{seed}` 抽一次；助手的人设固定。
- 历史：最多最近 3 条消息。对**自包含**的行为（交付、确认、首次请求、请求镜头/拼接、采纳等）不提供历史，以免模仿上一轮的套话。
- `Current speech act`：`ACT_GUIDANCE[action]` 加上按情况追加的控制语。例如：交付轮只做简短交付，不追加提议；首次资产请求不能假装存在源图；新视频不能说成重做旧片；非规则轮次不重复已有规则；`keep` 规则决策仍然要继续生成原请求的媒体。
- `Reply intention`：产物清单（`requested_products`：类型和数量，从计划中的交付行计算，不从格式描述推断）、按 `;` 拆开的内容要点、附加要求（规则带来的设置和持续时间）、shot staging（镜头请求）、source design 与当前属性状态（`reference_state_lines`，当前属性覆盖源描述中冲突的形容词）、主体名、已就绪媒体的类型和数量。
- `Important turn requirement`：上一次 verifier 给出的修改意见（只保留最新一条）。

`_format_problems` 只检查客观格式：非空；不含 JSON、代码块或 `<think>`；全英文；不含媒体标记。

## 3. Verifier

`QwenVerifier` 用同一个模型、独立的 system prompt 和 JSON schema（`REPORT_SCHEMA`），按 4 个维度打 0/1/2 分：

| 维度 | 检查 |
|---|---|
| `content_fidelity` | 是否执行了本轮意图，必要细节（媒介、外观、选项和版本、构图、顺序、范围）是否齐全 |
| `grounding` | 提到的主体、版本、事件、结果和范围是否都有依据；首次创建不能捏造旧图；开场不能假装助手已经说过话 |
| `conversation_coherence` | 在当前位置是否说得通；版本选择与指代纠正不能混淆；有意的歧义和冲突在计划的澄清之前是合法的 |
| `naturalness` | 是否是可理解的口语。语气、长度、人设不计入 |

verifier 收到的是一份事实简报（`brief`）：`reference_requirements`（来自 `reference_policy`：`clear` / `ambiguous_subject` / `ambiguous_shot_number` / `per_edit`）、草稿要点、产物、附加要求、绑定的参考、已展示和已交付的媒体、同一说话人最近几轮的开头等。对 `resolve_rule_conflict` 且决策为 `keep` 的轮次，使用专门的 `KEEP_SYSTEM`。

只有 4 项全为 2 才通过。不通过时，`critique` 必须以 "You should" 开头，并作为下一次 writer 的要求。writer 每轮最多 5 次尝试（`MAX_REWRITE_ATTEMPTS`），每次 seed 递增；仍失败则抛出 `RuntimeError`，这个 case 记为 VLM 阶段失败。全部调用记录在 `<dialogue>/{qwen|sonnet}/calls.jsonl` 和 `verifier/` 中。

## 4. 后端（`tools/dialogue_models.py`）

| backend | 模型 | 说明 |
|---|---|---|
| `sonnet` | `claude-sonnet-5-5`（`cocreation.ClaudeSonnet`） | 通过 Claude Code CLI 调用：`claude --print --model … --effort … --input-format stream-json --output-format stream-json --tools '' --json-schema …`，无工具、无会话持久化。`--think-mode low\|medium` 作用于 observer、prompt writer、reasoner；writer 和 verifier 固定使用 `low`。图片转为 base64 块，视频按 fps 抽帧后逐帧发送 |
| `vllm` | `nvidia/Qwen3.6-35B-A3B-NVFP4`（`cocreation.VLLMQwen`） | Linux 默认。thinking 默认关闭，支持 `response_schema` |
| `local` | `Qwen/Qwen3.6-35B-A3B`（`cocreation.Qwen`） | 进程内 CUDA Qwen |
| `ollama` | `qwen3-vl:30b-a3b-instruct-q4_K_M` | Mac 默认（只适合离线调试；本地开发约定不运行模型） |

采样参数默认为 temperature 0.9、top_p 0.95、top_k 50。verifier 使用 temperature 0.1，prompt writer 使用 0.3。所有调用都有上下文预算检查，超出时直接报错，不会截断最近的消息。

## 5. 写台词以外的调用

`QwenRealizer` 同时为 prompt 改写阶段提供以下入口，它们共用同一个后端连接：

- `generation_request`：prompt writer；
- `generation_candidates`：一轮多个候选的联合 prompt；
- `generation_reasoning`、`operation_reasoning`、`delivery_reasoning`、`rework_reasoning`：各类思考链；
- `production_opening`：真实生成前助手说的一句开场白。

详见 [generation_prompts.md](generation_prompts.md)。
