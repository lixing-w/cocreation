# scripts/dialogue_generation

视觉共创 agent 的数据构造管线。高层设计见仓库根目录的 [README](../../README.md)，生产运行见 [docs/native_batch.md](../../docs/native_batch.md)。本文件是模块地图，并列出调试单个 case 时可以单独运行的各阶段入口。

## 模块地图

| 阶段 | 文件 | 说明 | 文档 |
|---|---|---|---|
| 调度 | `native_batch.py` | 单卡批处理：VLM 批次 ‖ 媒体波次，SGLang 和评审线程池 | [native_batch](../../docs/native_batch.md)、[materialization](../../docs/materialization.md) |
| | `parallel_vlm.py` | 每个 case 的文本阶段子进程链（observer → generate [→ dry-run]） | |
| | `media_queue.py` | `GenerationQueue`（波次和推迟），以及 `AssetEvent` / `ReviewRequest` / `ReviewResult` | |
| | `sglang_media.py` | H3 / H3-Image SGLang 服务管理、lane 切换、Turbo LoRA 热切换 | [media_backends](../../docs/media_backends.md) |
| 规划 | `planner.py` | seeded、终点锚定的对话状态机，以及 `validate_plan` | [planner](../../docs/planner.md) |
| | `cross_turn_rule.py` | 跨轮规则层，以及 `validate_rule_timeline` | |
| | `asset_variations.py`、`attribute_catalog.json` | 属性目录、变化采样、语义归一 | [observation](../../docs/observation.md) |
| | `references.py` | 正向因果引用（`reference_materials`），以及校验 | [data_contracts](../../docs/data_contracts.md) |
| | `conflicts.py`、`generation_requirements.py`、`output_geometry.py`、`planner_rules.py` | 冲突需求模板、构图/姿态要求、画幅与尺寸、媒体当前规则 | |
| 台词 | `generate.py` | 规划 → 逐轮台词（writer + verifier）→ prompt 改写，支持续跑 | [dialogue_realization](../../docs/dialogue_realization.md) |
| | `stats.py`、`catalog.py` | 覆盖统计；`placeholders.json` 投影 | |
| Prompt | `generation.py` | 生成合约（`build_tasks`、`construction_order`） | [generation_prompts](../../docs/generation_prompts.md) |
| | `rewrite_prompts.py` | prompt writer、候选联合撰写、各类 reasoner、开场白 | |
| | `formal_sequence.py` | `generation_context` / `review_context`、正式导出、失败剪枝 | |
| 物化 | `materialize.py` | 物化会话协程：生成、审查、修复、亮度对齐、缓存 | [materialization](../../docs/materialization.md) |
| | `reviewed_turn.py` | 一轮多产物的顺序累积评审 | |
| | `generation_retry.py` | 尝试版本命名、修复重试任务 | |
| 工具 | `observation_contract.py` | 从属性目录导出 `observation.schema.json` 和 `observation_prompt.md` | |
| | `run_stats.py` | run 级别的耗时和 Claude 回执统计 | |
| | `repair_sequence_layout.py` | 不调用模型，重建正式导出 | |
| | `batch.py` | 早期的"多个 observation 共用一个 vLLM 跑对话"脚本（遗留） | |

上游和下游工具位于 `tools/`：`prompt_observer.py`、`video_observer.py`、`dialogue_realizer.py`、`dialogue_verifier.py`、`placeholder_prompter.py`、`candidate_prompter.py`、`prompt_reasoner.py`、`media_review.py`、`turn_media_review.py`、`prompt_fixer.py`、`video_editing.py`、`find_crop_in_video.py`。

`observation_prompt.md` 是 `observation_contract.py` 自动导出的产物（交给外部 VLM 的完整观察 prompt），不是说明文档，请不要手工编辑。

## 单独运行各阶段（调试）

以下命令都在仓库根目录运行。带模型的步骤需要在 GPU 节点上，或在有 `claude` 可用的环境中运行。

```bash
# 0) 纯 Python 规划（本地可运行）
python3 -c "import json; from scripts.dialogue_generation.planner import build_plan; \
p = build_plan(json.load(open('scripts/dialogue_generation/examples/tokyo_vlog.json')), seed=1, reworks=1); \
print(len(p['turns']), [t['intent']['action'] for t in p['turns']][:10])"

# 1) 文本 → observation
python -m tools.prompt_observer INPUT.json --source-format h3 --fps 24 --output-dir RUN/case/observation \
  --backend sonnet --think-mode medium --max-tokens 8192

# 2) observation → 对话 + 生成 prompt
python scripts/dialogue_generation/generate.py RUN/case/observation/observation.json \
  --output-dir RUN/case/dialogue --seed 17 --rewrite-seed 20261135 --reworks 1 \
  --backend sonnet --think-mode medium --max-tokens 4096 --context-tokens 65536 --timeout 600
#   --resume 续跑；--skip-generation-prompts 只写台词；--no-cross-turn-rule / --no-extra-behaviors 关闭可选行为
#   --modules direct candidates ... 限定准备模块；--user-persona casual|cute|precise|verbose|concise

# 2b) 单独补跑或强制重写生成 prompt
python scripts/dialogue_generation/rewrite_prompts.py RUN/case/dialogue --backend sonnet --think-mode medium [--force]

# 3) 物化计划检查（本地可运行，需要 ffmpeg）
python scripts/dialogue_generation/materialize.py RUN/case/dialogue --dry-run --native-video-output \
  --video-width 1024 --video-height 576

# 4) 重建正式导出 / 刷新 placeholders.json（不调用模型）
python scripts/dialogue_generation/repair_sequence_layout.py RUN/case/dialogue
python scripts/dialogue_generation/catalog.py RUN/case/dialogue
```

示例 observation：`examples/tokyo_vlog.json`（5 个资产、3 个镜头）和 `examples/alpine_kayak_vlog.json`（6 个资产、4 个镜头），各自带一条初始规则。
