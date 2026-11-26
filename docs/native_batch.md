# Native batch 运行手册

生产数据由 `run_native_dialogue_batch.sh` → `scripts/dialogue_generation/native_batch.py` 产生。每个 Slurm 作业占**一张 B200**，运行一个 Python 进程，处理一份清单里的全部 case。作业之间不通信；扩大规模的方式是切分清单，同时提交多个作业。

## 1. 输入清单

`manifest.json` 与它引用的输入文件放在同一目录（或子目录）下：

```json
{
  "version": 1,
  "cases": [
    {
      "id": "h3_tamsui_selfie",
      "source_format": "h3",
      "input_file": "h3_tamsui_selfie.json",
      "input_sha256": "7045e4c2…",
      "seed": 17,
      "reworks": 1,
      "video_width": 1024,
      "video_height": 576
    }
  ]
}
```

| 字段 | 含义 |
|---|---|
| `id` | case 目录名，必须是单层合法目录名且全局唯一 |
| `source_format` | `h3`（H3 结构化 prompt，按 24 fps 换算帧数）或 `openvid`（caption） |
| `input_file` | 相对清单目录的路径，不能逃出清单目录。内容是一条 JSON 记录（H3 用 `prompt` 字段，OpenVid 用 `caption`，需带时长元数据），见 [observation.md](observation.md) |
| `input_sha256` | 输入文件哈希，启动和续跑时都会校验 |
| `seed` | 规划 seed：决定模块、资产、规则、人设等全部结构随机性 |
| `reworks` | 预留返工数（0–2）。只有在 `--reworks-policy manifest` 时生效，当前生产命令就是这样用的；默认策略下会被忽略 |
| `video_width/height` | 视频尺寸（native 输出时作为原生网格的长短边） |

清单中的其他字段（标题、数据来源、revision 等溯源信息）会原样保存在 `inputs.json` 中，不参与执行。

输入文本（以及使用 video observer 时的源视频）只是**语义和创意的引导**：它决定 observation 里有哪些资产、属性和镜头。最终 sequence 里生成的视频**不需要**在画面上与源输入一致，只要和对话本身自洽即可。

`--reworks-policy` 决定每个 case 的预留返工数：

- `random_0_1`（代码默认）：用 `real-prompt-reworks-v1:{seed}:{id}` 作种子，可复现地抽取 0 或 1；
- `manifest`（**当前生产使用**）：直接使用清单里的 `reworks`，返工数的分布由构造清单时决定；
- `0` / `1` / `2`：统一覆盖。

实际取值写入 `cases.tsv`（7 列：id、source_format、input_file、seed、reworks、video_width、video_height）和 `rework_policy.json`。续跑时始终沿用 `cases.tsv` 中的取值。

## 2. 提交

当前生产命令：

```bash
cd /users/lwang223/cocreation
export COCREATION_SCRATCH=$(realpath scratch)
export COCREATION_SGLANG_RUNTIME="$COCREATION_SCRATCH/runtime/sglang-h3-benchmark"
export H3_VIDEO_TRANSFORMER="$COCREATION_SGLANG_RUNTIME/weights/minimax_h3_ref2va_nvfp4_mixed_sglang.safetensors"
export H3_TEXT_VIDEO_TRANSFORMER="$COCREATION_SGLANG_RUNTIME/weights/minimax_h3_fl2va_nvfp4_mixed.safetensors"

sbatch --parsable --export=ALL --job-name=dialogue-nvfp4-mt-48-c12 \
    run_native_dialogue_batch.sh \
    data/inputs/dialogue-native-48-reseed-20261125/manifest.json \
    --backend sonnet --think-mode medium --concurrency 12 \
    --resident-media-models --h3-vae mt \
    --no-match-video-brightness --reworks-policy manifest
```

- **环境变量**：`H3_VIDEO_TRANSFORMER`（ref2va，有参考的视频，包括视频编辑和 Turbo LoRA）和 `H3_TEXT_VIDEO_TRANSFORMER`（fl2va，纯文生视频）让 `scripts/serve_h3_sglang.sh` 加载 **NVFP4 mixed** 量化的视频 DiT。不设置时，脚本回落到 `*_pruned_int8_convrot`。图片服务没有设置 `H3_IMAGE_TRANSFORMER`，仍使用默认的 H3-Image `int8_convrot`。`sglang_media.video_profile()` 会从文件名识别出 `nvfp4_mixed`，并把它记录到每次生成的 `effective_parameters` 里。`--export=ALL` 保证这些变量进入作业（脚本头部也写了 `#SBATCH --export=ALL`）。
- **`--parsable`** 只输出作业号，方便写进脚本或记录；**`--job-name`** 覆盖脚本里的默认作业名，日志文件 `slurm_logs/%x-%j.{out,err}` 会随之改名。
- **输出目录**：只给了清单（第一个位置参数），没有第二个位置参数，所以 RUN_DIR 是 `data/runs/dialogue-native-$SLURM_JOB_ID`。也可以把 RUN_DIR 作为第二个位置参数显式指定。
- 脚本会加载 `miniforge3/25.3.0-3`、`cuda/12.9.0-cinr` 和 conda 环境 `cocreation`，并设置 `COCREATION_REPO`、`COCREATION_SCRATCH`（已经 export 的值优先）以及单线程 BLAS。正式作业总是运行主仓库的代码。

**选项必须全部显式写出。** 只要传了位置参数，脚本就会清空写死在脚本里的 `python_options`，回落到 `native_batch.py` 的默认值：

| 选项 | `native_batch.py` 默认 | 生产命令 |
|---|---|---|
| `--backend` | `vllm` | `sonnet` |
| `--think-mode` | `low` | `medium` |
| `--concurrency` | 24 | 12 |
| `--resident-media-models` | 关 | 开 |
| `--h3-vae` | `native` | `mt` |
| `--match-video-brightness` | 开 | **关**（`--no-match-video-brightness`） |
| `--reworks-policy` | `random_0_1` | `manifest` |

不带任何参数直接 `sbatch run_native_dialogue_batch.sh` 时，会使用脚本里写死的旧清单（`dialogue-native-48-20261124`）和旧选项（会开启亮度匹配、使用随机返工策略，也不会设置 NVFP4 权重的环境变量），不是当前的生产配置。

`--h3-vae mt` 使用重新训练过 decoder 的 H3-MT VAE，解决多轮视频编辑的细节丢失和亮度下降，所以生产命令关闭了亮度匹配（见 [video_brightness_matching.md](video_brightness_matching.md)）。媒体后端的选型原因见 [media_backends.md](media_backends.md)。

其他选项：`--vlm-context-tokens 65536`、`--vlm-max-num-seqs 16`、`--vlm-batched-tokens 16384`、`--vlm-kv-cache-gib 16`、`--vlm-max-pixels 1048576`、`--vlm-video-fps 1.0`（只允许默认值）。这些选项只在 `--backend vllm` 时影响本地 vLLM 服务。

`--backend` 的两种模式：

- `sonnet`：文本阶段（observer、writer、verifier、prompt writer、reasoner）和媒体评审都通过 `cocreation.ClaudeSonnet` 调用 Claude Code CLI（`claude --print …`）。作业的 `PATH` 里必须能找到 `claude`。评审调用的回执保存在 `sonnet_review/`。
- `vllm`：在同一张卡上启动 Qwen3.6-35B-A3B vLLM 服务（`scripts/qwen/serve.py`），文本阶段和评审共用它，服务和 SGLang 媒体服务共享 GPU 显存。

启动时会检查：`nvidia-smi` 恰好显示一张 B200；`vlm_execution.json` 中记录的 backend/think mode 与本次一致。

## 3. 作业内部的执行流程

```text
cases ──切成大小为 concurrency 的批──▶ batch 1, batch 2, …

后台 asyncio 线程（VLM）:  [batch 1 文本阶段] [batch 2 文本阶段] [batch 3 …]
主线程（GPU 媒体）:                         [wave 1: batch 1 媒体] [wave 2: batch 2 媒体 + batch 1 被拒重试] …
评审线程池（4 线程）:                        异步 VLM 审查，结果交回主线程处理
```

1. **VLM 阶段**（`parallel_vlm.run_case`，每个 case 一个子进程链，批内用 semaphore 限制并发）：
   - `tools/prompt_observer.py` → `<case>/observation/observation.json`（开启 thinking，8192 tokens）；
   - `scripts/dialogue_generation/generate.py` → `<case>/dialogue/`（规划、逐轮台词、生成 prompt 与思考链；`--rewrite-seed = 20261118 + seed`）。
   - native batch 设置了 `validate_plan=False`，所以跳过 `materialize.py --dry-run` 预检。
   - 单个 case 失败会被隔离（`isolate_failures=True`），记为 `failed / stage=vlm`，不影响同批其他 case。
2. **入队**：一批 VLM 完成后，为其中每个 case 打开一个 materialization 会话（`materialize.materialization_session`，共享同一个 `SGLangMedia` 生成器和评审模型），取得该 case 的媒体单元 DAG，放入 `GenerationQueue.buffer`。
3. **波次**：当前一波的主队列和在途评审都清空后，`promote()` 把 buffer 里的全部单元按 `model_order`（尽量少切换模型 lane，起点优先选当前已加载的模型）排进主队列，同时开始预取下一批的 VLM。
4. **调度**：主线程每次取一个依赖已满足的单元推进。遇到 `ReviewRequest` 就交给线程池，评审结果由主线程送回会话。被拒绝的单元连同它在主队列中的下游被推迟到下一波；如果它原本走 `video-base` 或 `video-text` lane，重试时改到 `video-turbo` lane，因为重试是对被拒结果的视频编辑。
5. **收尾**：会话结束后得到 `complete`、`partial`（有媒体在 3 次后仍被拒，相关交互已剪枝）或 `failed`。

协议细节见 [materialization.md](materialization.md)。

## 4. Run 目录

```text
RUN_DIR/
  inputs.json              原始清单（续跑时必须相同）
  cases.tsv                实际执行的 case 表（含抽取的 reworks）
  rework_policy.json       返工抽取策略与每个 case 的结果
  gpu.csv                  nvidia-smi 信息
  vlm_execution.json       {"backend", "think_mode"}，续跑校验
  pipeline_state.json      当前阶段（最新一条事件）
  pipeline_events.jsonl    全部阶段事件（时间、节点、作业号、批次、波次……）
  results.json             每个 case 的状态
  media_queue.json         第 1 波主队列快照
  vlm_summary.json | vlm-batches/NNN.json   每批 VLM 汇总（失败列表、耗时）
  sonnet_review/           Sonnet 评审回执（sonnet backend）
  vllm.log, vllm_server.json               （vllm backend）
  sglang/<job>-<n>-<lane>/server.log       每次启动 SGLang 服务的日志
  <case_id>/
    vlm_state.json         该 case 文本阶段的进度与分阶段耗时
    observer.log, dialogue.log
    observation/           observer 全部中间产物与 observation.json
    dialogue/              对话、生成计划、媒体、审查、正式导出（见 data_contracts.md）
    media_timings.json     该 case 媒体阶段耗时（image / video_generation / video_edit / review / media_active）
    materialize_error.log  媒体阶段异常栈（若有）
```

`results.json` 中的状态：`queued_vlm` → `queued_media` ⇄ `materializing` → `complete` | `partial` | `failed`。失败时 `stage` 为 `vlm`、`media_plan` 或 `media`。另外还有 `units`、`completed_units`、`media_count` 和 `pruning`（被剪掉的媒体和轮次）等字段。

`pipeline_state.json` 的 `phase`：`initializing`、`starting_vllm`、`vlm`、`media_buffered`、`media`、`media_wave_drained`，终态为 `complete`、`complete_with_partial`、`complete_with_failures`、`failed`、`interrupted`（SIGTERM / 超时会转为 `SystemExit`，被记录为 interrupted）。

进程退出码：全部 complete/partial 时为 0。有失败时，sonnet backend 返回 1；vllm backend 只要至少有一个 complete 就返回 0。

## 5. 续跑

设置相同的环境变量，用**同一份清单**，**把原 RUN_DIR 作为第二个位置参数**，并带上与首次完全相同的选项重新提交：

```bash
# 先 export 与首次相同的 COCREATION_SCRATCH / COCREATION_SGLANG_RUNTIME / H3_VIDEO_TRANSFORMER / H3_TEXT_VIDEO_TRANSFORMER
sbatch --parsable --export=ALL --job-name=dialogue-nvfp4-mt-48-c12 \
    run_native_dialogue_batch.sh \
    data/inputs/dialogue-native-48-reseed-20261125/manifest.json \
    /users/lwang223/cocreation/data/runs/dialogue-native-<原作业号> \
    --backend sonnet --think-mode medium --concurrency 12 \
    --resident-media-models --h3-vae mt \
    --no-match-video-brightness --reworks-policy manifest
```

不传第二个位置参数时，新作业会以新的作业号建一个新目录，从头开始，而不是续跑。backend 和 think mode 不同会直接报错。视频 transformer 路径、精度、VAE 和亮度匹配设置都参与媒体任务的 fingerprint，这些设置不同不会报错，但已完成的媒体会被判定为失效并重新生成。

复用规则：

- `observation/observation.json` 已存在就跳过 observer；不完整的 observation 目录会被改名为 `observation.incomplete-<ns>`。
- `dialogue/dialogue.json` 已存在时，`generate.py --resume` 会校验 plan、人设、采样参数和改写 seed 都一致，然后从已完成的台词前缀继续。prompt 改写阶段按任务 fingerprint 跳过已完成的项。
- `dialogue/media_manifest.json` 已存在时，媒体会话以 resume 模式打开：fingerprint 和文件哈希都匹配的媒体直接复用，已归档的尝试和审查也会复用，不会重新生成。
- 改了规划代码之后，旧 run 的 plan 可能对不上（resume 会报错）。这种情况应使用新的 RUN_DIR。

## 6. 监控与统计

```bash
cat RUN_DIR/pipeline_state.json
python3 -c "import json,collections;r=json.load(open('RUN_DIR/results.json'));print(collections.Counter(v['status'] for v in r.values()))"
tail -f slurm_logs/<job-name>-<jobid>.out      # [vlm] / [generation][T<wave>][case:pos/units] / [sglang] 行
python3 scripts/dialogue_generation/run_stats.py RUN_DIR [--include-calls]   # 汇总耗时与 Claude 回执 token
```

每个 case 的 `dialogue/dialogue.json` 中的 `stats` 字段记录覆盖统计：行为覆盖、模块分布、媒体按类型/操作统计、引用深度、规则决策、回退次数、writer 尝试次数和 token 等（`scripts/dialogue_generation/stats.py`）。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `Expected exactly one visible B200 GPU` | 检查 `--gres=gpu:1 --constraint=b200` |
| `Claude Code is missing from the Slurm job PATH` | sonnet backend 需要作业环境中能找到 `claude` |
| `Resume requires the original input manifest` | 清单内容和 `inputs.json` 不同。不要修改已运行的清单，换新的 RUN_DIR |
| `Resume requires the original VLM backend and think mode` | 续跑时补上首次的 `--backend/--think-mode` |
| `Input checksum changed` | 输入文件被改动，`input_sha256` 对不上 |
| 某 case `failed / stage=vlm` | 看 `<case>/observer.log`、`dialogue.log`、`vlm_state.json`。常见原因是 writer 连续 5 次被 verifier 拒绝，或 prompt 校验 3 次失败 |
| 某 case `partial` | 正常结果：有媒体 3 次评审仍失败，相关交互已剪枝。`results.json` 和 `media_manifest.json` 的 `pruning` 字段列出了被剪的内容 |
| `[sglang] reload failed` | 看对应的 `sglang/*/server.log`。SGLang 运行时位于 `$COCREATION_SCRATCH/runtime/sglang-h3-benchmark`，见 [media_backends.md](media_backends.md) |
| 想确认视频是否真的用了 NVFP4 mixed 权重 | 查看任一视频尝试记录中的 `effective_parameters.precision`（应为 `nvfp4_mixed`）和 `transformer` 路径（在 `media_manifest.json` 或 `generation_records/` 里）。如果是 `int8`，说明提交时没有 export `H3_VIDEO_TRANSFORMER` / `H3_TEXT_VIDEO_TRANSFORMER`，或者没有带 `--export=ALL` |
