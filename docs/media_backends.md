# 媒体后端与模型封装

管线里有两条生成路径：

| 路径 | 用途 | 入口 |
|---|---|---|
| **SGLang 服务**（生产） | native batch 的全部 H3 / H3-Image 生成 | `scripts/dialogue_generation/sglang_media.py` + `scripts/serve_h3_sglang.sh` |
| **ComfyUI 封装**（`cocreation` 包） | 单 case `materialize.py`、交互调试、示例脚本 | `cocreation.MiniMaxH3`、`cocreation.MiniMaxH3Image` 等 |

两条路径的 prompt 格式和参考标签完全一致（`<Picture N>`、`<Video N>`、H3 结构化章节），所以生成合约不需要关心实际走哪条路径。

## 0. 选型记录

| 决定 | 原因 |
|---|---|
| 默认后端用 SGLang | ComfyUI 路径速度慢；FreeVideo（VDN-H3，[vdn.md](vdn.md)）是 H3 的蒸馏模型，虽然快，但视觉质量很差。SGLang 跑原生 H3 和 H3-Image，可以常驻服务，并在同一服务内动态挂载 LoRA |
| 视频编辑用 H3 + Turbo LoRA，8 步 | 视频编辑（包括全部修复重试）占生成次数的大头，用少步数换取造数据的效率 |
| 其他视频生成用原生 H3，50 步 | 新镜头决定样本的基础视觉质量 |
| 视频 DiT 使用 **NVFP4 mixed**（4-bit）量化 | 经实际生成质量对比后选定。生产命令通过 `H3_VIDEO_TRANSFORMER` / `H3_TEXT_VIDEO_TRANSFORMER` 指定（见下表）。启动脚本本身的默认值仍是 `*_pruned_int8_convrot`，不设置这两个变量时会加载 int8 权重。`sglang_media.video_profile()` 会从文件名识别精度（nvfp4_mixed / nvfp4 / int8 / fp8 / bf16），并记录到生成参数中 |
| 视频和音频 VAE 用 H3-MT（`--h3-vae mt`） | 原生 VAE 在反复视频编辑时会逐轮丢失细节、亮度下降。MT VAE 重新训练了 decoder，专门解决这个问题。在此之前用[亮度对齐](video_brightness_matching.md)做部分弥补；生产命令使用 MT VAE，并以 `--no-match-video-brightness` 关闭了亮度对齐 |

---

## 1. SGLang（生产）

### 服务与 lane

| lane | `serve_h3_sglang.sh` kind | 权重 | 用途 |
|---|---|---|---|
| `image` | `image` | H3-Image `minimax_h3_image_pruned_int8_convrot` | 全部图片：文生、多图参考、编辑。6 步 |
| `video-base` | `video` | H3 ref2va：生产为 `minimax_h3_ref2va_nvfp4_mixed_sglang`（脚本默认 `ref2va_pruned_int8_convrot`） | 有参考的新视频，50 步，shift 12/3 |
| `video-turbo` | 同 `video` 服务（同一个 ref2va DiT）+ Turbo LoRA | `minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16`（nickname `lightx2v-ref2va-v1.1`） | 全部 `video_edit`，包括修复重试。8 步 |
| `video-text` | `text-video` | H3 fl2va：生产为 `minimax_h3_fl2va_nvfp4_mixed`（脚本默认 `fl2va_pruned_int8_convrot`） | 无参考的新视频（t2va） |

固定 recipe 见 `SGLangMedia.recipe()`：`video_edit` 使用 Turbo 8 步，其他视频使用 base 50 步。recipe 里写的是 `precision='int8'`，但 `video_recipe()` 会用 transformer 文件名识别出的精度覆盖它。`--h3-vae mt` 时视频和音频 VAE 换成 H3-MT 版本（`minimax_h3_{video,audio}_vae_mt_*`）。

### 运行时目录

`serve_h3_sglang.sh` 从以下位置读取（都可以用环境变量覆盖）：

```text
$COCREATION_SCRATCH/runtime/sglang-h3-benchmark/     ($COCREATION_SGLANG_RUNTIME)
  venv-cu130/bin/sglang                               SGLang（CUDA 13 用户态，cuda-compat）
  cuda-13.0/, cuda-compat-13-0/                       独立 CUDA 13 工具链（不影响其他环境）
  weights/minimax_h3_ref2va_nvfp4_mixed_sglang.safetensors     生产：export H3_VIDEO_TRANSFORMER 指向它
  weights/minimax_h3_fl2va_nvfp4_mixed.safetensors             生产：export H3_TEXT_VIDEO_TRANSFORMER 指向它
  weights/minimax_h3_ref2va_pruned_int8_convrot.safetensors    脚本默认（未设置 H3_VIDEO_TRANSFORMER 时）
  weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors     脚本默认（未设置 H3_TEXT_VIDEO_TRANSFORMER 时）
  weights/minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors ($H3_VIDEO_TURBO_LORA)
  native-models/MiniMax-H3, native-models/MiniMax-H3-Image     pipeline 配置
$COCREATION_SCRATCH/runtime/minimax-h3/models/         H3 文本编码器（qwen3vl_32b NVFP4-AWQ）与视频/音频 VAE
$COCREATION_SCRATCH/runtime/minimax-h3-image/models/   H3-Image 文本编码器与图片 VAE
$COCREATION_SCRATCH/cache/huggingface/hub/models--Comfy-Org--MiniMax-H3-Image/…  H3-Image transformer ($H3_IMAGE_TRANSFORMER)
```

仓库里**没有**安装 SGLang 运行时的脚本，它是在服务器上预先准备好的。`runtime/minimax-h3*/models` 由 ComfyUI 路径的安装脚本准备（`scripts/setup/setup_h3_image.py`，以及 `MiniMaxH3` 首次真实调用时的自动安装）。服务以 `HF_HUB_OFFLINE=1` 运行，不联网。

调试单个服务：

```bash
bash scripts/serve_h3_sglang.sh video --h3-vae mt --print-config     # 打印完整启动命令
bash scripts/serve_h3_sglang.sh image --port 30000                   # 前台启动（需在 GPU 节点）
```

HTTP 接口的用法见 [materialization.md §6](materialization.md)。

---

## 2. `cocreation` 包

```python
from cocreation import (Qwen, VLLMQwen, ClaudeSonnet, MiniMaxH3, H3Runtime,
                        MiniMaxH3Image, H3ImageRuntime, Flux, VDNH3, VDNRuntime, Reference)
```

导入包和实例化对象都**不会**导入 ML 库，也不会下载权重。

| 类 | 后端 | 输入 → 输出 |
|---|---|---|
| `VLLMQwen` | 已运行的 vLLM（`nvidia/Qwen3.6-35B-A3B-NVFP4`） | 文本、图片、视频 → 文本，支持 `response_schema` 和 `chat(messages)` |
| `ClaudeSonnet` | Claude Code CLI（`claude-sonnet-5-5`，effort `low`/`medium`） | 同上；视频按 fps 抽帧 |
| `Qwen` | 进程内（Linux：Qwen3.6-35B-A3B；Mac：Qwen3-VL-8B；`COCREATION_QWEN_MODEL` 可覆盖） | 同上；Qwen3-VL 模型支持 system prefix KV cache（`system_cache_path=`） |
| `MiniMaxH3` | ComfyUI + H3 Base-Ref2VA FP8（默认无 LoRA，50 步；`H3Runtime(turbo=True)` 启用 Turbo v4-600，6 步） | 文本 + 图片、视频、音频参考 → 带音轨的 MP4 |
| `MiniMaxH3Image` | ComfyUI + H3-Image W4A8（6 步，CFG 1，Euler/simple，约 16 MP 画布） | 文本 + 图片参考 → PNG |
| `Flux` | FLUX.2-klein-4B | 文本、图片 → PNG（遗留） |
| `VDNH3` | FreeVideo CLI（[vdn.md](vdn.md)） | 遗留 |

### 引用绑定

```python
image = Reference("image_2", "/path/to/image.jpg", "image")
video = Reference("video_4", "/path/to/video.mp4", "video")
MiniMaxH3(output_dir=..., runtime=H3Runtime(accept_model_license=True))(
    "The subject from {{artifact:image_2}} moves like in {{artifact:video_4}}.",
    [video, image], seconds=5, seed=42, dry_run=True)
```

- 只替换 `{{artifact:ID}}` 形式的引用。未附加的 ID、重复 ID、格式错误都会报错。
- 图片、视频、音频按引用列表中的出现顺序分别编号。H3 系列使用 `<Picture N>` / `<Video N>` / `<Audio N>`；Qwen / FLUX 使用 `<Image N>`。
- H3 限制：最多 9 张图、3 段视频、3 段音频，合计最多 12 个；视频和音频参考时长 2–15 秒；生成时长 4–15 秒，按 24 FPS 向上取整到 17k+5 帧网格；尺寸必须是 32 的倍数。参考视频长于目标时长会报错，不会被静默截断。
- `dry_run=True` 返回可序列化的计划：不读取素材、不启动进程、不联网。

### 生命周期

所有模型对象都采用 lazy persistent session：首次真实调用时启动对象专属的 worker，后续调用复用。`close()`、退出 `with`，或对象被回收时释放资源。Qwen 不保存跨请求的聊天历史。同一进程内的调用共享一把串行锁。每次请求保存 `request.json`、`runtime.log`、`result.json`（H3 还会保存 `workflow.json`、`history.json` 和预处理后的输入）。失败时抛异常，不会返回占位结果。

### ComfyUI 运行时

- H3：`scratch/runtime/minimax-h3/`（独立 venv、ComfyUI 固定提交、模型符号链接到 HF 缓存）。首次真实调用会自动安装，提交号和 HF revision 固定在 `cocreation/h3.py`。torch 2.10.0+cu128。只启用固定版本的 Turbo 自定义节点。
- H3-Image：`scratch/runtime/minimax-h3-image/`，用 `python scripts/setup/setup_h3_image.py` 安装（需在 GPU 节点上运行），ComfyUI 提交号固定在 `cocreation/h3_image.py`。
- `H3Runtime(auto_setup=False, local_files_only=True, accept_model_license=True)` 禁止安装和联网。

---

## 3. 存储与环境

服务器上，仓库和大文件分开存放：

```text
/users/lwang223/cocreation/scratch -> /users/lwang223/scratch/cocreation
  cache/huggingface/hub/, cache/pip, torch, triton, cuda
  runtime/qwen3.6-35b-a3b/      vLLM（见 qwen_server.md）
  runtime/minimax-h3/, runtime/minimax-h3-image/, runtime/sglang-h3-benchmark/, runtime/vdn/
  data/, runs/, tmp/
```

`COCREATION_SCRATCH` 可以覆盖这个根目录。Linux 上既没有 `scratch` 符号链接也没有该变量时，真实调用会报错（避免大文件写进 home）。初始化：`python scripts/setup/setup_scratch.py --target /users/lwang223/scratch/cocreation`。

主 conda 环境 `cocreation`（Python 3.11）只用于控制器和轻量依赖：

```bash
conda create -n cocreation python=3.11 pip -y && conda activate cocreation
source scripts/setup/scratch_env.sh
python -m pip install -r requirements.txt     # torch 2.10.0 / torchvision 0.25.0 cu128
conda install -c conda-forge ffmpeg -y
```

各个生成运行时都有自己独立的 venv（Comfy、vLLM、SGLang、VDN），互不影响。需要的系统工具：`ffmpeg` / `ffprobe`；sonnet 后端还需要 `claude`。
