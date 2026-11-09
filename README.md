# cocreation

Multimodal co-creation with evolving intent: synthesizing interleaved creative sessions through attribute-grounded provenance graphs.

## Python 接口

| 接口 | 后端 | 输入 → 输出 |
|---|---|---|
| `Qwen()` | Qwen3-VL-8B | 文字、图片、视频 → 文字 |
| `Flux()` | FLUX.2-klein-4B | 文字、图片 → PNG |
| `MiniMaxH3()` | 本地 ComfyUI / MiniMax-H3 pruned FP8 | 文字、图片／视频／音频参考 → 带生成音轨的 MP4 |

CPG、属性来源、素材版本与复杂创作意图由上层维护。封装只执行单次请求，不隐式改写 prompt，不调用官方 Context-IR 或其他云端生成服务。

```python
from cocreation import MiniMaxH3, H3Runtime, Reference

image = Reference("image_2", "/path/to/image.jpg", "image")
video = Reference("video_4", "/path/to/video.mp4", "video")

result = MiniMaxH3(
    output_dir="/path/to/output/minimax_h3",
    runtime=H3Runtime(accept_model_license=True),
)(
    "Create a new shot with the main subject and visual appearance from {{artifact:image_2}}. "
    "The camera moves just like in {{artifact:video_4}}. Keep the action coherent.",
    [video, image], seconds=5, seed=42, dry_run=True,
)
```

`dry_run=True` 返回可序列化计划，不读取素材、不创建目录、不启动进程、不导入 ML 库、不联网。包导入和实例化也不下载模型。真实运行改为 `dry_run=False`，且必须在 Linux NVIDIA GPU 节点上执行。许可开关是调用方的确认，不代替阅读模型许可，也不自动申请商业授权。

### 自动模型选择

- 没有 references：FL2VA pruned FP8，执行纯文字 `t2va`。
- 有 references：Ref2VA pruned FP8，执行多模态 `ref2va`。
- 图片是人物／风格等参考，不承诺成为第一帧；当前没有 `first`／`last` 接口。
- 不提供全尺寸 DiT 的隐式回退，也不使用 Turbo LoRA。

每次只加载一套 DiT；编码器和两个 VAE 共用。H3 并不自己决定权重，选择记录在请求的 `options.variant` 中。

### 引用与媒体语义

只替换 `{{artifact:全局ID}}`，普通文字中的数字不变。未附加 ID、重复 ID、格式错误会报错。图片、视频、音频分别按引用列表中的各自出现顺序编号。H3 使用 `<Picture 1>`、`<Video 1>`、`<Audio 1>`；Qwen/FLUX 保持原有 `<Image 1>` 标签。绑定写入请求与结果。

- H3 最多 9 张图片、3 段视频、3 段独立音频，总计最多 12 个素材；独立音频必须搭配图片或视频。
- 视频只参考画面，音轨不会隐式参与；需要声音参考时另外传 `Reference(..., kind="audio")`。
- 视频／音频参考须为 2–15 秒，同一类型总时长不超过 15 秒。
- 视频在任务目录中转成 24 FPS，原文件不变。为避免上游节点静默截断，参考视频长于目标视频时会报错，调用方须明确裁剪或增加 `seconds`。上游还会按 17k+5 帧规整参考末尾，最多去掉 16 帧。
- 生成时长接受 5–15 秒，按 24 FPS 向上规整到 17k+5 帧；`seconds=5` 实际为 124 帧（约 5.167 秒）。实际帧数和时长写入结果。
- 默认 `width=768, height=512, steps=50`，尺寸须是 32 的倍数。高分辨率、多参考和长视频的峰值显存仍需实测。
- Qwen 包装不接收音频；FLUX 仅接收图片。

## 存储与环境

服务器 repo 与大文件分离：

```text
/users/lwang223/cocreation/
  scratch -> /users/lwang223/scratch/cocreation

/users/lwang223/scratch/cocreation/
  data/samples/
  runs/example/
  cache/huggingface/hub/
  cache/pip/、torch/、triton/、cuda/
  runtime/minimax-h3/
    ComfyUI/              # 固定提交的引擎源码
    venv/                 # 独立 Python 环境
    models/               # 指向 HF 缓存的符号链接
    ready.json
  tmp/
```

初始化（不下载模型）：

```sh
cd /users/lwang223/cocreation
python scripts/setup_scratch.py --target /users/lwang223/scratch/cocreation
```

`COCREATION_SCRATCH` 可覆盖根目录。Linux 上没有符号链接且没有该变量时，真实调用会报错，避免大文件写进 home。已有普通目录、冲突链接不会被覆盖；旧文件不会被自动迁移或删除。

主环境只用于控制器、Qwen 和 FLUX：

```sh
conda create -n cocreation python=3.11 pip -y
conda activate cocreation
cd /users/lwang223/cocreation
source scripts/scratch_env.sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
python -m pip check
```

`requirements.txt` 使用 PyTorch 2.10.0 / torchvision 0.25.0 的 cu128 wheel。驱动 575 环境不应被安装步骤替换为 cu130 wheel。Comfy 使用独立 venv，避免其 PyAV、transformers 等依赖影响 Qwen/FLUX。需要 `git`、Python 的 `venv`、网络、足够的 scratch 空间；视频／音频预处理需要 `ffmpeg` 与 `ffprobe`。

### 首次真实调用自动准备 H3

首次调用 `MiniMaxH3(...)(..., dry_run=False)` 时：

1. 验证输入及许可确认，在 scratch 建立独立环境。
2. 克隆 ComfyUI 并固定提交；安装 cu128 PyTorch 和 Comfy 依赖。`pip check` 通过后写入准备标记。
3. 检查 CUDA 可用，再通过 `huggingface_hub.hf_hub_download` 获取选中任务的四个文件。
4. 启动仅监听 `127.0.0.1` 的临时 Comfy 服务，禁用自定义节点和云端 partner 节点。
5. 提交生成的 API 工作流，等待完成、收集 MP4，并在成功或失败后关闭服务进程。

**现在安装主项目、运行测试或 dry-run 都不下载权重。** 首次纯文字调用只下载 FL2VA DiT；首次参考调用只下载 Ref2VA DiT。后续复用 HF 磁盘缓存，不重复下载共用文件。HF 认证继承 `HF_TOKEN` 等环境配置。

固定版本（实现常量位于 `cocreation/h3.py`）：

- ComfyUI：`0decb39ee5b5e17c200334589fc317dbbe98310e`
- Hugging Face `Comfy-Org/MiniMax-H3`：`54472221d27c695fca9621b0c3ed52f06c4af707`

下载清单：

```text
diffusion_models/minimax_h3_{fl2va或ref2va}_pruned_fp8_scaled.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

FP8 DiT + NVFP4-AWQ 编码器负责降低主要权重占用；VAE 保持 FP16/FP32，避免依赖 cu130 的 ConvRot VAE 路径。NVFP4-AWQ 是存储格式，不承诺原生 NVFP4 加速。Comfy 的 cu128 路径会禁用它要求 cu130 的优化 CUDA 后端，使用可用的回退算子；实际性能待 GPU 验收。

权重缓存完成后，可使用 `H3Runtime(auto_setup=False, local_files_only=True, accept_model_license=True)` 禁止环境安装和 HF 联网下载；缺少所需文件会报错，不自动切换任务或模型。`root` 可指定已准备的独立运行目录。`timeout` 默认 7200 秒，`startup_timeout` 默认 300 秒。

## 示例与样本

```sh
# 不下载、不推理
python examples/plan.py --model minimax_h3

# 下载少量根素材（不是模型）
python scripts/download_samples.py --dry-run
python scripts/download_samples.py

# 首次真正执行才下载 H3 模型、准备环境
python examples/plan.py --run --model minimax_h3 --accept-model-license
```

示例中的 `qwen`、`flux`、`minimax_h3` 可分别运行；默认 `all` 依次调用。`--image`、`--video` 可传自己的素材，`--output-dir` 可改输出目录。

样本源是 DAVIS 2017 和 Open Images V7。脚本下载少量 DAVIS JPEG 帧并以 24 FPS 合成为无声视频；这只是演示播放帧率，不是对原视频 FPS 的确认。使用 HTTP Range 读取 ZIP；不支持 Range 时可用 `--davis-zip /path/to/existing.zip`。素材来源、原始元数据和合成配置写入 manifest。数据准备独立于模型推理。

## 生命周期、日志与验证

同一主进程中三个模型共享串行锁。Qwen/FLUX 使用独立子进程，H3 使用任务 worker 管理 Comfy 进程；调用结束释放显存，不让三个模型同时常驻。H3 同一个运行目录另有进程锁，避免并发修改环境。不同主进程的 Qwen/FLUX 不共享锁，服务器调度仍应保证 worker 独占 GPU；保留调度器提供的 `CUDA_VISIBLE_DEVICES`。

每个任务保存 `request.json`、`runtime.log`、`result.json`；H3 另存 `workflow.json`、`history.json`、预处理输入和 `video.mp4`。返回结果含路径、绑定、版本、实际帧数。失败抛异常并保留日志，不返回占位生成结果。

```sh
python3 -m unittest discover -s tests -v
python3 examples/plan.py
```

已做离线接口、路由、下载清单、参考视频规整、错误传播和服务清理测试，并对照固定 Comfy 源码核实节点接口。**尚未下载权重或进行 GPU 端到端推理**，因此 575 驱动上的依赖安装、显存峰值、NVFP4 回退与生成质量仍须在服务器首次运行验收。依赖包含上游版本范围，不是完整的传递依赖锁文件。

参考：[H3 模型文件](https://huggingface.co/Comfy-Org/MiniMax-H3)、[Comfy H3 节点](https://github.com/Comfy-Org/ComfyUI/blob/0decb39ee5b5e17c200334589fc317dbbe98310e/comfy_extras/nodes_minimax_h3.py)、[Comfy 本地 API](https://docs.comfy.org/development/comfyui-server/comms_routes)。

## VDN-H3 (FreeVideo CLI, FP8)

`VDNH3` adds mixed image/video/audio references with two-pass **8+3** generation.
It uses an isolated Python 3.12 runtime constrained to **torch 2.10.0+cu128**;
no ComfyUI web server or automatic cu130 upgrade. See [VDN setup and GPU testing](docs/vdn.md).

```bash
python scripts/setup_vdn.py --probe
python examples/plan.py --model vdn_h3 --run
```

The CUDA compatibility adaptation is ready for server testing, not yet GPU validated.
