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

Qwen 将文本中首次出现的已绑定 `<Image N>` / `<Video N>` 标签与对应媒体放在一起；未提到的媒体按引用顺序放在文本前。重复标签只保留文字，不重复编码媒体；未绑定的普通标签保持文字。编号始终由引用列表决定，不因文本位置改变。这一步仅组装结构化 content，随后仍使用官方 chat template、`process_vision_info` 与 processor。

- H3 最多 9 张图片、3 段视频、3 段独立音频，总计最多 12 个素材；独立音频必须搭配图片或视频。
- 视频生成模型默认同时参考视频画面和音轨，也支持独立的 `Reference(..., kind="audio")`。Qwen3-VL 只读取视频画面，不接收独立音频输入。
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
python scripts/setup/setup_scratch.py --target /users/lwang223/scratch/cocreation
```

`COCREATION_SCRATCH` 可覆盖根目录。Linux 上没有符号链接且没有该变量时，真实调用会报错，避免大文件写进 home。已有普通目录、冲突链接不会被覆盖；旧文件不会被自动迁移或删除。

主环境只用于控制器、Qwen 和 FLUX：

```sh
conda create -n cocreation python=3.11 pip -y
conda activate cocreation
cd /users/lwang223/cocreation
source scripts/setup/scratch_env.sh
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
python examples/plan.py --model vdn_h3

# 下载少量根素材（不是模型）
python scripts/setup/download_samples.py --dry-run
python scripts/setup/download_samples.py

# 首次真正执行才下载 H3 模型、准备环境
python examples/plan.py --run --model vdn_h3 --accept-model-license
```

示例中的 `qwen`、`flux`、`vdn_h3` 可分别运行；默认 `all` 依次调用。`--image`、`--video` 可传自己的素材，`--output-dir` 可改输出目录。

样本源是 DAVIS 2017 和 Open Images V7。脚本下载少量 DAVIS JPEG 帧并以 24 FPS 合成为无声视频；这只是演示播放帧率，不是对原视频 FPS 的确认。使用 HTTP Range 读取 ZIP；不支持 Range 时可用 `--davis-zip /path/to/existing.zip`。素材来源、原始元数据和合成配置写入 manifest。数据准备独立于模型推理。

## 生命周期、日志与验证

四个模型对象均采用 lazy persistent session：实例化和 dry-run 不加载模型；首次真实调用启动对象专属 worker，后续调用复用。Qwen/FLUX 缓存已加载的 model/processor/pipeline；MiniMaxH3 保持 Comfy 服务和 loader cache；VDNH3 连接 FreeVideo 原生 resident worker/model bank，而不是每轮启动一个独立推理 worker。

`model.close()`、退出 `with` 或对象最后一个引用被释放时关闭 worker，归还其资源。`close()` 可重复调用，之后再次生成会重新加载。`model.generate(...)` 是 `model(...)` 的别名。会话异常或超时会关闭失败 worker；下一次调用重新加载。不要在已加载对象上改 model_id/runtime；先 close。

权重复用与聊天历史独立：Qwen 每次调用只组装本次 messages，不保存聊天历史或跨请求 KV cache。可以将同一个 Qwen 对象传给找帧工具，复用权重而保持干净上下文。

同一主进程的调用共享串行锁，但不同对象可以同时持有权重；切换大模型前由调用方 close 不再使用的对象。不同主进程不共享这个锁。视频后端内部仍保留原生显存调度：offload、显存/内存压力或模型分区/执行配置变化可能淘汰组件并重新加载；持久 session 并不保证每个视频组件永远驻留 HBM。保留调度器提供的 `CUDA_VISIBLE_DEVICES`。

每个任务保存 `request.json`、`runtime.log`、`result.json`；`runtime.log` 指向该对象首次请求目录下的 `session/session.log`，后续请求共用这一进程日志；H3 另存 `workflow.json`、`history.json`、预处理输入和 `video.mp4`。返回结果含路径、绑定、版本、实际帧数。失败抛异常并保留日志，不返回占位生成结果。

```sh
python3 -m unittest discover -s tests -v
python3 examples/plan.py
```

已做离线接口、路由、下载清单、参考视频规整、错误传播和服务清理测试，并对照固定 Comfy 源码核实节点接口。**尚未下载权重或进行 GPU 端到端推理**，因此 575 驱动上的依赖安装、显存峰值、NVFP4 回退与生成质量仍须在服务器首次运行验收。依赖包含上游版本范围，不是完整的传递依赖锁文件。

## VDN-H3 (FreeVideo CLI, FP8)

`VDNH3` adds mixed image/video/audio references with two-pass **8+3** generation.
It uses an isolated Python 3.12 runtime constrained to **torch 2.10.0+cu128**;
no ComfyUI web server or automatic cu130 upgrade. See [VDN setup and GPU testing](docs/vdn.md).

```bash
python scripts/setup/setup_vdn.py --probe
python examples/plan.py --model vdn_h3 --run
```

The CUDA compatibility adaptation is ready for server testing, not yet GPU validated.


## 对象生命周期与找帧工具

```python
from cocreation import Qwen, Flux, Reference
from tools.find_video_frame import find_video_frame

qwen = Qwen()  # 不加载
first = qwen("Describe the image.", [Reference("i", "/data/image.jpg", "image")])
second = qwen.generate("Describe its colors.", [Reference("i", "/data/image.jpg", "image")])
# 同一套权重，两个互不携带聊天历史的请求。
timestamp = find_video_frame("/data/video.mp4", "The person facing the camera, face unobstructed",
                             qwen=qwen, frame_output="/data/selected.png")
qwen.close()  # 切换大模型前显式释放；del qwen 且没有其他引用也会释放。
if timestamp is not None:
    with Flux() as flux:
        image = flux("Use the person from {{artifact:frame}} in a watercolor portrait.",
                     [Reference("frame", "/data/selected.png", "image")])
```

命令行：

```sh
python tools/find_video_frame.py scratch/data/samples/video.mp4 \
  "A clear front view of the main subject" --frame-output scratch/runs/reference.png
```

stdout 仅输出一个秒数或 `null`。时间以视频第一个显示帧为零点；最终帧选择使用 ffprobe 的实际帧时间戳；官方粗定位使用帧索引/FPS 的时间编码，对可变帧率视频可能有近似误差。第一轮直接将视频传给官方 `qwen-vl-utils==0.0.14` 与 `Qwen3VLProcessor`，默认请求 2 fps、最多 24 帧（由官方按时间 patch 约束调整），让 Qwen 返回粗定位时间或 null。第二轮在粗定位附近按真实 PTS 抽出最多 17 张图，Qwen 选择候选 ID，程序映射为确定的原视频帧与时间戳。最终返回值不是第一轮模型估计的时间。`--video-fps`、`--coarse-frames`、`--refine-frames` 可配置预算。搜索范围有限，`null` 表示在检查的候选帧中未找到，不证明全片不存在。

工具的 Python 返回值为 `float | None`；可选 `frame_output` 导出选中 PNG，默认不覆盖已有文件。详细来源、原视频帧序号、两轮候选及选中 PNG 保存在 `scratch/runs/find_video_frame/<id>/`；未找到时不导出图。推理失败、无效 JSON、非法候选编号会报错，不伪装成未找到。支持本地视频路径，不自动下载远程 URL。

工具自行创建的 Qwen 在返回前关闭；调用方传入的 Qwen 保持可复用。需要 ffmpeg/ffprobe；Mac 上可测抽帧和接口，真实 Qwen/FLUX/H3 仍在 CUDA 服务器验收。CPG 属性与模态抽样的讨论稿见 [docs/cpg.md](docs/cpg.md)。


### Extract visual attribute nodes

`Qwen(...)(prompt, references, system_prompt=None)` accepts a per-call system
message. None preserves the user-only input; system messages are not retained
between calls and do not affect model-weight reuse.

```python
from cocreation import Qwen, Reference
from tools.extract_attribute_nodes import extract_attribute_nodes

with Qwen() as qwen:
    result = extract_attribute_nodes(
        Reference("root_video", "scratch/data/samples/video.mp4", "video"), qwen=qwen)
```

The tool accepts a Reference or `{artifact_id, path, kind}` mapping. A supplied
Qwen remains open; an internally created Qwen is closed on every exit. It returns
a Python dictionary (JSON object), or None for invalid model output. Missing files
and inference errors raise. The CLI prints JSON/null:

```bash
python tools/extract_attribute_nodes.py scratch/data/samples/image.jpg --artifact-id root_image --kind image
```

The prompt file contains a copied Definitions snapshot without the document's
narrative examples, followed by two complete JSON examples. All 29 categories are
present, with null for unobserved categories. Observed entries have local IDs,
text (at most 15 English whitespace-separated words), parent_id, links and optional
time_range. Arrays encode multiple parents even for per-parent single values.
Validation checks schema, cardinality, reference types and image temporal exclusions;
it cannot certify observational accuracy. Video extraction uses official sampled
video input (default 2 fps, cap 32 frames), so brief events can be missed. No CPG
storage is modified. The future orchestrator belongs in scripts/cpg/.
