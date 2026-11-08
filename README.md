# cocreation

Multimodal co-creation with evolving intent: synthesizing interleaved creative sessions through attribute-grounded provenance graphs.

## 第一版：三个独立模型系统

| Python 接口 | 默认模型 | 输入 → 输出 |
|---|---|---|
| `Qwen()` | `Qwen/Qwen3-VL-8B-Instruct` | 文本、图片、视频 → 文本 |
| `Flux()` | `black-forest-labs/FLUX.2-klein-4B` | 文本、可选多张图片 → PNG |
| `FreeVideo()` | MiniMax H3，经 FreeVideo | 文本、图片/视频参考，或首尾帧 → MP4 |

CPG、attribute 的版本、历史引用及对话序列由上层脚本维护。这里仅封装单次推理；不生成 CPG、不隐式改写生成意图、不把视频悄悄变成图片。视频生图须由上层显式抽帧后交给 FLUX。

### Mac 上离线开发（不会下载权重）

在仓库根目录运行，无须安装 ML 库：

```sh
python3 -m unittest discover -s tests -v
python -u examples/plan.py
```

`dry_run=True` 仅返回可序列化的请求计划：不读取参考文件、不创建任务目录、不导入 torch、不启动子进程、不联网。导入包和实例化三个系统也无下载副作用。Mac 不自动使用 Metal；如要真正进行 CPU 推理，显式指定 `Runtime(device="cpu", dtype="float32")`，但那会加载/下载模型，且速度和 RAM 成本很高。FreeVideo 真实执行仅开放给 Linux NVIDIA 环境。

### GPU 机器安装与自动下载

建议 conda 只管理 Python 环境，环境内用 pip 安装项目和推理依赖。在 Linux NVIDIA 机器的仓库根目录执行：

```sh
conda create -n cocreation python=3.11 pip -y
conda activate cocreation
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -u examples/plan.py
```

`requirements.txt` 选用 PyTorch 2.10.0 / torchvision 0.25.0 的 CUDA 12.8 wheel，版本组合来自 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/#v2100)。它面向 Linux NVIDIA，不用于 Mac；不是完整的传递依赖锁文件，尚待服务器实测。无需另外用 conda 安装 `cudatoolkit` 来运行这些预编译 wheel；服务器仍须有兼容的 NVIDIA 驱动。

系统安装 CUDA Toolkit 12.9，不意味着 pip wheel 必须是 `cu129`：这里的 cu128 wheel 自带 CUDA 12.8 运行库，兼容的 NVIDIA 驱动可以运行它。Blackwell 支持从 [PyTorch 2.7 的 CUDA 12.8 构建](https://pytorch.org/blog/pytorch-2-7/)开始提供。若“CUDA 12.9”指 `nvidia-smi` 顶部的版本，它表示驱动支持的 CUDA 上限，并不代表环境实际安装了哪个 Toolkit。自定义 CUDA 扩展的编译则需另外核对 Toolkit 与 PyTorch 的兼容性。

选择 2.10.0 是为了升级同时保留当前 qwen-vl-utils 使用的 torchvision 视频读取接口；并非声称它是最新 PyTorch。FreeVideo 使用独立环境，其驱动要求与内部 PyTorch 版本仍由上游管理。

`pyproject.toml` 只保留包名、Python 要求和构建配置，所有运行依赖统一维护在 `requirements.txt`。不能只删除它：requirements 中的 `-e .` 依赖此文件安装本地 `cocreation` 包。如果未来决定完全按脚本运行，可以一起移除 `-e .` 和包安装配置，但从任意目录导入包将不再自动成立。

**`examples/plan.py` 默认只打印请求计划，不生成媒体、不下载权重。** 它默认引用 `data/samples/image.jpg` 和 `data/samples/video.mp4`，由下述下载脚本生成；dry-run 不读取文件。传入 `--run` 才真正推理，`-u` 只控制输出缓冲。

在已分配 GPU 的节点上检查环境：

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No visible GPU')"
```

本 requirements 管理主程序、Qwen 和 FLUX 的环境；FreeVideo 仍使用它自己的独立环境，避免混装其特殊依赖。安装 Python 包不会下载模型权重；首次真实推理才会下载。


Qwen 和 FLUX 在首次真实调用中通过 `from_pretrained()` 自动获取权重、processor/tokenizer、图像模型的 text encoder、VAE、scheduler 等全部配套组件。遵循 Hugging Face 的 `HF_HOME`、`HF_TOKEN` 和缓存约定；`Runtime` 还提供 `cache_dir`、`revision`、`local_files_only`。第二次读取本地缓存，但仍重新加载模型。

FreeVideo 自带独立环境与下载器。首次真实调用会克隆源代码到 `VideoRuntime.source_dir`，固定到提交 `7c3a03c1a93580454f71628b78ac3df9d9e1880e`，然后调用官方 managed setup 的 `--no-launch` 路径，安装推理环境及其模型组件，不启动 WebUI。使用官方模型下载源。已有安装可指定 `source_dir`、`root`；`auto_setup=False` 禁止自动安装。下载/访问需要的账户授权或上游许可确认仍遵守上游行为；确认过模型条款后可设置 `accept_model_license=True` 让 setup 非交互运行。

FreeVideo 在 Linux 需要其上游要求的 NVIDIA 驱动（当前说明为 580+），以及安装工具所需的网络与磁盘空间。其缓存独立于 Qwen/FLUX 的 HF 缓存。不要在 Mac 上运行真实 FreeVideo 请求。

### 下载少量真实根素材与运行示例

之前选定的数据集是 **DAVIS 2017**（视频）和 **Open Images V7**（图片）。数据准备只用 CPU，与模型推理相互独立。

```sh
# 在已激活的 conda 环境、仓库根目录中；只需执行一次依赖安装
conda install -c conda-forge ffmpeg -y
python -m pip install -r requirements.txt

# 仅看计划，不下载
python -u scripts/download_samples.py --dry-run

# 真正下载：默认 4 张图片、bear 和 blackswan 两个视频序列
python -u scripts/download_samples.py

# 默认 dry-run；此时路径指向已下载的真实文件
python -u examples/plan.py

# 在 CUDA 节点上真实运行三个独立调用
python -u examples/plan.py --run

# 或先只验证一个模型
python -u examples/plan.py --run --model flux
```

脚本按 ID 从 Open Images 官方指定的 CVDF 镜像下载图片，流式读取少量 validation 元数据；保留图片 ID、原始元数据、来源地址。DAVIS 通过 HTTP Range 读取官方 ZIP 的目录和选中帧，不下载全部压缩包或分割标注。若服务器不支持 Range 会报错，不会自动改成下载全集；可用 `--davis-zip /path/to/existing.zip` 读取已有压缩包。当前脚本尚未真实联网下载验收。

DAVIS 原始发布形式是 JPEG 帧序列。脚本每段最多取前 72 帧，以 **24 FPS 的演示播放帧率**合成为无声 MP4（最多 3 秒，源序列不足 72 帧则更短）；这不是对原始拍摄帧率的确认。帧名、合成 FPS、实际时长写进 manifest，避免后续把合成时间轴误当原视频时间轴。可通过 `--max-frames` 和 `--fps` 调整，输出时长限制为 1–15 秒。

下载完成后的固定入口是：

```text
data/samples/
  image.jpg             # 第一张 Open Images 图片的副本
  video.mp4             # 第一段 DAVIS 合成视频的副本
  manifest.json         # 固定入口 → 原素材 ID，以及 SHA256
  images.json           # 图片来源与元数据
  videos.json           # 视频来源、帧列表、播放帧率
  openimages/*.jpg
  davis/<sequence>/frames/*.jpg
  davis/<sequence>/video.mp4
```

这些 data 文件不会进入 Git。默认小样本预计数 MB 到几十 MB（取决于帧内容；未实测），不是几百 GB 的全量下载。可以用 `--output-dir` 改下载目录，并把结果交给示例：

```sh
python -u examples/plan.py --image /data/my-image.jpg --video /data/my-video.mp4 --run
```

**真实运行的预期输出：** 三个模型顺序执行且相互独立，不把前一个生成结果再接到下一个模型。Qwen 返回对两份媒体的文字观察；FLUX 以输入图片为参考生成水彩风格图片；FreeVideo 参考图片的主体/外观和视频的镜头运动生成约 5 秒视频。随机取来的图片和视频未必语义匹配，这只是接口 smoke test，不是完整 CPG 管线，质量与属性遵循情况须看实际结果。

默认保存到仓库下（与启动时所在目录无关）：

```text
runs/example/
  qwen/<uuid>/result.json      # text、token 统计
  flux/<uuid>/image.png
  freevideo/<uuid>/video.mp4
```

每个任务目录还含 `request.json`、`result.json`、`runtime.log`；终端每完成一个模型立即打印结果和绝对路径，失败时保留已完成任务。日志内容在 `runtime.log` 中，不会持续打印模型加载进度。可以用 `--output-dir /scratch/my-run` 改输出根目录。FreeVideo 首次 setup 如果需要确认上游许可，在确认后传 `--accept-model-license`；已准备好的运行环境直接复用。

数据规模参考：DAVIS trainval 480p 官方 ZIP 的 HTTP Content-Length 为 **832,766,765 字节（约 833 MB / 794 MiB）**，包含图片和标注，脚本只读取选中帧。Open Images V7 约 900 万张图；官方指定的 CVDF 1024px 稠密标注子集约 190 万张、**561 GB**，其中 validation 41,620 张、约 **12 GB**。CVDF 对全部原分辨率图的历史估计约 **18 TB**，不是本脚本会下载的数量。

来源：[DAVIS 官方下载页](https://davischallenge.org/davis2017/code.html)、[Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)、[CVDF 下载规模](https://github.com/cvdfoundation/open-images-dataset)。

### 调用示例

```python
from cocreation import Qwen, Flux, FreeVideo, Reference, Runtime, VideoRuntime

qwen = Qwen()
image_model = Flux(runtime=Runtime(cpu_offload=True))
video_model = FreeVideo()

video4 = Reference("video_4", "/data/video4.mp4", "video")
video5 = Reference("video_5", "/data/video5.mp4", "video")
portrait = Reference("portrait_v2", "/data/portrait.png", "image")

plan = video_model(
    "The camera moves forward, just like in {{artifact:video_4}}. "
    "Use the character from {{artifact:video_5}}.",
    references=[video4, video5],
    seconds=5, seed=42, dry_run=True,
)
assert "<Video 1>" in plan["prompt"]
assert plan["bindings"]["video_5"]["label"] == "<Video 2>"

# GPU 机器上：删去 dry_run=True 才会下载/加载权重并推理。
observation = qwen(
    "Describe the camera motion in {{artifact:video_4}}.",
    [video4], video_fps=1, video_max_frames=16, dry_run=True,
)
edit = image_model(
    "Keep the identity in {{artifact:portrait_v2}}, change the hair to red.",
    [portrait], seed=42, dry_run=True,
)
animation = video_model("The character turns and smiles.", first=portrait, dry_run=True)
```

只替换 `{{artifact:全局ID}}`，不做普通字符串的数字替换。引用列表顺序决定局部编号；图片与视频分别编号。每个 ID 应指向不可变的 artifact 版本。文本中 `just like` 等属性引用语义由调用方提供，封装保留原意并把对应文件一起传给模型。未挂载的 ID 和重复 ID 会报错。

Qwen 的视频按可配置 fps 和帧数上限采样，同时将时间信息传给 processor；这只是输入采样，不是完整的上下文压缩系统。第一版 Qwen 使用确定性解码，可配置输出 token 数和视觉像素预算。

FreeVideo 自动选择 `t2va`、`ref2va`、`i2va`、`l2va`、`fl2va`。首尾帧不能与普通参考列表混用。第一版不输入参考音轨（视频参考带 `sound:false`）；生成 MP4 仍可能包含模型生成的声音。参考时长、帧规整、数量和显存限制仍由上游执行；上游可能截短长参考，具体见每次调用的 `runtime.log`。不要把接口接收多参考理解成质量或任意数量的保证。

### 单卡生命周期与输出

同一主进程内的所有真实模型调用共用串行锁。Qwen/FLUX 每次在新 Python 子进程中加载并推理；子进程退出后释放全部模型组件和 CUDA 内存。FreeVideo 通过自己的 managed 运行环境启动推理进程。这样避免三个模型同时驻留；代价是每次重新加载。**这里保留磁盘权重缓存，不保留 RAM 中的常驻模型。** 不进行模型跨卡切分。

每次真实调用创建独立 `runs/<uuid>/`，保存：

- `request.json`：原始/实际 prompt、引用顺序、全局→局部映射、模型配置、seed。
- `runtime.log`：加载和推理日志，包括失败信息。
- `result.json`：文本结果或生成媒体路径。
- `image.png` 或 `video.mp4`：生成产物；FreeVideo 另外保存 `prompt.txt`、`media.json`。

函数返回字典，含 `text` 或 `path`，以及 `job_dir`、`request_path` 和 `bindings`。失败直接抛异常，保留日志，不用占位输出伪装成功。

默认 Qwen/FLUX 使用 CUDA BF16。FLUX 可开 CPU offload；Qwen 可显式选择兼容的官方 FP8 checkpoint（例如 `Qwen/Qwen3-VL-8B-Instruct-FP8`，需要相应 GPU/运行库支持），这里不做未经验证的通用 INT8 转换。FreeVideo 通过 `prepared_format="auto"/"int8"/"fp8"` 使用其官方硬件策略。`CUDA_VISIBLE_DEVICES` 会被保留；FreeVideo 默认取其中第一张，显式 `VideoRuntime(gpu=...)` 可覆盖。不同外部主进程之间不共享这里的串行锁，调度时每个 worker 应独占一张 GPU。

### 验证范围

目前只完成无权重离线接口测试与官方源代码对照，尚未完成真实 CUDA 推理、显存测量或生成质量验收。依赖版本范围不是已验证的 GPU 环境锁文件；首次服务器 smoke test 后再锁定完整环境。

接口依据：[Qwen3-VL 官方示例](https://github.com/QwenLM/Qwen3-VL)、[FLUX 模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)、[Diffusers v0.37.0 Klein 实现](https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)、[FreeVideo 固定版本](https://github.com/FlashML-org/FreeVideo/tree/7c3a03c1a93580454f71628b78ac3df9d9e1880e)。
