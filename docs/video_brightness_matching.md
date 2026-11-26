# 视频编辑亮度对齐

> **现状。** 原生 H3 的视频/音频 VAE 在反复视频编辑时会逐轮丢失细节、亮度下降。亮度对齐是最早的补救方案，只能修正整体亮度，无法恢复已丢失的细节。后来我们换成了 **H3-MT VAE**（`--h3-vae mt`，重新训练了 decoder，专门解决多轮编辑的细节丢失），最新的 run 不再需要亮度对齐。
> 当前生产命令（见 [native_batch.md](native_batch.md)）同时使用 `--h3-vae mt` 和 `--no-match-video-brightness`。注意代码默认值没变：`native_batch.py` 和 `materialize.py` 的 `--match-video-brightness` 默认仍为开启，提交时漏掉这个选项就会重新启用亮度对齐。本文其余部分描述该功能本身。

H3 视频编辑（`video_edit`）的结果经常整体变亮或变暗。开启时，物化会把编辑结果的亮度对齐到锚点视频（`cocreation/video_tone.py`，策略 `root-joint-y-v1`）。`--no-match-video-brightness`（native batch 或 `materialize.py`）或 `materialize(..., match_video_brightness=False)` 可以关闭。

## 锚点

- 普通编辑：沿可执行的 `references` 回溯编辑链（只跟随 `video_edit` 和不带 clip_edits 的 `copy` 边），以链上**第一个视频**为锚点。例如 `v1 → v2 → v3`，v2 和 v3 都对齐到 v1，但 v3 的模型输入仍是对齐后的 v2。新生成和拼接会开始一条新链。
- 修复重试：重试是对上一次被拒结果的编辑，锚点就是这次被拒的结果（`references` 的最后一项）。
- 一个 `video_edit` 必须恰好有一个视频参考作为编辑底图，否则在推理前就会报错（`brightness_anchor`）。新镜头可以有多个视频设计参考，这种情况不做对齐。

## 方法

在剪辑时间轴上均匀采样若干帧（降分辨率），用结构和色度的一致性找出稳定的空间区块，排除发生了变化的物体轮廓。联合拟合暗部、中间调、高光的分位数（并加一个较弱的全稳定区域约束），用加权单调回归得到一条作用于整段剪辑的 Y 通道 LUT。合法的黑白端点和越界值保持不变，U/V 不调整，所有帧使用同一条曲线。只有在独立的留出帧上平衡误差确实改善、且没有任何一个色调段明显变差时，才应用这条曲线。

以下情况会跳过对齐：稳定内容不足、构图或色彩编码不兼容、HDR 或位深不受支持、没有可验证的改善。跳过时保留模型的原始输出，并记录原因。对齐会以 libx264 **CRF 9** 重新编码视频，保留几何尺寸、时间戳和音频包。

## 记录与缓存

- `media_manifest.json` 的顶层 `video_brightness_matching` 列出每个任务计划使用的锚点（`--dry-run` 也会输出）。
- 每条编辑记录的 `brightness_matching` 字段包含：锚点引用和 sha256、状态和原因、稳定区域覆盖率、留出误差、实际应用的 LUT。
- 对齐策略和锚点哈希参与编辑任务的 fingerprint。续跑时打开对齐，只会让编辑及其下游任务失效，独立生成的根节点仍可复用。
