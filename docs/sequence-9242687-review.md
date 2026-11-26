# Sequence 9242687 复盘

运行：`scratch/runs/sequence-9242687-20261117T205258Z`。核对了 dialogue、plan、rewriter 的 input/system、generation_plan、实际 H3 generation_records、media_manifest，并查看视频抽帧及参考图。

## Turn 13–16：提前出现的羊

- Turn 13 只要求当前人物的视频，提出竖屏 9:16；turn 14 询问与长期格式规则的冲突；turn 15 撤回竖屏要求，选择继续使用横屏 16:9。
- Turn 16 的 Python placeholder 为 `video_0001`：逻辑操作 `generate`，`asset_ids=['A1']`，`shot_ids=[]`，唯一输入是人物参考 `image_0003`。羊首次在 turn 17 被提出。
- `CrossTurnRules.preview_context()` 选择一个已 ready 的主体；`challenge()` 生成一个独立规则预览。它不是正式 target shot，不经过 `Planner.render_due()` 的整镜资产就绪检查。
- `generation.action_context()` 却按主体重叠挑选 target shot，没有要求该 shot 的所有主体都属于当前请求。于是选到 S1：`A wide shot of a canal at sunset with a person riding a pink sheep in the distance.`
- 这句话确实进入了 `prompt_rewriter/video_0001/input.txt`。该输入只保留 description，进一步丢失了 S1 包含 A2/A4 的范围信息。最终 H3 prompt 明确写了骑粉色羊、运河和日落。
- H3 system 的示例也包含羊/运河故事，存在额外的示例偏置风险；直接证据仍是上述 `action_context`。

普通 target shot 的顺序是：观察得到 shots 与 asset 依赖；准备并采纳主体版本；`render_due()` 在该 shot 的全部 asset 都 ready 后生成。跨轮规则预览是另一条路径。修正方向应是让预览只使用当前请求确认的主体与动作，不能强行引入未来主体来满足 target shot。这次先复盘该上下文泄漏，没有修改这一分支的调度与动作来源。

## Turn 38 与后续视频

Turn 38 = `video_0004`，H3 收到四张生成图，没有 source video，也没有 target video 帧：

| H3 槽位 | 参考 | 主体 | 交付 turn |
|---|---|---|---|
| Picture 1 | image_0003 | 白发、白黄外套人物 | 10 |
| Picture 2 | image_0007 | 紫色棉质羊 | 28 |
| Picture 3 | image_0001 | 绿色皮包 | 4 |
| Picture 4 | image_0006 | 黄昏运河 | 24 |

实际运行记录的 bindings 与 manifest 一致。`reference_media=[]` 仅是当前视频时长/音频元数据列表，不代表缺少图像参考；bindings 有四张图，H3 编码记录有 2304 个 reference video tokens。

- Turn 36/video_0003：人物、羊、运河三张图生成 S1。
- Turn 38/video_0004：上述四张图生成 S2。
- Turn 40/video_0005：同样四张图独立生成 S3，没有继承 video_0004。
- Turn 44：video_0003→0006、0004→0007、0005→0008，各自做一次编辑。S1 加粉色羊参考 image_0008；S2/S3 加粉色羊 image_0008 和棕色包 image_0009。
- Turn 46/video_0009：拼接 0006/0007/0008，不调用 H3；有一次 ffmpeg H.264 CRF 18 重编码。

视频生成配置始终是 1024×576、16 个 base steps、single pass、0 refine steps；成品缩放至 1280×720。抽帧中能看到人物和建筑细节偏软，但没有后期悄悄降低尺寸/步数，也不是 38→40→44 的连续多次视频编辑链。低原生分辨率与纯缩放、复杂四主体合成、turn 44 再生成均是应分别对照测试的因素；当前记录不能确定模糊的唯一原因，也不能把所有后续损失都归给编辑累积。

## 此次落实的改动

- rewriter 输入明确提供每个输出的 `output_format`：image/video、具体比例、方向。混合长期规则会解析成该输出的 1:1/16:9/9:16。
- 显式比例要求会写入 FLUX 最终 prompt，或 H3 的第一镜描述，并检查没有遗漏；执行端仍通过原生 width/height 落实尺寸。
- 本 sequence 两次竖屏提议都被撤回（turn 15、33），因此实际视频全部横屏 16:9 符合最终决策。
- 下一次 `run_pipeline.sh` 对服务、observation、dialogue/review、prompt rewriting 明确关闭 thinking；保留可选 thinking 功能。
- 未提交新实验，未修改 Slurm 日志输出位置。
