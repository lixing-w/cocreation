# 对话媒体的实际制作参考规则

`inputs` 描述对话里看起来怎样制作媒体；`reference_materials` 描述离线实际怎样构造这些媒体。
两者使用同一套 `[[image_0001]]` / `[[video_0001]]` 占位符，但表示两张不同的依赖图。
`inputs` 只允许指向对话里已经出现的媒体；`reference_materials` 可以指向对话后面出现的媒体。
实际制作按 `reference_materials` 的拓扑顺序执行，不能按对话消息顺序执行。

| 媒体类别 | 实际参考来源 | 规则 |
| --- | --- | --- |
| 最终 target video | 无 | `final.media` 是唯一根，`reference_materials=[]`、`reference_depth=0`；其 `source_video` 对应原始输入视频。它也是对话最后一个产生的媒体。 |
| 与目标设计一致的人物、物体、室内/室外环境参考；图片或短视频候选 | target video | 完整 attributes 与目标一致的版本直接从原视频提取、分离或编辑。时间提示沿用 `source_frames`。 |
| 返工前版本、被否决候选、临时替代设计、多次编辑版本 | 相邻版本 | 把同一资产的 `edit_parent` 关系视为无向编辑图，从目标兼容版本向外定向。因此红色初稿可以引用后来出现的蓝色目标版本，且无需同时重复引用 target video。 |
| 没有编辑路径连接目标的独立候选 | 该资产最终采用的目标版本 | 在该候选所在编辑分支中，选择与目标 attributes 差异最少的版本作为起点，再沿已有编辑关系向外派生；相同分数按占位符稳定排序。 |
| 静态组合试拍、任务切换中的测试视频 | 对话前文已经准备的资产 | `purpose=exploration` 的测试视频保留前向组合，用其 `inputs` 作为实际参考。不会强行从目标里找一个根本不存在的试拍。 |
| 黑白预览、一次性原色预览 | 原来的资产参考 | `purpose=exploration` 的预览图片从既有参考派生，不改采用版本。 |
| 正式镜头，以及正式镜头返工前/后的版本 | target video | `purpose=shot` 的镜头具有目标的动作和构图结构（允许设计返工差异），直接提取/编辑目标视频。资产绑定仍由 `bindings` 约束；不依赖对话 `inputs` 重做整段。 |
| 中间剪辑、重排和替换后的预览 | 组成它的镜头版本 | 组合 `inputs` 中的已有镜头；顺序由 `shot_ids` / `shot_versions` 保留。最终剪辑本身按根处理。 |

若最后一个正式结果之后还展示了独立预览，计划会增加一次明确的最终视频导出，使最后产生的媒体仍然对应 target video。该导出的对话输入是已有成片，实际来源仍是原始 target。

## 深度

除根以外：`reference_depth = 1 + max(所有 reference_materials 的 reference_depth)`。
这是统计字段，不参与模型提示或限制编辑次数；也不同于对话中的 `edit_depth`。
例如 target → 蓝包 → 红包 → 试拍分别为 0、1、2、3。
若试拍使用的设计经历了更多派生，深度可以是 4 或更高。

东京 seed 1 的实际例子（`--reworks 2 --p-compile 0.5`）：

- `video_0008` 是 target，深度 0。
- 最终 matte 包 `image_0007` 引用 target，深度 1。
- semi-matte canvas 初稿 `image_0001` 引用 `image_0007`，深度 2。
- semi-matte wood 变体 `image_0003` 引用 `image_0001`，深度 3。
- 临时木包试拍 `video_0001` 引用 `image_0003` 和人物 `image_0002`，深度 4。
- 正式街头镜头 `video_0003` 直接引用 target，深度 1。

杯子正好体现反向关系：对话 `image_0009.inputs=[image_0005]`，表示从渐变杯改到纯色杯；
实际 `image_0005.reference_materials=[image_0009]`，表示先从 target 得到纯色杯，再反向制作渐变杯。

## 字段与消费方式

`plan.json` 和 `dialogue.json` 的每个完整 placeholder，以及 `placeholders.json` 的每个简化记录，都保存 `reference_materials` 和 `reference_depth`。
简化记录中的根还保存 `source_video`，因此仅凭占位符列表也能解析回原始视频。
Flux / H3 的 prompt writer 和正式调用应使用 `reference_materials` 解析的实际多模态材料，
同时使用描述、完整 attributes、source_frames 和正式镜头 bindings 限定哪些内容需要保留或编辑。
仅用简化记录时，完整属性/镜头绑定仍应从 plan 读取；本次生成器只规划引用，不执行真实图像或视频制作。

验证会检查引用存在、没有重复或环路、唯一根、每个深度、最后一个媒体与 target 一致，以及各类别的参考选择规则。
对话时间线的因果检查继续独立执行；实际反向引用不放宽 `inputs` 的时间顺序约束。
