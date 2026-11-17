# 正向媒体构造

原始视频只供 observer 提取剧本、身份描述、属性和镜头动作。专家执行合同不附带原视频、原视频帧或原视频裁剪。最终视频由本条对话中生成的镜头组成。

- 首次资产可以没有输入：FLUX 文生图，或 H3 文生视频。
- 图片修改引用此前生成的图片；版本回退继续引用被选中的历史版本。
- 首次正式镜头引用已经生成的身份图片；镜头修改引用该镜头的既有视频和相关身份图片。同一身份的图片与视频绑定到同一个 Subject。
- 取消规则后重新生成资产，仍从该身份的既有素材继续；不重新无条件抽取身份。
- 图片若依赖此前生成的视频，调用 `tools/find_crop_in_video.py` 找到最佳帧和 bbox，按原分辨率裁剪，然后将裁剪图交给 FLUX。
- 镜头重排由 ffmpeg 按 `shot_ids` 和 `shot_versions` 确定性拼接。仅额外的显式外观修改需要 H3。

`reference_materials` 是输出的完整因果依赖，全部必须早于当前交付轮次。初始资产有多个零入度根，`reference_depth=0`；其余深度为输入的最大深度加一。

`dialogue.json.turns[].input_media` 在生成交付轮次记录真实依赖；澄清时也可记录重新展示的旧媒体。`intent.display_media` 指定需要追加到正文的已有附件。其他历史语义指针保存在 `intent.context_media`，不自动作为聊天附件。`output_media` 是本轮新生成的结果。

`placeholders.json.references` 是有顺序的实际专家插槽，记录类型、label、身份角色和必要的裁剪参数。确定性拼接使用 `clips` 的精确次序；它与无序依赖集合不同。

## 执行

先启动 vLLM，完成 observer、generate 和 prompt rewriting。随后在同一 GPU 节点关闭 vLLM：

```bash
bash scripts/vllm/stop.sh
python scripts/dialogue_generation/materialize.py /absolute/path/to/dialogue
```

支持 `--dry-run` 检查合同，`--resume` 继续中断的执行。专家按正向依赖执行，切换模型前释放前一个 worker。遇到视频到图片的依赖时，先释放生成模型；裁剪阶段自动临时启动 vLLM（若服务未运行），结束后只关闭由本次裁剪启动的服务，再继续生成。已有裁剪按输入视频哈希、query 和工具版本缓存于 `_inputs/generated_crops/`。

`--prepare-frames` 可提前处理已经生成的视频，不会读取 guide。尚未生成的父视频会被列为 pending，正常 materialize 到达该依赖时再处理。

真实媒体保存到同一 run 的 `media/`，`media_manifest.json` 记录状态和哈希。完成后 `dialogue.md` 使用 HTML 图片与视频标签，原占位符版本保存为 `dialogue.placeholders.md`。

## 资产和提示词

每个观察资产有简短的 `description`，记录区别性细节。只管理人类直觉上显著的身份；微小装饰或短暂次要道具不单列。其他属性优先选择 catalog 选项；模型新增的字符串字段或值在每个身份的 `attribute_extensions` 中登记，随计划和专家状态传递，不修改全局 catalog。

图像默认是独立产品照、自然人物图或连贯场景图。`generation_requirements` 仅保留显式 composition / pose。

之前包产品照实验使用的原始 prompt：

> Create one studio product photograph of the exact brown leather shoulder satchel in <Image 1>, standing upright at a natural front three-quarter angle on a neutral background with a soft contact shadow. Preserve the same flap shape, leather panels, buckle straps, shoulder strap and white bird-shaped detail visible on the bag in <Image 1>. The bag identity, construction, proportions and brown leather color must faithfully match <Image 1>. Show one bag alone.

新管线首次创建资产使用相同的独立产品照风格，但不带媒体引用：描述来自 observer 和当轮选定属性；后续修改才引用已生成的资产。FLUX / H3 rewriter 中已加入多组简短例子。H3 编辑同时接收 source_action，明确保留源动作，只将新的身份图绑定到既有 Subject。
