# 从目标视频观察构造多轮对话

输入是视频观察 JSON；`generate.py` 不读取视频，也不提取帧。
上游工具 `tools/video_observer.py` 负责从视频产生这个 JSON。
Python 决定动作、对象、候选数量、采纳、回溯、任务切换、返工范围和拼接顺序。
Qwen 将每条已确定的意图表达为自然英文，再由同一模型使用独立 SYSTEM 验证；失败时重写。
不生成图片或视频；媒体全部是占位符。主对话完成后自动调用纯文字 prompt rewriter，为每个生成任务写入 `generation_prompt` 和专家执行合约；纯提取/拼接记录为确定性操作。详见 [generation prompts](../../docs/generation_prompts.md)。

## 运行

Oscar 正式批跑从主仓库提交：

```bash
cd ~/cocreation
sbatch run_native_dialogue_batch.sh
```

这个 launcher 默认使用 24 条 sequence 清单、Sonnet 5.5 `medium` 和单卡 B200。正式作业直接读取主仓库。`data/` 或运行目录中的代码快照只用于复盘和调试。后续代码改动合入主仓库。

Linux 默认通过 vLLM 使用 **Qwen3.6-35B-A3B NVFP4**，`run_pipeline.sh` 在复盘 sequence 9242687 后默认关闭 thinking；独立工具仍可用 `--enable-thinking` / `--no-enable-thinking` 显式选择。视频观察、对话和 prompt 改写共用服务，支持并发 batching。Mac 仍默认使用下表中的 Ollama Instruct 模型。部署和新模型命令见 [服务器指南](../../docs/qwen_server.md)。
模型别名与完整名称见 `tools/dialogue_models.py`：

| 仓库别名 | Ollama 模型 | local checkpoint |
| --- | --- | --- |
| `qwen3-vl-30b-instruct`（默认） | `qwen3-vl:30b-a3b-instruct-q4_K_M` | `Qwen/Qwen3.6-35B-A3B` |
| `qwen3-vl-30b-thinking` | `qwen3-vl:30b-a3b-thinking-q4_K_M` | `Qwen/Qwen3.6-35B-A3B`（enable_thinking=true） |

首次安装：`ollama pull qwen3-vl:30b-a3b-instruct-q4_K_M`。示例：

```bash
python scripts/dialogue_generation/generate.py \
  scripts/dialogue_generation/examples/tokyo_vlog.json \
  --output-dir scratch/runs/dialogue-demo --seed 5 --reworks 1
```

另一个虚拟输入是 [alpine_kayak_vlog.json](examples/alpine_kayak_vlog.json)：晴朗白天的
高山湖皮划艇 vlog，4 个镜头、6 个资产，包含跨镜头复用的皮划艇、木桨和防水盒。

```bash
python3 scripts/dialogue_generation/generate.py \
  scripts/dialogue_generation/examples/alpine_kayak_vlog.json \
  --output-dir scratch/runs/alpine-kayak-demo \
  --backend ollama --seed 1 --reworks 2 --p-compile 0.5
```

其他 Ollama 服务可指定 `--base-url http://host:11434`。`--model` 接受上述仓库别名或完整模型名。
本机旧名称 `qwen3-vl:30b` 仍是 Thinking，不会被偷偷重指向 Instruct；新运行请用明确别名。
自定义名称可通过 `--model-mode instruct|thinking` 指定模式。`--backend local` 复用仓库 CUDA Qwen 对象；全程不调用 FLUX/H3。
输出目录必须不存在，避免覆盖其他运行。
若中途失败，可用相同输入及规划参数，增加 `--resume` 继续已有目录；脚本校验计划、人设、
模型及已完成前缀。脚本保留完整前缀，需要承接前文的改写最多提供最近三条消息，
不会重新调用已完成消息的 Qwen。
修改规划规则或切换模型后，请使用新的输出目录；旧的 8B checkpoint 不会自动迁移到 30B。

此前旧属性规则下，用本机真实 Qwen 运行的历史示例（seed 296、`--reworks 1 --p-compile 1`）：
[可读对话](examples/tokyo_dialogue.md)、[对话 JSON](examples/tokyo_dialogue.json)、
[占位符清单](examples/tokyo_placeholders.json)。本次为可爱型用户，60 条消息、17 个媒体占位符，
覆盖 15 类行为；第一张包的参考图被隐式采纳，无额外确认消息。此样例保留作历史参考，
不是当前属性采样规则下的重新运行结果；新规则下同一 seed 会得到不同计划。

可调参数：

- `--seed`：只控制动作、对象、候选、顺序和人设的规划，不固定 Qwen 措辞随机性。
- `--temperature 0.8 --top-p 0.95 --top-k 50`：默认随机采样，Ollama 和原生 Qwen 显式使用同一组参数。
  temperature 必须大于0；原生 Qwen 同时开启 `do_sample=True`。
- `--rewrite-seed 123`：单独固定措辞随机种子。省略时，每次新运行抽一个随机种子，
  所以相同规划 seed 也能产生不同措辞；在相同推理环境中显式固定此参数便于复现。
  实际参数和种子记录在 `dialogue.json` 的 `realization.sampling`、`realization.rewrite_seed` 中。
  `--resume` 自动恢复已保存的改写种子，并核对采样参数；重试使用不同种子。
- `--reworks 0|1|2`：预留返工的不同资产数量，默认 1，最多不超过可编辑资产数；不限制准备阶段的试错次数。
- `--p-compile 0.65`：每个新正式镜头出现后，产生拼接预览的概率。
- `--modules direct candidates trial_edit clarify relative_preview task_switch temporary_rollback`：
  限定准备模块。先从合法模块抽取，再抽所需资产。跨任务模块需要两个未完成资产；
  只剩一个资产且没有可用单任务模块时，回退到 `direct`。
- `--no-extra-behaviors`：关闭可选的指代澄清、约束生命周期与拍摄要求冲突片段。
- `--max-tokens 256 --timeout 180`：每条 Qwen 回答的预算，最终台词仍要求简短。
- `--user-persona casual|cute|precise|verbose|concise`：固定用户口吻；默认在启动时按 seed
  随机抽取一次，整条对话保持不变。五种口吻分别为随意、可爱、冷静准确、啰嗦、简练。
  助手始终使用温和、专业、简洁的人设，不模仿用户的口癖。
- `--context-tokens 32768`：本次改写提示词的上下文预算，包括 SYSTEM、人设示例、必要的近期上下文和当前任务。
  保守检查超出预算时明确报错，不静默截断这三条消息。
- `--quiet`：关闭逐条进度。

## 从视频生成输入 JSON

在仓库根目录运行：

```bash
python3 tools/video_observer.py scratch/runs/data/story_example.mp4 \
  --output-dir scratch/runs/story-observation \
  --model qwen3-vl-30b-instruct

python3 scripts/dialogue_generation/generate.py \
  scratch/runs/story-observation/observation.json \
  --output-dir scratch/runs/story-dialogue \
  --backend ollama --model qwen3-vl-30b-instruct \
  --seed 1 --reworks 2
```

Ollama 通过 `/api/chat` 接收按时间排序的抽帧图片，**没有直接发送 MP4，也没有使用原生视频编码或音轨**。
默认在全片实际显示时间上均匀选择最多 24 帧（包括首尾），长边最多 768 像素。
可用 `--frames` 和 `--max-side` 调整；短暂事件可能落在采样间隙中，镜头边界是相邻采样帧时间的中点估计。
当前下游契约仅支持 1–4 个镜头；更多镜头会合并为最多四个相邻片段。

先观察主要资产与镜头，再逐个资产观察一两张代表帧，填写对应类别的全部属性。
每次只给出当前类别的字段、候选值与相容性规则，不给对话规划、虚构样例或生产要求。
随后逐资产保守复核：每个字段只能保留提议值或改成 `unknown`，不允许换成另一个推测值。
这可以撤回没有视觉依据的选择，但枚举与复核仍不能保证视觉判断完全正确。
属性值直接取自 `attribute_catalog.json`，Ollama 使用 JSON Schema 约束输出，程序再校验枚举、
相容性、重复资产名、资产引用和采样帧编号。不能判断的字段使用 `unknown`，不适用的字段使用 `not applicable`。
资产名和镜头动作仍需简短英文，不能从属性枚举中生成这些自由文本。程序以资产名填入稳定描述，减少重复和额外推测。
程序填入视频路径、版本、真实采样时间、镜头顺序及空的 `constraints`，不会从画面推导用户规则。

最终输入是 `observation.json`；`sampling.json`、抽帧图片、`inventory.json` 以及各阶段的提示词、
Schema、原始回答和校验错误都保留在同一目录。校验失败默认额外尝试一次，仍失败则不产生最终 JSON。
输出目录必须不存在。Ollama 保持模型权重 15 分钟，后续对话生成使用相同模型名即可复用服务中的权重。

Python 调用支持复用现有 CUDA Qwen；调用方传入的实例不会被关闭：

```python
from tools.video_observer import observe_video

observation = observe_video(
    "scratch/runs/data/story_example.mp4",
    "scratch/runs/story-observation-local",
    qwen=qwen,
)
```

也可用 `VideoObserver(...)` 连续处理多个视频。`--backend local` 创建工具自有的 CUDA Qwen，
结束后关闭；Ollama 和 local 都观察同一套抽帧，不保留跨请求聊天历史。

## 输入约定

完整输入见 [tokyo_vlog.json](examples/tokyo_vlog.json) 和
[alpine_kayak_vlog.json](examples/alpine_kayak_vlog.json)，所有资产已补齐对应类别的属性。
这是新版输入契约，旧 `identity` / `environment` 和缺少字段的属性表会明确报错，不自动猜测。

四类枚举的唯一来源是 [attribute_catalog.json](attribute_catalog.json)：

| kind | 字段数 | 范围 |
| --- | --- | --- |
| person | 19 | 性别呈现、外观年龄、头发、五官、妆容、身高身材、服装饰品 |
| object | 13 | 颜色及明度饱和度、大小形状比例、材质表面、图案装饰、新旧、透明度 |
| indoor_environment | 19 | 用途、风格、空间结构、墙地材质、家具窗户、整洁人群、光线时段、呈现机位 |
| outdoor_environment | 20 | 场景地形、季节天气、地表植被水体、能见度风力人群、光线时段、呈现机位 |

[observation_prompt.md](observation_prompt.md) 可以直接提供给观察视频的上游 Qwen，包含全部字段、
全部候选值、相容性要求和三个完整观察 JSON 的输入输出示例。这是独立导出的完整提示词；`tools/video_observer.py` 使用逐阶段提示词，只提供当前任务需要的信息。
[observation.schema.json](observation.schema.json) 是同一份枚举自动导出的 JSON Schema。
修改 catalog 后重新导出：

```bash
python3 scripts/dialogue_generation/observation_contract.py
```

下面是一个完整的单资产输入：

```json
{
  "video": "fictional://demo.mp4",
  "assets": [
    {"id": "bag", "kind": "object", "name": "托特包",
     "description": "宽肩带托特包", "frames": [1.0],
     "attributes": {
       "color": "蓝色", "color_lightness": "适中", "color_saturation": "中等饱和",
       "size": "未知", "shape": "矩形", "aspect_ratio": "未知", "material": "帆布",
       "surface_finish": "哑光", "texture": "编织", "pattern": "无图案",
       "decoration": "无装饰", "condition": "未知", "transparency": "不透明"
     }}
  ],
  "shots": [
    {"id": "S1", "description": "托特包静物特写", "assets": ["bag"]}
  ],
  "target_order": ["S1"]
}
```

- 目标共 1–4 个镜头。`shots[].assets` 是唯一必需的资产依赖表。
- `kind` 必须是上述四种之一，不再根据名字或描述猜类型。
- 每个资产给 1–2 个非负秒数，作为未来媒体反向提取的来源记录；本脚本不访问这些帧。
- `description` 只补充枚举未覆盖的稳定内容，不能重复颜色、材质、发型等可变属性。
  基础设计由 `attributes` 给出，当前有效用户规则可覆盖其输出外观。`name` 应是不含版本颜色、材质等的短名称。
- `attributes` 必填该类别的全部字段，每个字段只取一个枚举值。VLM 新增字段或枚举值会作为该 identity 的扩展保留并参与后续阶段。
  看不清/无法推断选 `未知`，没有意义选 `不适用`；这两种值不参与变化采样，不从描述猜补。
- 镜头 `description` 建议写动作、构图、环境关系，可变资产外观由 `assets` 的参考决定。
- `target_order` 可省略，默认按 `shots` 列表顺序。
- `constraints` 可选，必须给出可执行的结构化规则。`scope` 是 `global` 或 `partial`；
  global 覆盖全部新产生的图片和视频，partial 用 `selector.kinds`（image/video）或 `selector.asset_ids` 指定范围。
  `effect` 支持 `aspect_ratio`、`season`、`asset_attributes`。它是用户规则，不是永久属性锁，允许冲突、例外和替换。
  顶层 `asset_attributes` 可转换为结构化 effect；不从纯文本猜测规则。
  规则不需要与最终画面一致：当前接受的规则和设计可以保留至最终成片；只有明确的用户请求才会取消规则，不会为贴近 guide 自动恢复。

- `video` 是来源标识，不要求文件存在；额外 VLM 字段会保留/忽略，不触发推理。
- 每个管理资产必须至少被一个目标镜头使用。不需要把背景中的所有小物体拆出来。
- 初版假设每个资产只有一个终态外观；若目标内同一个包本来就有不同设计，
  先以不同资产 ID 表示，避免将原片合法差异错误当成返工。

## 并行规则层 cross_turn_rule

`cross_turn_rule.py` 与单资产/双资产准备模块并行，不占它们的 `--modules` 抽选名额。
默认随额外行为启用；`--no-cross-turn-rule` 只关闭随机规则事件，显式输入规则仍会建模。
`--no-extra-behaviors` 同时关闭随机规则、指代歧义和单轮拍摄冲突。

例如全局秋季背景，以及仅对视频生效的场景属性：

```json
[
  {"scope": "global", "effect": {"season": "fall"}},
  {"scope": "partial", "selector": {"kinds": ["video"]},
   "effect": {"asset_attributes": {"scene": {"time_of_day": "night"}}}}
]
```

媒体只有 `image` 和 `video`，没有 trial/formal/final 类型。人物/物品/场景仍是主体内容标签，用于选择合法 variation。
生成、编辑、拼接、导出保留为操作；主体绑定、版本和拼接顺序保留为引用关系。
`asset_ids` 选择涉及指定主体的输出，不代表局部像素遮罩。

规则只在完整工作段之间插入。季节规则支持保留、一次请求例外、永久替换和取消。
比例规则支持明确设置、永久替换和一次请求例外，不提供取消。
明确的单次比例请求直接执行，无需再次询问许可；下一次请求恢复持续规则。
例外以当前请求的实际媒体比例比较，不能拿名称不同但图片比例同为 1:1 的 preset 制造假冲突。
同一个请求含图片/视频时，指定图片的例外只作用于图片。

每个 placeholder 记录：

- `base_description`：基础内容/设计，后续规则变动不会回写历史版本。
- `render_state`、`applied_rules`：这个输出生成当时的有效要求和规则快照。
- `effective_attributes`：资产在适用规则下的属性；`attributes` 保留基础设计属性。
- `reference_render_states`：输入参考的既有处理，避免把参考图的局部处理误当成当前输出规则。
- `description`：最终可交给媒体合成工具的完整指令，包含实际输出要求。

季节作为输出场景要求可作用于人物/物体背景；环境主体的有效季节属性也会同步。
取消季节规则只影响后续输出，已有媒体和版本保留。比例规则不会自动取消或恢复默认。

输出的 `rules.json`、`plan.json.rules` 和 `dialogue.json.rules` 保存规则时间线；
`dialogue.md` 附可读时间线。`stats.cross_turn_rules` 统计范围、保留/例外/替换和收尾取消数。
所有时间线、输出快照、一次例外恢复及最终状态都由 Python 校验；VLM 只负责台词表达。
普通 writer 不接收整段持续规则；verifier 和生成专家接收当前输出适用的状态。

人工复核用例：Tokyo seed 68、Alpine Kayak seed 26，均使用 `--reworks 2 --p-compile 0.5`。
它们覆盖 global/partial、多个规则重叠，以及保留、一次例外和永久替换三个决定。
旧 seed 产生的计划和消息位置已变化，旧输出目录不要用于新计划的 `--resume`。

## 采样与状态

准备模块分为两组：

| 单任务完成 | 跨任务完成 |
| --- | --- |
| 直接制作、多候选比较、先试再改、模糊请求澄清、相对描述预览 | 暂挂并恢复、结合另一资产试拍后回溯 |

五个单资产模块是 `direct`、`candidates`、`trial_edit`、`clarify`、`relative_preview`。
`task_switch` 处理第二个资产时，只在这五个兼容模块里抽取，不再嵌套另一个跨资产模块。
这七种是全部“准备模块”；镜头生成、返工、版本同步、拼接、编号澄清、约束生命周期和最终确认另行调度。

`trial_edit` 和 `relative_preview` 共用先出图再编辑的骨架，区别是监督目标：
前者明确否决并指定绝对改值，可在合法枚举中换值；后者要求“再浅一点/大一点”等相对变化，
只使用有顺序的属性和相邻等级，助手先给快速预览询问是否符合想法，随后暂用。

每个顶层准备模块结束后，其抽中的资产都已准备，再集中检查镜头依赖。镜头所需资产齐备，就安排
用户请求及镜头交付。用户可能在多轮资产探索之后才组合镜头，或先拼接已完成片段。
未指定拼接顺序时按生成顺序/当前拼接后追加；指定时由 Python 抽取排列。

“已准备”不等于最终定稿。预留返工的资产先生成不同初版，至少出镜一次后才提出修改。
资产明确时，assistant 在询问范围的同一轮附上新版参考图预览，并说明视频仍未更新；只问“当前提到的镜头”或“所有已经生成且使用它的镜头”。不询问未来镜头。
用户确认后，assistant 一次附上全部受影响的更新 clips；若已有拼接成片，同一回复也提供自动换好片段的成片，不要求用户逐镜头重复发指令。
只改单镜头时，其他现有镜头与未来新镜头默认保持旧资产；全改时，全部现有相关镜头更新，未来新镜头默认使用新资产。具体镜头的版本绑定仍可显式选择。
局部选择不会在收尾时被强制同步到其他镜头，最终可以保留用户接受的混合版本。

若剩余两项返工，合并为一条用户请求。只有指代已经明确时才生成预览；有指代歧义时先一次问清对象和范围，再在同一 assistant 交付中提供新版资产与全部更新 clips。
多个 edits 涉及同一镜头时只重做一次。返工后按照既有 shot_versions 重新拼接，未受影响的片段保持原文件。

`rework_plan` 记录提前选中的资产、字段与初版差异；返工事件的 `rendered_uses` 记录当时已经
生成的引用镜头。一个共享资产若当时只出现在一个已生成镜头里，只询问当前/后续的作用范围；
若已经有两个或更多引用镜头，助手具体指出其他镜头也使用同一版，询问是否同步。
两种分支都会发生：全部更新，或只更新当前镜头、其他保留旧版。局部修改后的后续同步也有明确
用户请求；结尾仍回到所有目标镜头兼容终态的状态。

可选片段进一步覆盖模糊指代和纠正、动态镜头编号、持久预览规则、单次例外、
规则恢复/覆盖，以及冲突取舍。采纳区分否决、暂用和明确采用。
拍摄冲突从 `conflicts.py` 选择五种显式互斥要求：完整全景与画面只剩特写、固定机位与摄影机环绕移动、
背景全部清晰与完全虚化、一镜到底不剪切与硬切、同一个输出画面同时要求9:16与16:9。
每种都绑定自己的冲突原因和澄清问题，`intent.conflict_type` 标记子类型。
另有新要求违反持久黑白预览限制的冲突，询问一次例外还是持续覆盖。
`behavior_coverage` 和每条 `intent.behavior_ids` 对应此前定义的 15 类用户行为。
单条轨迹只覆盖部分行为，多种 seed 覆盖完整集合。

对只有一个明确当前版本的“明确采纳”，独立随机流以 50% 概率省略用户采纳及助手确认两条消息，
直接更新选中状态；`selections` 记录隐式/显式采纳及发生位置。否决、暂用、回溯和多候选选择
仍需表达决定，避免省略实际必要的信息。多候选用语言指出第几个，不重复上传媒体。

所有准备模块和返工现在共用同一份属性枚举：先按视觉显著性加权抽样有合法替代值的字段，再从明显不同的合法值中
选另一个合法值，一次只改变一个字段。不会用字段候选值多少来隐式增加该字段被选中的概率。
`未知` / `不适用` 不作为被修改的原值，也不作为新值。无合法变化时，只能直接制作或澄清；
若用户限定的模块都不合法，回退到直接制作，不编造属性。

枚举中的相容性条件同时用于输入和采样校验，例如夜晚不用自然日光、透明物品不换成木材、
帆布不变成镜面表面。形状等字段有变化分组，避免直接把细长船体变成球体。它们是有限的
保守规则，不是完整物理/语义模拟器。其他未建模的功能约束仍需后续扩展。

相对预览只选择标记了 `ordered_values` 的字段，前版和目标值相邻，例如小号→中号、偏暗→适中。
程序根据方向生成“再大一点”“再亮一点”，而不让 Qwen 决定方向。物品变化明确是本体颜色明度，
不能改写为环境光。其他离散变化直接给出明确新值。若资产预留后续返工，本阶段目标仍可为临时版。

每个资产 placeholder 保存完整 `attributes`；编辑保存 `attribute_changes`（字段、旧值、新值）。
简短 description 从当前结构化版本渲染，包含当前变化字段；不会残留旧颜色或旧发型。
普通台词只发送当前版本的必要摘要，不把71字段的整份目录交给逐轮措辞模型。
校验不仅检查 `goal_match`，还会比对终态属性和编辑差异，拒绝标志正确但实际属性不一致的数据。

动态编号先抽一个实际的交换/后移操作，`intent.numbering_operation` 记录两个位置及操作类型。
若原始生成顺序与当前拼接对同一编号有不同解释，助手先问按哪套编号，用户确认当前编号后才执行。
新拼接记录 `previous_compilation` 和操作；校验会重放操作验证结果，不允许未澄清便生成。
若这一步尚未达到最终顺序，再补完整顺序调整。不会再用“挪回合适位置”代替实际操作。

每个 placeholder 记录来源资产、参考时间、输入媒体、编辑父节点及深度。
同一反向根之后至多连续编辑两次。镜头组合和纯拼接记录输入依赖，不视作同一资产
的连续视觉编辑。准备顺序、媒体编号和版本历史由程序管理，不交给 Qwen 决定。

## Qwen 的输入输出

人设与原有 tone examples 原文保留。首轮 plan 的 `intent.description` 在原要求前加上 `A fresh conversation starts.`；改写控制同时明确用户在发起请求，没有前文需要确认。改写请求每次发送：

1. `Speak as the User/Assistant with this tone`：固定人设原文；用户口吻示例单独放在 `Tone examples` 下，明确只模仿语气，不复用示例的对象、请求或决定。
2. `Previous messages`：需要承接前文的动作最多给最近三条消息。自足的新媒体请求、交付和确认省略无关历史；不额外附上媒体完整描述，也不向 rewriter 展示重复开头样本。
3. `Now, write a natural reply ... from the User/Assistant's point of view`：明确本轮说话者。
4. `Reply intention`：只给本轮应表达的事实、必要属性/范围/顺序和指代。动作控制与修正要求放在 SYSTEM，不混入内容清单。交付和确认不附带无关外观。
5. 最后一行始终是 `Return only the user's/assistant's spoken line.`，重试反馈也放在这行之前。

Rewriter 的 SYSTEM 通过显式 ChatML system 段发送，verifier 通过原生 `/api/chat` 的 system 消息发送。
Instruct 不添加 thinking 前缀。保留的 Thinking 改写接口使用闭合答案前缀，只说明已定角色和动作，不复制完整 draft。
Thinking verifier 使用原生 thinking，一次调用完成判断与报告；不再追加单独的报告生成调用。
CUDA 使用相同 SYSTEM 和输出协议；本机仅实测 Ollama，CUDA 仍需 NVIDIA 主机验证。

生成意图显式写为 `Request the assistant to make ...` / `Request the assistant to update ...`，明确创作由助手执行。
用户表达请求、修改或选择；可以在合适的情境使用 `Got it` 等承接语。是否发生角色倒置由 verifier 根据语义判断。
人设和示例原文保留，语气只控制表达，不能省掉草稿要求。用户请求根据当前内容要求润色；助手的交付/确认使用中性状态说明，避免完整台词模板和未交付的画面内容。澄清问题保留具体的候选和范围事实。草稿是模型输入，
不是失败后的模板回退；每条最终台词都必须得到一次真实 Qwen 回答。达到输出 token 上限时，会让 verifier 判断实际台词是否完整自然，而不是仅凭上限标记重试。

当前任务必要的资产绑定和纠正指代事实仍会提供；这不是重新追加历史对话。
不发送三条之前的对话正文、完整历史媒体目录、完整观察 JSON、未来轮次、隐藏的最终顺序或完整状态表。最近三条消息只供理解，当前意图决定
本轮说什么。指代澄清确认用户所指的对象，并询问要改什么；这不是采纳版本，也不擅自提出修改方向。用户纠正保留否定，不写成助手口吻。
Rewriter 只返回简短自然语言，不返回 JSON；啰嗦人设允许两三句。程序只为明确展示参考的澄清消息
附加“参考”行，为生成交付附加新输出占位符。确认、采纳、历史回溯和合成请求不重复附媒体。

语义检查由独立的 Qwen verifier 完成，与 rewriter 使用同一模型、不同的 SYSTEM，默认自动启用。
它评估本轮台词、当前内容要求、最近三条消息、当前绑定/重新展示的参考，以及同角色最近八条开头片段；首次创作还提供该主体实际的既往相关发言，避免虚构历史讨论。当前阶段没有实际生成媒体，
因此不向它伪造图片或视频输入。Verifier 输入使用分段自然语言，不再序列化嵌套的 context JSON：明确说话者、是否开场、动作、必需内容、对象、选中的版本及其属性、范围、参考与交付状态。采纳和回退另提供最近三条消息，旧版本的原始交付即便不在这个窗口内，明确绑定的版本仍是已知事实。多主体试拍的属性只绑定到对应主体；不把未来计划中的 alternative 当作当前事实。刻意模糊的轮次仍不暴露隐藏答案。角色设定与 tone examples 原文不变。旧消息的开头仅供 verifier 判断句式重复，不发送给 rewriter。

Verifier 的四项 rubric 及完整英文标准位于 `tools/dialogue_verifier.py`：

- `content_fidelity`：动作、内容与必要细节是否忠实于当前 draft 和 intent；接受自然同义改写。
- `grounding`：是否编造设计、历史版本、制作进度或修改范围。
- `conversation_coherence`：是否承接前文，保持说话者职责、指代和选择清晰；保留计划故意设置的歧义或冲突。
- `naturalness`：是否符合人设、流畅自然，避免明显重复；允许情境合理的 Got it、Yeah、Let's。

每项 0–2 分：2 表示完全满足当前任务，1 表示有可指出的小问题，0 表示实质性错误。
Verifier 在同一次调用中检查所有内容要求，只输出五个扁平字段：

```json
{"content_fidelity": 2, "grounding": 2, "conversation_coherence": 2, "naturalness": 2, "critique": ""}
```

四项全部为 2 才通过；有缺陷时 critique 说明具体问题和修正方向，交给 rewriter。通过时 critique 为空。
Python 只检查 JSON 结构和整数分数范围，不通过台词字符串匹配判断语义。
程序只将正向表述的修正要求交给 rewriter，最多尝试 5 个台词候选；不回灌被拒候选，不累积已经过时的诊断。
评分器没有返回结构完整的合法报告时，先对同一句台词重新评分，最多 3 次；合法且完整的 JSON 不会仅因 token 上限标记被拒绝。持续失败会明确报错，不静默放行。

Python 不再通过属性关键词、候选编号措辞、角色用语、镜头词汇或范围短语来判断台词语义。
只保留非空、禁止 JSON/代码块/控制标记、模型不能自行插入占位符等明确的输出协议检查。行数、字数、英文自然度、是否完整都由 verifier 判断；token 截断标记会交给 verifier 作为证据，不再仅凭标记拒绝一条实际上完整的台词。
`validate_plan` 仍检查结构化计划的引用存在性、占位符、资产绑定、镜头顺序、reference graph 等确定事实。

Draft 以内容要求而非示例台词提供：确认/交付使用中性的制作状态与决定说明；选择绑定从规划的媒体引用确定。
不再把完整 draft 重复到 assistant prefill。改写和评分都不会看到未来对话或完整目标状态。

`qwen/requests.jsonl` 保存 rewriter 实际请求（含 SYSTEM），`qwen/calls.jsonl` 保存候选与对应的 verification。
`qwen/verifier/requests.jsonl` 保存独立评分请求，`calls.jsonl` 保存原始报告、解析结果和 critique；
改写及评分各自的 `responses.jsonl` 保留 Ollama 原始响应和耗时/token 元数据，均不作为训练对话。
默认 Instruct 评分温度 0.1，输出上限 512 tokens；Thinking 保留 4096 tokens 总预算。没有二次报告生成。
只有报告 JSON 格式损坏时才重试同一句的评分；合法低分会让 rewriter 修改台词，完整日志可追溯。
已有完整 checkpoint 不会重新评分；如需整段使用新提示和评分，请使用新的输出目录。

进度里的总数是用户和助手消息的总条数。Python在调用Qwen前完整规划，保存到`plan.json`；
Qwen只改写这条已确定的路径。相同代码、输入、规划参数和seed产生相同路径与条数；
不同seed通常改变模块、对象、版本、采纳、返工和拼接路径，条数可能不同，也可能相同。
措辞重试不增加计划消息，实际模型调用次数可能超过消息数。
格式失败和 verifier 未满分都在现有的五次改写额度内重试；最终失败保留检查点、最后的具体 critique 和原始调用记录，
不会静默用固定模板替代模型回答。

## 输出

```text
output/
  observation.json    原始虚拟/真实观察输入
  plan.json           确定的动作轨迹，content 尚为空
  dialogue.json       每条自然对话、底层意图、媒体引用和最终状态
  dialogue.md         可读对话及完整占位符列表
  placeholders.json   精简媒体清单：description / prompt / optional references / 深度 / 操作参数
  generation_plan.json 完整专家执行合约及构建顺序
  qwen/calls.jsonl    台词候选、格式检查和 verifier 评分
  qwen/requests.jsonl Ollama 改写请求，含完整 SYSTEM 和 ChatML
  qwen/verifier/     评分器原始请求、四项评分和 critique
```

`dialogue.json.turns[]` 中的 `intent` 记录操作、模块、用户行为标签、目标资产/镜头、
采纳态度或更新范围。媒体字段为：

- `input_media`：生成交付的实际正向依赖，或澄清时重新展示的既有媒体。
- `output_media`：本轮新生成的媒体占位符。

`intent.display_media` 指定追加到正文的历史附件；`intent.context_media` 保留非附件的历史语义指针。镜头更新的 `intent.reference_updates` 记录每个身份的 before/after 版本。

原视频是剧本和身份设计 guide；专家合同仅引用对话中更早生成的媒体。首次资产文生图/视频，后续自然编辑和组装，最终镜头不复制原视频。实际一致性过滤属于后续管线。`goal_match` 仍是计划状态标签，不是对实际媒体的视觉评分。

`generation_requirements` 仅记录明确的 `composition` / `pose`，修改继承它。人物图采用自然角色呈现，物品图采用独立产品照，环境为连贯场景。

完整正向依赖、视频裁剪、模型生命周期和运行方式见 [reference_materials.md](reference_materials.md)。

版本选择和回退明确写为选择动作；只有纠正错误指代时才使用 “I meant”。连续讨论同一对象的变体请求使用 `the same ...`；切换过对象后，使用 `the previous ...` 重新点名。每轮 `intent.reference_policy` 标记清晰指代、刻意模糊物体、刻意模糊镜头编号或逐项修改的策略。
Verifier 同时收到原始 `intent.description` 和自然语言指代要求。没有计划歧义时，严格根据可见会话检查听话人能否识别对象，不能用 planner-only 的资产绑定替台词补全；计划了歧义时，仅对相应维度放宽，批量修改逐项采用 `edits[].ambiguous`。明确代词指代时不强迫重复名词。`wood texture design` 可以作为 wood 变体的普通视觉表达，不要求材料工程式措辞。
`long shot` 在台词草稿中注明 `(wide shot)`，模型优先用 `wide shot`，不改为延长时长。
单一用途资产的返工明确采用新版本继续使用；存在多个用途时，先询问局部/全局范围再执行。

## 验证

```bash
python -m unittest discover -s tests -p 'test_dialogue*.py' -v
```

检查 200 个种子的终态收束、全部 15 类行为的总体覆盖、1–4 镜头、局部更新范围、
不重做无关镜头、拼接版本新鲜度、媒体因果关系、编辑深度、媒体展示规则、隐式采纳、
相对变化类型、固定人设、最近三条消息上下文、排除更早媒体目录、不泄露未来目标，以及实际制作媒体的正向因果依赖、无环性和深度。
测试中的措辞替身只用于离线回归；实际 CLI 默认真实调用 Ollama。

## Stats 与行为名称

完整的 15 项名称以 `planner.py:BEHAVIORS` 为程序唯一来源；此前 Markdown 示例引用过部分名称，但没有完整名称表。
`dialogue.json.stats.user_behaviors_covered` 直接复用下表 exact 名称，按 ID 排序，不另作翻译。

| ID | Exact behavior name |
| --- | --- |
| 1 | Version rollback |
| 2 | Goal clarification |
| 3 | Ambiguous reference |
| 4 | Multiple candidates in one turn |
| 5 | Separate preparation and assembly |
| 6 | Trial shots and assembly |
| 7 | Revision scope clarification |
| 8 | Reference correction |
| 9 | Task suspension and resumption |
| 10 | Multiple-reference composition |
| 11 | Context-dependent shot numbers |
| 12 | Relative edits and previews |
| 13 | Persistent rules and exceptions |
| 14 | Conflicting requirements |
| 15 | Adoption decisions |

`stats` 随每个检查点更新；失败只统计已完成消息的 coverage，不把未实现的计划当作已覆盖。
`number_of_turns` 是完成的消息数（双方合计），`number_of_attempts` 是实际改写尝试数，包含失败候选；
`number_of_assets` 是已交付媒体 placeholder 数（图片、视频及版本各计一个），`number_of_logical_assets` 是其中不同对象/环境的数量。
`planned_number_of_turns/assets` 是计划总量。`time_elapsed_seconds` 是累计实际运行时长，续跑累加但不包含两次运行间的等待时间。
`generation.verifier_attempts` 单独计算评分尝试，`model_requests` 是总模型请求数；Ollama 同时记录 prompt/completion tokens、加载/预填充/解码时间。
`generation` 还记录接受数、格式拒绝数、语义拒绝数和评分器故障数；local 没有 token 遥测时注明 unavailable。

其余覆盖统计包含：用户各行为的消息数与覆盖比例、角色/动作/模块分布、资产类型、媒体类型/用途、目标镜头与镜头版本数、
试拍次数、修改属性、返工范围、冲突类型、显式/隐式采纳、回溯次数、构造参考深度分布和最大深度、各对象版本数、人设和规划 seed。
这些都是结构化计划标签的覆盖，不能当作独立的语义正确率或自然度人工评估。

改写输入按动作裁剪：交付只提供媒介类型与交付状态；编辑只提供对象、目标变化及必要范围；拼接明确列出现有视频 clip 的顺序。Verifier 另用清楚的自然语言列出本轮必要的对象、版本和参考关系，不发送无关的完整媒体描述或未来监督状态。

## 实际媒体生成：materialize

在 GPU interactive session 中运行。输入为一个已经完成对话及 generation prompt
rewrite 的 run dir；执行依据是其中的 placeholders.json，而不是对话时间线的 inputs。
需要同目录的 dialogue.md，以及涉及 shot 裁剪时的 observation.json。FLUX.2-klein-4B
和原版 MiniMax-H3 Base-Ref2VA FP8 / Turbo v4-600 EMA 的环境、权重须已安装；执行器使用已有离线 runtime，不进行自动安装。
开跑前会检查合同指定的模型缓存，缺失会直接报告模型名称；FLUX 加载固定到已检查的缓存 revision。
新合同默认使用 Klein 4B（4 steps，guidance 1.0），也尊重旧合同显式指定的 model_id。

在 vLLM 还运行时，可先按原视频资产的语言描述精确选择 target reference frames：

    python scripts/dialogue_generation/materialize.py "$RUN" --prepare-frames

此步骤调用 tools/find_video_frame.py，保存 frame_selections.json、选中的 PNG 和请求记录。
它只检索 target video 中实际存在的原始身份；修改版本的外观仍由后面的 FLUX 编辑实现。
选帧结果按原视频哈希和 query 校验，可重复运行并复用已有结果。如果不运行此准备步骤，
执行器使用 placeholders.json 已记录的 observation frame_seconds。中间 asset reference
clip 始终按合同中的 frame_seconds（当前为 0 秒）确定性抽帧，无需 VLM。

随后在同一 GPU 节点关闭 vLLM，并运行：

    bash scripts/vllm/stop.sh
    python scripts/dialogue_generation/materialize.py "$RUN" --resume

使用 cocreation conda 环境运行，确保其 bin 目录在 PATH 中以提供 ffmpeg/ffprobe。
materialize 自动按需加载 Flux/MiniMaxH3，默认 7 steps；保持 topological ordering，并尽量连续执行同一模型的就绪任务，切换时释放前一个 worker。默认 H3 按合同请求 4 秒镜头，接受 4–15 秒请求。可确定执行的复制、抽帧和拼接会立即执行，以解锁当前模型的后续任务。旧 VDN 合同仍可显式执行。
普通执行阶段不会调用 vLLM、改写 prompt 或重新规划。--dry-run 仅校验输入并打印构建顺序。
--seed 控制每个 placeholder 的稳定生成 seed。--source-video 可显式指定搬迁后的真实源文件；
fictional:// 虚拟示例没有可执行的真实媒体，必须先提供内容对应的实际 target video。

输出全部位于输入 run dir：

    media/image_NNNN.png
    media/video_NNNN.mp4
    media_manifest.json
    dialogue.md
    dialogue.placeholders.md
    _inputs/target_frames/       # 可选的前阶段语言选帧结果
    generation_records/          # 每次生成的参数、耗时和精简运行记录
    _work/materialize/           # 临时文件；每个资产完成后清理，全部成功后删除 _work

target video 按原字节复制，保留原始容器扩展名；copy 也保留来源扩展名。其他操作调用
确定性的裁剪、抽帧和拼接工具，或向相应模型提交保存的 prompt 与严格有序的参考槽位。
shot 边界按 observation 在源视频中细化。VDN 输入不足一秒的参考会明确延长为末帧停留，
生成结果再裁至实际镜头时长；生成视频适配 target 分辨率以便和原始 shot 拼接。
视觉 video_edit 复用源参考视频音轨，reference_generation 保留模型生成的音轨。

每个节点写入状态、文件 SHA256、输入指纹、seed 和生成参数；--resume 检查文件及输入，
跳过仍有效的节点，重建缺失、损坏或输入改变的输出。不覆盖未被 manifest 管理的同名文件。
失败保留已完成节点，并保持 dialogue.md 原样；所有节点完成后才把其中的占位符替换成
相对路径的 HTML img / video 标签（video 带 controls），同时首次保存 dialogue.placeholders.md。
dialogue.json 的 media_generated 随全部成功变为 true；对话正文与 placeholders.json 合同保持不变。


选帧检索使用和 video observer 相同的实际图片帧读取方式与 768px / JPEG quality 90
预处理。优先检查 observation.assets[].frames，随后以默认 4 FPS、至少 32 个样本覆盖全片，
每次最多提交 16 张图片。粗采样全部未命中时，默认继续分批检查剩余原始帧；局部细化未命中
不会丢弃已确认的候选。导出的 selected.png 始终是原始分辨率帧。此流程不受 vLLM 原生 MP4
输入的 1 FPS / 16 帧上限影响。检索结果记录实际检查的帧索引、时间、批次和覆盖数量。


### Forward asset appearance policy

Only object color/material, person clothing color/age group, and environment season/time of day/weather are sampled. Shape, surface finish, texture, garment type/fit, hair and all other observed fields remain unchanged. The baseline candidate pools are in `asset_variations.py`; accepted per-identity values in `attribute_extensions` join those pools for the already enabled fields. Unknown/not applicable baselines remain ineligible. Other observed fields, including custom hair or garment-type values, remain fixed and are preserved in planning and generation. Legacy observation values remain readable. Canonical aliases and unordered color combinations are compared before sampling, so a proposed value cannot repeat the current version. Explicit rollback to an older version remains supported.

Final acceptance preserves active user constraints and the current generated designs. There is no automatic cancellation, asset restoration or shot regeneration to match the guide. Explicit rule conflicts/cancellations, combined edit requests, and deterministic final shot reordering remain available.

Isolated object and character images keep a minimal neutral studio setup, including local edits. Seasons add sparse dressing: spring blossoms/leaves, one small lush tree and sunlight for summer, fallen leaves around the base for autumn, snow around the base and on the subject for winter. Environment assets retain their actual setting. The current image presentation contract is always passed to the rewriter.

`run_pipeline.sh` uses classic 8+3 two-pass H3 (`--video-steps 8 --video-two-pass`). The standalone materialize API retains its single-pass default unless these options are supplied. By default videos generate at 1024×576 and are delivered at 1280×720; reference images are 1024×1024. Each new synthetic clip requests four seconds; H3's 17n+5 frame grid samples 107 frames (4.4583 s at 24 fps), then the delivered clip is trimmed to exactly four seconds. Guide shot lengths no longer lengthen synthetic clips. `--video-steps`, `--video-width`, `--video-height`, `--output-width`, and `--output-height` can override the defaults.

The expert receives only an existing clip and the revised identities needed for that edit. Full source binding tables stay in planning metadata. Revisions after concatenation reuse individual clip files, render only affected clips, then concatenate them back in order. If an assembled output needs a new appearance treatment, `clip_edits` holds independent short expert requests; the whole cut is never submitted to H3.

Prompt rewriting explicitly analyzes conflicts among original captions, current attributes, references and requested edits. `attribute_overrides` is authoritative for each current subject. Source descriptions are kept as lower-priority evidence; incompatible adjectives/clauses must be omitted from the final prompt. Age-only edits explicitly preserve hair and garment properties. Related subjects (for example a carried bag) use their own current version attributes. Names are neutralized deterministically before planning.

Black-and-white output rules, monochrome-specific sampling/rewriting, hue-only deferral and deterministic desaturation have been removed. Black/white remain ordinary object/clothing colors.

`media_manifest.json` records each asset's `generation_seconds` (actual model call) and `elapsed_seconds` (complete asset operation). Compact receipts remain under `generation_records/`. Successful request files are removed after the final media is saved. Failed work is retained for diagnosis/resume; once the run succeeds, all `_work` is removed after model workers close. The run lock lives outside `_work`.


### Persistent format rules

`effect.aspect_ratio` accepts `portrait` (all media 9:16), `landscape` (all media 16:9),
`square_assets_landscape_video` (asset images 1:1, videos 16:9), or
`square_assets_portrait_video` (asset images 1:1, videos 9:16).
A user sets a rule once. Format supports explicit setting, keep, one-request exception and permanent replacement. It cannot be canceled. No explicit requirement means square 1:1 images and landscape 16:9 videos, without an opening format dialogue.
Snapshots preserve the conditions of already produced media. Each generation task resolves its own ratio;
command-line dimensions are fallback settings. Images use exact 1024×1024, 1024×576 or 576×1024 canvases;
videos generate at those model-compatible dimensions and finish at 1280×720 or 720×1280.
Assemblies normalize differently sized inputs with padding so existing shot content is preserved.
Format-only changes do not invoke generative video editing.

Launch settings can configure the output canvas. Explicit observation constraints still supply intentional project rules. The planner continues to sample random rules, including at the opening. Older batch manifests may contain the retired format-rule field, which is ignored for fresh dialogue generation.


The pipeline explicitly disables Qwen thinking for observation, dialogue/review and expert prompt rewriting for the next experiment. Optional thinking mode remains available with a bounded reasoning budget.
Observation allows 8192 output tokens; dialogue/review allows 4096; expert rewriting allows 4096 (FLUX) / 6144 (H3).
The vLLM reasoning parser keeps reasoning separate from final dialogue/JSON. Per-stage request counts and latency are recorded in `generation.stages`.


Independent image/video requests are composed by the prompt writer using only the supplied subject set, references and current attributes. They do not receive target-shot action prose. Formal `generate_clip` requests continue to use their declared shot description after all required assets are ready. H3 examples contain no target-story characters or settings.

Aspect ratio is executor metadata. Python resolves each output to `width`/`height`, which become FreeVideo's `--width`/`--height`; generation prose does not set the canvas. The writer may use the orientation for composition, but must not repeat numeric ratios or sampler settings in the generation prompt.

With the current video canvases and 2x latent upscale, 8+3 is 512×288 → 1024×576 for landscape, 288×512 → 576×1024 for portrait, or 512×512 → 1024×1024 for square. The final 1280×720/720×1280 export is a separate resize. These stage sizes were checked against the installed FreeVideo planner, not estimated from the prompt.
