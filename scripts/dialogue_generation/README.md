# 从目标视频观察构造多轮对话

输入是视频观察 JSON；`generate.py` 不读取视频，也不提取帧。
上游工具 `tools/video_observer.py` 负责从视频产生这个 JSON。
Python 决定动作、对象、候选数量、采纳、回溯、任务切换、返工范围和拼接顺序。
Qwen 将每条已确定的意图表达为自然英文，再由同一模型使用独立 SYSTEM 验证；失败时重写。
不生成图片或视频；媒体全部是占位符。

## 运行

默认使用 **Qwen3-VL-30B-A3B-Instruct**，不生成 thinking。两阶段共用 Ollama 已加载的同一模型权重，分别调用并使用不同 SYSTEM。
模型别名与完整名称见 `tools/dialogue_models.py`：

| 仓库别名 | Ollama 模型 | local checkpoint |
| --- | --- | --- |
| `qwen3-vl-30b-instruct`（默认） | `qwen3-vl:30b-a3b-instruct-q4_K_M` | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `qwen3-vl-30b-thinking` | `qwen3-vl:30b-a3b-thinking-q4_K_M` | `Qwen/Qwen3-VL-30B-A3B-Thinking` |

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
  终态外观完全由 `attributes` 给出。`name` 应是不含版本颜色、材质等的短名称。
- `attributes` 必填该类别的全部字段，每个字段只取一个枚举值。不接受未知字段或自由文本值。
  看不清/无法推断选 `未知`，没有意义选 `不适用`；这两种值不参与变化采样，不从描述猜补。
- 镜头 `description` 建议写动作、构图、环境关系，可变资产外观由 `assets` 的参考决定。
- `target_order` 可省略，默认按 `shots` 列表顺序。
- `constraints` 可选，例如 `[{"text":"正式镜头都在东京夜晚", "scope":"shots"}]`。
  `scope` 为 `shots` 或 `all`；上游应只提供与目标视频兼容的规则。
  需要机器保证属性不变时，显式补充 `asset_attributes`，如 `{"street":{"time_of_day":"夜晚"}}`。
  `shots` 限制进入正式镜头的版本，探索图可以临时变化；`all` 同时限制所有资产探索。
  程序不会把任意自然语言限制自动解析成属性锁。
- `video` 是来源标识，不要求文件存在；额外 VLM 字段会保留/忽略，不触发推理。
- 每个管理资产必须至少被一个目标镜头使用。不需要把背景中的所有小物体拆出来。
- 初版假设每个资产只有一个终态外观；若目标内同一个包本来就有不同设计，
  先以不同资产 ID 表示，避免将原片合法差异错误当成返工。

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

“已准备”不等于最终定稿。预留返工的资产先生成与终态不同的初版，至少出镜一次后
才能返工到目标版本。共享资产先询问同步范围。若只改一个镜头，就固定该镜头的新
版本，其他镜头继续使用旧版；随后有一条预留的同步事件完成收束。
返工仅更新绑定发生变化的镜头，已有拼接显式替换相应片段，保持旧视频历史不变。
结尾完成剩余返工、同步、拼接和顺序调整；单镜头无需拼接。

返工触发时机不是“首个镜头一出就立即返工”：

1. 开始前从有已知、可编辑属性的资产抽取 `--reworks` 个，优先物品，提前构造偏离终态的初版。
2. 每个准备模块结束后，先生成所有依赖已齐备的镜头，并按概率拼接。
3. 此时若存在预留返工且已出镜的资产，以 60% 概率抽其中一个返工；未触发则继续准备其他资产。
4. 主循环结束后，剩余返工必须执行，以保证回到目标视频。若剩余两项，合并成同一条用户请求。

集中返工使用 `batch_revision`，依次为 `request_reworks`、可选的 `clarify_reworks` 和
`resolve_reworks`、`deliver_reworked_assets`。`intent.edits` 逐项记录历史目标、目标镜头及属性改值。
每项独立采样是否模糊指代、只改本镜头还是全部同步；助手在同一条回复里问完所有指代和范围问题，
用户在同一条回复里分别回答，不能漏问、提前生成或串换范围。
澄清可集中展示历史候选，但用户不重新上传参考。模糊用户请求不会被额外提供隐藏的真实对象绑定。
确认全部问题后，一次交付各项新参考，再更新受影响镜头；两个修改若影响同一镜头，只更新一次。
若结尾还有多个局部版本待同步，也合并为一次用户确认，再统一更新。只有一项未完成时沿用单资产返工。

例如高山湖输入 `--seed 1 --reworks 2 --p-compile 0.5` 会产生两项都含模糊指代的集中返工。

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

所有准备模块和返工现在共用同一份属性枚举：先等概率选一个有合法替代值的已知字段，再等概率
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
  placeholders.json  placeholder / kind / description / reference_materials / reference_depth 列表
  qwen/calls.jsonl    台词候选、格式检查和 verifier 评分
  qwen/requests.jsonl Ollama 改写请求，含完整 SYSTEM 和 ChatML
  qwen/verifier/     评分器原始请求、四项评分和 critique
```

`dialogue.json.turns[]` 中的 `intent` 记录操作、模块、用户行为标签、目标资产/镜头、
采纳态度或更新范围。schema version 3 区分三种媒体字段：

- `input_media`：本条在对话时间线中看起来采用的已有媒体，只记录生成交付消息的依赖；不代表聊天附件，不追加到正文。离线实际制作使用 `reference_materials`。
- `reference_media`：助手本轮重新展示的历史参考，例如“是这张吗”；程序将其追加到正文。
- `output_media`：本条新生成的媒体占位符，追加到正文。

其他历史指向保存在 `intent.context_media`，供后续执行和评测，不代表用户二次上传。
镜头更新还记录 `intent.reference_updates`（逐资产的 before/after 引用），草稿直接点名修订的是哪个主体，
避免连续更新时套用上一镜头，或把一件物体误说成另一件物体的替代品。该字段仍属于对话时间线，
不是实际制作时的 `reference_materials`。
`content` 是 Qwen 生成的文字及按上述规则附加的占位符。对话和计划 JSON 由 Python 写入；独立 verifier 的评分使用 JSON 输出协议。
模糊指代的真实目标单独保存在 `intent.target_media` 监督标签中，不附到用户台词上。
训练/评测时，用户的 `intent` 是标注信息，不应作为主 agent 可见的输入。

目标视频在这里提供的是合成监督和终态约束。`goal_match` 是程序构造标签，
不是模型对实际媒体的视觉验证结果；`media_generated` 始终为 false。
本脚本不声称这些 placeholder 已被真实图像/视频生成器实现。

实际制作媒体时，使用 `reference_materials`，不能把对话时间线中的 `inputs` 当成制作参考。
完整分类、反向版本派生和深度定义见 [reference_materials.md](reference_materials.md)。
这些字段由 Python 在完整计划形成后确定，保存在 `plan.json`、`dialogue.json` 和 `placeholders.json`；
不会作为额外历史或未来目标传给台词改写模型。
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
相对变化类型、固定人设、最近三条消息上下文、排除更早媒体目录、不泄露未来目标，以及实际制作参考图的反向依赖、无环性和深度。
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
