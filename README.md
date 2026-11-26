# cocreation：视觉共创 Agent 的数据构造管线

我们在做一个**视觉共创（visual co-creation）agent**。它在多轮对话里和用户一起做视频：先理解用户想要什么，再记住用户定下的跨轮规则（"以后都用冬天背景"、"视频一律竖屏"）。它会从对话历史中找出该引用的视觉资产（之前生成并被用户采纳的人物图、道具图、已有镜头），然后用 tool call 调用 **MiniMax-H3-Image**（图片）和 **MiniMax-H3**（音视频）生成或编辑资产，最后把镜头拼成成片。

第一步是**大规模自动构造训练数据**。本仓库的主体 `scripts/dialogue_generation/` 就是这条数据管线：输入一条视频描述文本，输出一段完整、可训练的 agent 轨迹。轨迹里有用户的自然语言请求、assistant 的思考、真实执行过的生成调用、真实生成的图片和视频、assistant 对结果的自我审查，以及审查不通过时的重试过程。

> 文档以代码为准。本 README 讲高层设计、管线构造、用户模型和各处的设计考量。字段级契约、对象设计和维护指南放在 [`docs/`](#文档索引)。

---

## 一条训练样本长什么样

最终产物是每个 case 的 `dialogue/sequence.json`（人类可读版本是同目录的 `dialogue.md`）。它由交替的 user / assistant 轮次组成，assistant 轮次被拆成有类型的 segment：

```text
User:      Can you make a reference image of the tote bag? Wide shoulder strap, purple canvas, matte.
Assistant: [text]      Sure, I'll make a still reference of the tote bag first.
           [thinking]  <think>Let's analyze what the user is asking for. ...
                       Now let's look at which cross-turn rules are effective. ...
                       Now let's gather visual references for this generation. ...
                       Now, I'll make a tool call to generate the image.</think>
           [tool_call] <tool_call>{"name": "minimax-h3-image",
                         "arguments": {"references": [], "prompt": "...",
                                       "parameters": {"ratio": "1:1", "res": "4096p"}}}</tool_call>
           [media]     [[image_0002]] -> media/image_0002.png
           [thinking]  <think>（对生成结果的第一人称审查：是否满足请求、规则、参考一致性）</think>
           [text]      Here's the tote bag reference.
User:      That works. ...
```

审查不通过时，序列里会出现被拒绝的结果、拒绝理由和修复后的再次调用（`[[image_0002_01]]` 等版本号）。所以样本本身就包含了自我纠错行为。

---

## 管线总览

```text
 文本描述（H3 结构化 prompt / OpenVid caption）
        │  tools/prompt_observer.py           LLM 抽取，无任何媒体                    ◀ 反向：确定终点
        ▼
 observation.json   ── 资产（人物/物体/室内/室外）+ 结构化属性 + 1–4 个镜头 + 目标顺序
        │  planner.py + cross_turn_rule.py    纯 Python，seeded，不调用模型          ◀ 反向：从终点倒推剧本
        ▼
 plan.json          ── 每一轮的对话行为、意图、媒体占位符、版本/引用 DAG、规则时间线
        │  tools/dialogue_realizer.py         LLM writer + LLM verifier，逐轮写台词
        ▼
 dialogue.json      ── 自然语言对话（媒体仍是占位符）
        │  rewrite_prompts.py                 LLM：生成合约 → H3 prompt → 思考链       ▶ 正向：prompt 先于媒体
        ▼
 generation_plan.json / placeholders.json ── 每个占位符的可执行生成任务
        │  materialize.py + sglang_media.py   H3-Image / H3 真实生成 + VLM 审查 + 修复重试 ▶ 正向：按对话顺序真实执行
        ▼
 media/ + media_manifest.json
        │  formal_sequence.py                 只发布真实发生的事情
        ▼
 sequence.json / dialogue.md   ── 正式训练样本
```

materialize 之前的步骤（observer → planner → realizer → prompt rewriting）都是纯文本工作，统称 **VLM 阶段**。materialize 是 **media 阶段**，在 GPU 上真实生成。生产环境里两个阶段在同一个单卡 Slurm 作业中流水线并行，见[运行方式](#运行方式)。

---

## 为什么这样构造：倒着写剧本，顺着拍

### 纯正向构造的两个问题

最直接的做法是**纯正向**：让 VLM 扮演用户，自由地提需求；让另一个 VLM 扮演 assistant，自由地回应和生成。我们没有这么做，因为它有两个结构性的问题：

1. **多样性瓶颈。** VLM 缺乏想象力。让它从零"想一个视频项目"，它会反复收敛到少数几种常见题材、构图和人物。数据量越大，重复越严重，多样性会成为最主要的瓶颈。
2. **不可控。** 对话完全由 VLM 自由发挥时，走向无法控制：可能原地打转，可能早早结束，也可能偏离到与视频创作无关的话题。我们建模的 15 类用户行为（见[用户模型](#用户模型)）也很难保证真的出现在数据里，更无法控制它们的比例。

### 解法：先用 Python 规划，再转写成自然对话

- **创意来自真实语料。** 每条数据的"故事"来自真实的视频描述文本（H3 数据集的结构化 prompt、OpenVid caption）。observer 把它拆解成资产、属性和镜头，作为这条对话要抵达的终点。题材的多样性等于语料的多样性，不依赖 VLM 自己去想。
- **结构由 Python 决定。** Planner 是一个 seeded 状态机，决定每一轮做什么：哪种行为、涉及哪个资产、产出什么媒体、引用哪些旧版本、何时设立或冲突一条规则。15 类行为由代码显式调度，每条样本的覆盖情况都会统计到 `stats` 里。
- **LLM 只负责措辞。** Writer 把每一轮"已经决定好的意思"写成自然口语，verifier 检查有没有写错。这样既有自然语言的多样性，又不会改变对话结构。

### 混合结构："反向定剧本，正向拍摄"

整条管线是**反向**和**正向**的混合体。我们称之为 **plan backward, render forward（倒着写剧本，顺着拍）**：剧本从终点倒推出来，拍摄则严格按时间顺序、用真实工具一步步执行。

| | 反向（从终点 / 从结果倒推） | 正向（按因果顺序真实执行） |
|---|---|---|
| 做什么 | observation 确定终点；planner 从终点倒推出一条合理路径（包括故意的弯路，如预留返工）；reasoner 从 prompt 和对话历史倒推思考链 | 每个媒体都由对话中的 prompt 经真实 tool call 生成；只能引用已经生成的媒体；审查的是真实生成结果；导出只发布真实发生的事情 |
| 带来什么 | 给序列注入创意；让用户朝着一个设定好的目标前进，不会随机原地打转；行为可控、可统计 | prompt 和生成出来的资产天然对齐 |

**源输入只是语义和创意的引导。** 输入的 caption 或源视频只用来给 observation 提供题材、资产和镜头设定。终点是 planner 层面的结构化目标：采纳哪些资产版本、拍哪些镜头、按什么顺序拼。我们**不要求**最终生成的视频在画面上还原源视频或 caption。生成的画面只要和这段对话自洽（每个资产、镜头、修改都和对话里说的一致），就达到了目的。所以 observer 不需要看到真实画面（`video_verified: false`），源视频也从不作为生成输入。

**为什么正向部分不可省略。** 也可以"先有画面、再配 prompt"：从目标视频反向编辑出各个中间态资产，再去想象什么样的 prompt 能生成这些中间态。但这样写出的 prompt 只是"看起来合理"，和工具的实际生成行为很可能对不上：同一个 prompt 交给 H3，未必能生成那张图。agent 学到的就会是一些名不副实的 prompt。正向执行保证每个资产都**正是**工具根据那条 prompt 生成的结果；生成不理想时，assistant 必须审查、修复、重试，而这个过程本身也成为训练数据。

**为什么思考链是反向构造的。** 在一次生成调用里，prompt 最重要，而 prompt 要做什么在 planner 阶段就已经基本确定了。如果让 reasoner 先自由思考、再根据思考写 prompt，它很容易节外生枝：想出计划里没有的细节，或者与最终调用对不上。所以我们先确定 prompt，再让 reasoner 拿着 prompt 和真实的对话历史，以"还没写 prompt"的第一人称视角，倒推出一段思考（分析请求 → 判断生效规则 → 解释为什么需要这些参考）。这样思考与调用严格一致，reasoner 也不会改变已经定好的行为。

### 这样设计的好处

- **多样性有多个独立来源。** 语料 × 规划 seed（模块、资产、规则、返工）× 5 种用户人设 × 独立的措辞 seed × 真实生成的随机性。任何一个维度都可以单独控制或复现。
- **分布可控、可度量。** 每种行为、模块、规则决策、编辑类型都有显式的概率和开关，每条样本的覆盖统计都写入 `dialogue.json.stats`，可以按需重新配比。
- **正确性由代码保证。** 结构约束都由 Python 重放校验（`validate_plan`、`validate_rule_timeline`、`validate_reference_materials`），例如：不能引用未来的媒体、澄清解决前不能出图、拼接不能使用过期镜头、规则快照必须一致。不依赖 LLM 当裁判，所以不会有"裁判也看错了"的问题。
- **思考、调用、结果三者一致。** prompt 先于媒体，思考对齐 prompt，审查针对真实结果。
- **自带纠错数据。** 生成失败时的审查、拒绝、修复重试都按真实发生的顺序进入样本。
- **成本低。** LLM 只负责写台词、写 prompt、写思考、做审查；规划完全不消耗模型调用。
- **能容错、能续跑。** 单个媒体彻底失败时只剪掉依赖它的那段交互（case 记为 `partial`），不会浪费整条样本。每个阶段都有 fingerprint 缓存和原子写入。

---

## 用户模型

"用户模型"指 planner 里对**一个合成用户**的完整建模：他心里想要什么、怎么说话、记得什么、会做出哪些行为。以下内容以 `planner.py`、`cross_turn_rule.py`、`tools/dialogue_personas.py` 为准。

### 建模了什么

| 维度 | 建模方式 |
|---|---|
| **隐藏目标** | 用户心里有一个目标视频（observation：每个资产的目标属性、每个镜头、最终顺序），但从不直接说出来，也从不上传素材；只能通过请求让 assistant 一步步做出来。对话结束时，planner 层面的状态（采纳版本的属性、镜头、最终顺序）必须到达这个目标；生成的画面只需与对话自洽，不需要还原源输入 |
| **会改主意** | 预留返工：部分资产（`--reworks`，生产默认按 seed 随机取 0 或 1 个，优先物体）的初版故意偏离目标一个属性。等它被用进镜头后，用户再要求改回来 |
| **会试错、会犹豫** | 试稿、候选对比、临时采纳、回退等，见下面的行为表 |
| **持续偏好** | 跨轮规则：季节、时段、画幅、某个资产的某个属性。用户可以设立、取消、在冲突时保留、只这次例外、以后替换 |
| **说话风格** | 5 种人设之一（casual、cute、precise、verbose、concise），按 seed 抽一次，整段对话不变；assistant 的人设固定 |
| **记忆与指代** | 用户会用"之前那个"、"第 2 个镜头"之类的指代，可能有歧义，也可能说错，需要 assistant 澄清 |
| **决策权** | 用户决定采纳、拒绝、选哪个、作用范围、规则冲突的处理方式；assistant 交付、确认、提问，不替用户做决定 |

主要概率（都在代码里，可调）：采纳中 50% 是隐式的（不额外说 "用这个"）；`direct` 模块 30% 临时采纳；返工前 70% 插入一段指代歧义；每段对话最多一次单镜头冲突需求（30%）；资产进镜头后 60% 立即返工；每个新镜头后 `--p-compile`（0.65）概率要求拼接预览；规则层每个检查点分别有 20% 设立、20% 取消、30% 挑战一条规则。

### 15 种行为

下面每种行为都附了例子。台词是按 planner 的实际结构写的示意（真实措辞由 writer 按人设生成），场景取自示例 `examples/tokyo_vlog.json`（旅行者、托特包、咖啡杯、咖啡店、东京街道）。`U` 是用户，`A` 是 assistant。

**1. 版本回退（Version rollback）**：试过新版本后，主动回到旧版本；或在返工时发现想要的设计以前做过，直接复用旧版本，不重新生成。

```text
U: Let's try the bag in gold instead.             A: Here's the gold version. [[image_0004]]
U: Use the gold one for now and put it with the traveler in a short test clip.
A: Here's the test clip. [[video_0001]]
U: Hmm, go back to the earlier purple bag.        A: Got it, back to the purple bag.
```
```text
U: I prefer the earlier white coffee cup. Use that version again in the close-up clip.
A: Here's the close-up with the white cup restored. [[video_0003]]
```

**2. 目标澄清（Goal clarification）**：用户一开始只有模糊的想法，assistant 追问，用户再给出具体设计。

```text
U: I'd like to figure out the look of the coffee shop.
A: What would you like the coffee shop to look like?
U: Wooden tables, a glass entrance, warm light at night. Can I get a reference image?
A: Here's the coffee shop based on that. [[image_0001]]
```

**3. 指代歧义（Ambiguous reference）**：用户想改某个东西，但说得含糊，"那个东西"可能指好几个资产。planner 把这一轮标记为"有意歧义"，verifier 不会要求用户把话说清楚。

```text
U: I'd like to change that thing in one of the earlier images.
A: Are you referring to this one? [[image_0003]]（故意展示一个错误的候选，见行为 8）
```
批量修改中的歧义：
```text
U: Make these edits together: in shot 2, change that thing from the earlier image to white; ...
A: Before I update the clips — for that edit, did you mean the coffee shop or the coffee cup?
```

**4. 一轮多候选（Multiple candidates in one turn）**：一次要两个只差一个属性的版本来对比，assistant 把两个都交付出来，由用户选（大多是图片，偶尔是视频）。

```text
U: Make two reference images of the Tokyo street to compare — option 1 at night, option 2 at dusk.
A: Here are both options. [[image_0009]] [[image_0010]]
U: Let's go with option 1.                        A: Option 1 it is.
```

**5. 先准备后组装（Separate preparation and assembly）**：先把各个资产单独做好、定稿，再用它们去组成镜头。

```text
U: Let's prep the traveler first. We'll put them with the tote bag in a test shot later.
A: What should the traveler look like?
...（各自做好参考图并采纳）
U: Now make the shot: a medium side-tracking shot of the traveler walking down the street with the bag, using the references we made.
A: Here's the shot. [[video_0002]]
```

**6. 试拍后拼接（Trial shots and assembly）**：做好几个镜头后，要求按某个顺序拼起来预览；也可以把拼接和修改合在同一个请求里。

```text
U: Can you put those shots together so I can see how they play?
A: Here's the preview cut. [[video_0004]]
```
```text
U: Put the existing clips together as shot 1 then shot 2. Also, in shot 1, make the bag blue.
A: I'll assemble them in that order, and per our rules I'll also bring the clips to a fall background. Before updating, ...
```

**7. 修改范围澄清（Revision scope clarification）**：要修改的资产出现在多个已经做好的镜头里，assistant 先问只改用户提到的这个镜头，还是所有用到它的镜头。

```text
U: In shot 1 of the current cut, make the tote bag blue.
A: I'll update the bag reference first. The same bag also appears in the coffee-shop entrance clip — update just shot 1, or all existing shots that use it?
U: Update all of them.
A: Here's the updated bag and the reworked clips, plus the refreshed cut. [[image_0006]] [[video_0005]] [[video_0006]] [[video_0007]]
```

**8. 指代纠正（Reference correction）**：assistant 理解错了用户指的东西，用户纠正。这是在澄清"指的是哪个"，不等于采纳某个版本。

```text
A: Are you referring to the coffee shop image?
U: No, not the coffee shop — I meant the earlier tote bag.
A: Got it, the tote bag. What would you like to change about it?
U: In the entrance clip, set its color to blue.
```

**9. 任务挂起与恢复（Task suspension and resumption）**：手上的资产做到一半先放下，去做另一个，之后回到原来的草稿上继续改。

```text
U: Make a quick test image of the tote bag, metal finish.   A: Here's the draft. [[image_0004]]
U: Put the bag aside for now — let's switch to the coffee cup.
A: Sure, I'll keep the bag draft as is.
...（完成咖啡杯）
U: Back to the tote bag. In that earlier test image, make it blue.
A: Picked the bag back up — here's the blue version. [[image_0007]]
```

**10. 多参考组合（Multiple-reference composition）**：一次生成同时引用多个已经做好的资产：人物、道具和场景一起进一个镜头，或者两个资产合拍一段试拍。

```text
U: Make the shot using the traveler, the tote bag and the Tokyo street we made: a medium tracking shot from behind as the traveler pushes open the glass door.
A: Here's the shot. [[video_0003]]   （tool_call 中的 references：<Picture 1> traveler、<Picture 2> bag、<Picture 3> street）
```

**11. 依赖上下文的镜头编号（Context-dependent shot numbers）**："第 1 个镜头"既可以按生成顺序数，也可以按当前剪辑里的顺序数。用户说得含糊，assistant 追问。

```text
U: Swap shot 1 and shot 3.
A: Do you mean shot 1 in generation order — the street tracking shot — or shot 1 in the current cut, the coffee-cup close-up?
U: The numbering in the current cut.
A: Here's the reordered cut. [[video_0008]]
```

**12. 相对修改与预览（Relative edits and previews）**：用户不说具体值，只说"再年轻一点"、"再浅一点"。planner 只在有序属性上取相邻等级，assistant 先给预览确认，用户先临时采用。

```text
U: Make an initial reference image of the traveler.   A: Here it is. [[image_0002]]
U: Make the person a bit younger.
A: Here's a younger version, in the adult age group. Is that what you meant? [[image_0003]]
U: Use this for now while we explore.
```

**13. 持续规则与例外（Persistent rules and exceptions）**：用户立一条"以后都这样"的规则（全局，或只对图片、只对视频、只对某个资产），之后可以取消，也可以只在某一次破例。

```text
U: From now on, use winter as the background season for all images and videos.
A: Noted — winter for all subsequent images and videos.
...
U: Make a reference of the coffee cup, but as a 9:16 portrait image just this once.
...
U: Cancel the night-time rule for videos. Leave everything else as is.
A: Done — videos no longer need to be at night; the other rules still apply.
```

**14. 冲突需求（Conflicting requirements）**：有两种。一种是物理上做不到的要求（同一个镜头既要全景又要特写、固定机位却要 360° 环绕、一镜到底却要两个硬切、背景既要清晰又要全糊、一个画面同时是 9:16 和 16:9），assistant 解释为什么矛盾，并请用户选一个。另一种是新请求和已有规则冲突，assistant 提醒用户，让用户在三种处理方式中选择。

```text
U: New shot: one continuous unbroken take, with exactly two obvious hard cuts in the middle.
A: A continuous take can't contain hard cuts — those would split it into separate shots. Which do you want?
U: Drop the cuts. Make a single take: a medium side-tracking shot of the traveler on the street.
```
```text
U: Make the street shot in spring.
A: The existing rule says winter for all images and videos. Keep it, make an exception just for this shot, or switch to spring going forward?
U: Spring for this one only; keep winter for later.
```

**15. 采纳决策（Adoption decisions）**：对每个产出做出决定：接受、临时采用、否决后修改、多选一、最终验收。约一半的接受是隐式的（用户直接继续往下做）。

```text
U: That one works.                                     （接受）
U: Use this version for now while we explore.          （临时采用）
U: Set that one aside. Make the coffee cup gray instead.（否决并修改）
U: Choose option 2.                                    （多选一）
U: Let's make this the final cut.                      （最终验收）
```

7 个**准备模块**把这些行为组合成完整片段：`direct`、`candidates`、`trial_edit`、`clarify`、`relative_preview`（单资产），以及 `task_switch`、`temporary_rollback`（双资产）。镜头渲染、冲突、返工、拼接、编号歧义、规则事件和最终验收由主循环调度。一个 Tokyo 示例一般有约 50 轮对话、14–16 个媒体占位符，覆盖 11–13 类行为。每个模块和 action 的细节见 [docs/planner.md](docs/planner.md)。

---

## 其他关键设计

### 有限的属性目录，可校验的变化

所有可编辑外观都落在 `attribute_catalog.json` 的四类资产字段里：person 19 个，object 13 个，indoor 21 个，outdoor 20 个。

- **只在视觉上明显的字段上采样变化**：物体的颜色/材质，人物的服装颜色/年龄段，环境的季节/时段/天气。看不出来的字段（`unknown`）不参与变化。
- `canonical_value` 做语义归一（grey→gray，autumn→fall，颜色列表不计顺序）。`design_key` 记录每个资产试过的完整设计组合：不会重复"试一个已经试过的设计"，要回到旧设计就引用旧版本。
- observer 可以给某个资产新增字段值（如一套特殊制服），这些值登记在该资产的 `attribute_extensions` 里随管线传递，不修改全局目录。
- **身份名中立化**：资产名只描述类别（"teacup" 而不是 "porcelain teacup"），外观都放进属性。这样修改材质后，不会和旧镜头描述里的形容词冲突。所有写作环节都遵循 `ATTRIBUTE_AUTHORITY`：当前结构化属性优先于源描述。

### 跨轮规则：按快照生效

- 规则是独立的一层，只在完整工作段之间插入，不会打断一次澄清。
- **冲突不靠专门制造演示媒体。** 挑战会先挂起，等下一个真正命中这条规则的生成请求出现，再附加到这个请求上。
- 每个媒体在创建时记录 `render_state` / `applied_rules` 快照，后来的规则不会回写历史媒体。纯拼接只受画幅规则约束，不会被新规则偷偷变成一次外观重渲。
- 默认画幅：图片 1:1，视频 16:9，可以被用户规则覆盖。

### 正向因果引用图

每个媒体的 `reference_materials` 只能引用对话中已经出现过的媒体：首次资产图零输入；编辑引用被编辑的版本；新镜头引用当前采纳的资产图；镜头修改引用旧镜头和变化了的资产图；拼接按 `shot_versions` 精确取片段。源视频从不作为生成输入。

### Writer + Verifier

Writer 拿到人设、最多 3 条最近消息（自包含的行为不给历史，避免模仿套话）、本轮意图、产物清单、当前属性。Verifier 按 `content_fidelity / grounding / conversation_coherence / naturalness` 打 0/1/2 分，全部为 2 才通过，否则给出一句 "You should …" 的修改意见。每轮最多 5 次尝试。planner 有意设计的歧义会以 `reference_policy` 告诉 verifier，不会被误判。

### 生成 → 审查 → 修复 → 剪枝

每个生成结果都由 VLM 评审（生产默认 Claude Sonnet）。评审看到真实对话、当前规则、实际参考和测得的画幅，按 5 个维度打 0/1/2 分，**只有明确的实质失败（0 分）才拒绝**。被拒绝后，prompt fixer 以刚被拒的结果为底图写一条编辑指令重试，而不是从头抽卡；每个媒体最多生成 3 次。一轮多个产物按顺序逐个生成、逐个评审。3 次仍失败时，剪掉依赖这个媒体的整段交互。

### 单卡调度

- **一张卡一个进程，卡与卡之间不通信。** 大规模并行靠同时提交多个 Slurm 作业。
- 单卡内：上一批 case 的媒体在 GPU 上生成时，下一批 case 的文本阶段已经在跑。
- 所有 case 的媒体单元汇入一个 `GenerationQueue`，按波次调度，每波求一个尽量少切换模型的拓扑序。被拒绝的单元推迟到下一波，不阻塞别的 case。
- 媒体评审在线程池中异步进行，但只有调度线程推进会话、写状态，不需要锁。

---

## 媒体生成后端的选型

| 选择 | 原因 |
|---|---|
| **SGLang 服务**（native batch 默认） | ComfyUI 路径太慢；FreeVideo（VDN-H3）是 H3 的蒸馏模型，虽然快，但视觉质量很差。SGLang 跑原生 H3 和 H3-Image，可以常驻，还支持动态挂载 LoRA |
| **视频编辑：H3 + Turbo LoRA，8 步** | 视频编辑（包括所有修复重试）占生成次数的大头。用 Turbo LoRA 8 步换取效率；同一个视频服务里动态挂载和卸载 LoRA，不用重启 |
| **其他视频生成：原生 H3，50 步** | 新镜头决定整条样本的视觉质量，使用原生 50 步 |
| **视频 DiT：NVFP4 mixed（4-bit）量化** | 经过实际生成质量对比后选定。生产命令通过环境变量 `H3_VIDEO_TRANSFORMER` / `H3_TEXT_VIDEO_TRANSFORMER` 指向 `minimax_h3_{ref2va,fl2va}_nvfp4_mixed*.safetensors`；不设置时，启动脚本回落到 `*_int8_convrot`。详见 [docs/media_backends.md](docs/media_backends.md) |
| **H3-MT VAE**（`--h3-vae mt`） | 原生 H3 反复做视频编辑时，会逐轮丢失细节、亮度下降。最初用[亮度匹配](docs/video_brightness_matching.md)稍作弥补，后来换成 MT VAE：它重新训练了 decoder，专门解决多轮编辑的细节丢失问题。最新的 run 使用 MT VAE，并用 `--no-match-video-brightness` 关闭了亮度匹配 |
| 图片：H3-Image，6 步，约 16 MP 画布 | 1:1 为 4096²，16:9 为 5472×3072，9:16 为 3072×5472 |
| 视频保持原生几何 | 4 秒，24 FPS 帧网格，保留生成的音轨 |

---

## 运行方式

### 生产：单卡 native batch

在服务器（Oscar，B200）主仓库提交。当前生产使用的命令：

```bash
cd /users/lwang223/cocreation
export COCREATION_SCRATCH=$(realpath scratch)
export COCREATION_SGLANG_RUNTIME="$COCREATION_SCRATCH/runtime/sglang-h3-benchmark"
export H3_VIDEO_TRANSFORMER="$COCREATION_SGLANG_RUNTIME/weights/minimax_h3_ref2va_nvfp4_mixed_sglang.safetensors"
export H3_TEXT_VIDEO_TRANSFORMER="$COCREATION_SGLANG_RUNTIME/weights/minimax_h3_fl2va_nvfp4_mixed.safetensors"

sbatch --parsable --export=ALL --job-name=dialogue-nvfp4-mt-48-c12 \
    run_native_dialogue_batch.sh \
    data/inputs/dialogue-native-48-reseed-20261125/manifest.json \
    --backend sonnet --think-mode medium --concurrency 12 \
    --resident-media-models --h3-vae mt \
    --no-match-video-brightness --reworks-policy manifest
```

| 部分 | 作用 |
|---|---|
| `H3_VIDEO_TRANSFORMER` / `H3_TEXT_VIDEO_TRANSFORMER` | 让 SGLang 视频服务加载 NVFP4 mixed 量化的 DiT（ref2va：有参考；fl2va：纯文生）。`--export=ALL` 把这些变量带进作业 |
| 清单（第一个位置参数） | 本批的 case；没有给第二个位置参数时，输出目录为 `data/runs/dialogue-native-$SLURM_JOB_ID` |
| `--backend sonnet --think-mode medium` | 文本阶段和媒体评审都通过 Claude Code CLI 调用 Sonnet 5.5 |
| `--concurrency 12` | 每批同时处理 12 个 case |
| `--resident-media-models --h3-vae mt` | H3-Image 和 H3 的 SGLang 服务常驻；视频和音频 VAE 使用 H3-MT |
| `--no-match-video-brightness` | MT VAE 已经解决多轮编辑的亮度问题，关闭亮度匹配（代码默认是开启的） |
| `--reworks-policy manifest` | 每个 case 的预留返工数直接取清单里的 `reworks`（默认策略是按 seed 随机取 0 或 1） |

> ⚠️ 只要传了位置参数，脚本里写死的默认选项就会全部失效，回落到 `native_batch.py` 自身的默认值（`--backend vllm --think-mode low --concurrency 24`、非常驻、native VAE、开启亮度匹配）。所以上面的命令把所有选项都显式写了出来，提交时不要省略。不带任何参数直接 `sbatch run_native_dialogue_batch.sh` 会使用脚本里写死的旧清单和旧选项，不是当前的生产配置。

**横向扩展**：把总清单切成若干份，每份提交一个单卡作业。每个作业有独立的 run 目录（目录名带 `SLURM_JOB_ID`），`.batch.lock` 防止两个作业写同一个目录。作业之间不需要协调。

**续跑**：设置同样的环境变量，用同一份清单、**把原来的 RUN_DIR 作为第二个位置参数**、带上同样的选项重新提交（不传 RUN_DIR 的话，新作业号会生成一个新目录）。observation、对话前缀、生成 prompt、已接受的媒体和审查都会复用。

清单格式、run 目录结构、状态文件和故障排查见 [docs/native_batch.md](docs/native_batch.md)。

### 本地开发（Mac）

本地没有 GPU，**不要在本地运行任何模型**。可以做的事：

```bash
# 纯 Python 规划：不调用任何模型
python3 -c "
import json
from scripts.dialogue_generation.planner import build_plan
plan = build_plan(json.load(open('scripts/dialogue_generation/examples/tokyo_vlog.json')), seed=1, reworks=1)
print(len(plan['turns']), 'turns', len(plan['placeholders']), 'placeholders')"

# 纯规划相关的单元测试（部分其他测试依赖服务器上的 ML 环境，本地不一定能跑）
python3 -m unittest tests.test_dialogue_generation tests.test_cross_turn_rule tests.test_planner_rules

# 从已拷回的 dialogue 目录查看生成计划，不生成
python3 scripts/dialogue_generation/materialize.py path/to/case/dialogue --dry-run
```

各阶段的单独入口（调试单个 case 用）见 [scripts/dialogue_generation/README.md](scripts/dialogue_generation/README.md)。

---

## 仓库结构

```text
run_native_dialogue_batch.sh      生产入口（sbatch，单卡）
scripts/dialogue_generation/      数据管线主体
  native_batch.py                 单卡批处理调度器（VLM 阶段 ‖ 媒体阶段）
  parallel_vlm.py                 每个 case 的文本阶段（observer → generate）
  planner.py / cross_turn_rule.py 对话规划与跨轮规则（纯 Python，用户模型所在）
  asset_variations.py             属性目录、变化采样、语义归一
  generate.py                     规划 + 逐轮台词 + prompt 改写
  generation.py / rewrite_prompts.py  生成合约与 prompt / 思考链
  materialize.py / media_queue.py / sglang_media.py / reviewed_turn.py  真实生成、审查、调度
  formal_sequence.py              导出 sequence.json / dialogue.md
tools/                            observer、writer、verifier、prompt writer、reasoner、reviewer、fixer、ffmpeg 工具
cocreation/                       模型封装（Qwen / vLLM / Claude Sonnet / H3 / H3-Image / Flux / VDN）与存储
skills/                           H3 / H3-Image / FLUX 官方 prompt 指南快照（嵌入 prompt writer 的 system prompt）
scripts/qwen, scripts/vllm        Qwen3.6 vLLM 服务
scripts/serve_h3_sglang.sh        H3 / H3-Image SGLang 服务启动
scripts/setup/                    运行时与权重安装
tests/                            单元测试
run_pipeline.sh, run_actual_prompt_pipeline.sh, scripts/dialogue_generation/batch.py
                                  早期的多阶段/串行 launcher，保留作参考，不再是生产入口
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/native_batch.md](docs/native_batch.md) | 生产运行手册：清单、sbatch、扩展、续跑、状态文件、故障排查 |
| [docs/data_contracts.md](docs/data_contracts.md) | 各阶段 JSON 契约：observation / plan / dialogue / placeholders / generation_plan / media_manifest / sequence |
| [docs/observation.md](docs/observation.md) | prompt observer、video observer、属性目录 |
| [docs/planner.md](docs/planner.md) | Planner 状态机、模块、返工、拼接、跨轮规则、校验器，以及如何扩展 |
| [docs/dialogue_realization.md](docs/dialogue_realization.md) | Writer / verifier / 人设 / 后端 |
| [docs/generation_prompts.md](docs/generation_prompts.md) | 生成合约、prompt writer、reasoner、production opening |
| [docs/materialization.md](docs/materialization.md) | 媒体会话协议、调度、审查与修复、剪枝、正式导出 |
| [docs/media_backends.md](docs/media_backends.md) | SGLang H3 / H3-Image 服务与选型、`cocreation` 模型封装、ComfyUI 路径 |
| [docs/video_brightness_matching.md](docs/video_brightness_matching.md) | 视频编辑亮度对齐（MT VAE 之前的方案） |
| [docs/qwen_server.md](docs/qwen_server.md)、[docs/qwen_precision.md](docs/qwen_precision.md) | Qwen3.6 vLLM 部署（`--backend vllm` 时使用） |
| [docs/vdn.md](docs/vdn.md) | VDN-H3 / FreeVideo（遗留后端：速度快但视觉质量差，生产管线不使用） |
