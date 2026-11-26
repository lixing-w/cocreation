# Sequence review and policy changes

`scratch/runs/sequence-sbatch-20261117T054244Z`（server）已完成；88 turns、14 images、15 videos。

检查范围：逐条阅读全部对话、intent、属性差异、规则时间线和 generation records；查看全部图像，并对每个视频抽取约 10%、50%、90% 位置检查。以下视觉结论基于这些采样帧，不是模型质量保证。原始 run 保持原样。

## 主要发现

- 物品比例、透明度、纹理和地面状态等变体往往没有可辨变化；衣服 fit 修改还连带改变颜色/款式。
- 既有季节/黑白规则反复被加入普通用户请求，例如 turns 19、43、55、57、59、63。
- 视频 1/2/3 的包试拍实际扩展为女孩骑羊的场景；对白将部分 preview 称为 final。
- turns 74/82 将随附 cut 也计入 updated clips 的说法不精确。
- 黑白视频已经正确去色；但旧策略直接排除了黑白镜头的颜色返工案例，没有训练“保存新版、保留旧黑白镜头”的行为。

## 已应用的策略

- 物品只抽颜色和纸/玻璃/木/布织物/金属/棉/粉末/皮革；人物只抽服装颜色（含明确双色）和四档年龄。
- 场景只抽四季、夜晚/黎明/白天/傍晚和七档天气。室内新增 season/weather，旧输入默认 unknown。
- 按当前版本 canonical 值过滤相同属性；只允许明确 rollback 重用历史版本。采样不会拿已知属性换成 unknown。
- 普通对话改写去掉持久规则上下文；规则由 Python 保存，冲突/撤销/override 单独表达。生成合同仍接收当前输出条件。
- 最新镜头均黑白时，纯颜色返工只更新独立资产；旧视频与 cut 保留，未来彩色镜头绑定新版。
- 新增四种比例 preset，尺寸落实至模型调用、图片、视频、混合尺寸拼接和导出；单纯格式转换不调用 VDN。

## 全部图像

| 图像 | 引入 turn | 视觉检查 |
|---|---:|---|
| image_0001 | 4 | 初始包已是常见梯形/近矩形外观，未体现 conical。 |
| image_0002 | 8 | 羊仍为不透明常规羊；透明设定没有清晰实现。 |
| image_0003 | 10 | 比例修改与前版视觉差异很小。 |
| image_0004 | 20 | 白黄外套、短白发；两只包；底部有雪。 |
| image_0005 | 22 | 改合身同时变成灰色长衣，超出所请求字段。 |
| image_0006 | 30 | 棕包夏季布景；主形状基本延续。 |
| image_0007 | 32 | 纸箱没有清晰 carved 纹理。 |
| image_0008 | 34 | 平滑纸箱与前版几乎相同。 |
| image_0009 | 42 | 冬季布景但外套变为多色，偏离白黄设计。 |
| image_0010 | 44 | 夏季树/阳光；多色外套继续。 |
| image_0011 | 46 | 要求 dry 却仍有运河水面。 |
| image_0012 | 48 | standing water 已在前版实现，视觉改动很小。 |
| image_0013 | 72 | 包的形状返工未形成清楚对比。 |
| image_0014 | 80 | 羊原本已不透明，opaque 返工缺少可见变化。 |

## 全部视频

| 视频 | 内容/检查 |
|---|---|
| video_0001 | 彩色冬季试拍；女孩骑羊沿运河远去，地面有雪。 |
| video_0002 | 彩色例外；骑羊运河场景，人物服装/发色偏离独立资产。 |
| video_0003 | 黑白恢复，骑羊远去。 |
| video_0004 | 黑白广角远去。 |
| video_0005 | 黑白侧面跟拍。 |
| video_0006 | 侧面→广角拼接。 |
| video_0007 | 黑白窗边开包取盒；三个采样帧能看到动作推进。 |
| video_0008 | 侧面→广角→窗边拼接。 |
| video_0009 | 包返工后的广角，动作保留。 |
| video_0010 | 包返工后的侧面，动作保留。 |
| video_0011 | 包返工后的窗边取盒，动作保留。 |
| video_0012 | 替换三镜头后的原顺序 cut。 |
| video_0013 | 羊 opaque 返工后的窗边；原羊早已不透明，差异有限。 |
| video_0014 | 替换窗边后的 cut。 |
| video_0015 | 最终顺序广角→侧面→窗边，保留黑白。 |

## 每个 turn 的核对索引

| Turn | Role/action | Spoken line | Outputs |
|---:|---|---|---|
| 1 | user/vague_request | Let’s start designing the Brown Bag, but don’t settle on any visual details just yet. | — |
| 2 | assistant/clarify_goal | How would you like the Brown Bag to look? | — |
| 3 | user/specify_goal | I mean the bag: make a reference image of a small, plain, matte brown leather satchel with a white bird emblem attached to the flap. | — |
| 4 | assistant/deliver_asset | Here is the still reference image you asked for. [[image_0001]] | [[image_0001]] |
| 5 | user/set_cross_turn_rule | From now on, use fall as the background season for all generated images and videos, including assets, trials, intermediate results and finished footage. | — |
| 6 | assistant/confirm_cross_turn_rule | Got it—I'll use fall as the background season for all images and videos, including assets, trials, and final footage. Just to be clear, this rule applies only to backgrounds; other output categories remain outside this particular rule, and other existing rules still apply. | — |
| 7 | user/request_trial | Let's start with a test still of the Pink Sheep. I want a large, elongated, transparent sheep with voluminous pink wool riding on the water. | — |
| 8 | assistant/deliver_trial | Here is the completed initial still test image. [[image_0002]] | [[image_0002]] |
| 9 | user/relative_request | For the Pink Sheep, make the object proportions a little longer and narrower. | — |
| 10 | assistant/preview_for_confirmation | I made the object proportions of the Pink Sheep a little longer and narrower in this preview. Is that what you meant? [[image_0003]] | [[image_0003]] |
| 11 | user/adopt | I'll accept the current version of the Pink Sheep for now. | — |
| 12 | assistant/confirm_selection | Got it. I’ll go with that version of the Pink Sheep. | — |
| 13 | user/request_rule_conflict | Let's do a trial clip of the Brown Bag with winter in the background. | — |
| 14 | assistant/clarify_rule_conflict | The current rule requires fall as the background season for all generated images and videos, including trials. For the Brown Bag, would you like to keep this rule, make an exception just for this clip, or replace it going forward? | — |
| 15 | user/resolve_rule_conflict | Replace the earlier rule: from now on, use winter as the background season for all generated images and videos, including assets, trials, intermediate results and finished footage. Keep other rules unchanged. | — |
| 16 | assistant/confirm_rule_resolution | Got it. I've updated the rule to use winter as the background season for all future images and videos, including assets, trials, intermediate results, and finished footage, while keeping the other rules unchanged. | — |
| 17 | user/request_rule_followup | Generate a trial clip of the Brown Bag with winter as the background season. | — |
| 18 | assistant/deliver_rule_preview | Here is the final video preview based on the agreed rules. [[video_0001]] | [[video_0001]] |
| 19 | user/request_trial | Please create an initial reference image of the girl. She should have short white hair and a calm, feminine appearance, wearing an oversized white and yellow multicolored coat while carrying a brown bag with a white bird emblem, set against a winter background. | — |
| 20 | assistant/deliver_trial | The initial still test image is finished and ready for you. [[image_0004]] | [[image_0004]] |
| 21 | user/relative_request | Make the clothing fit on the girl a little more fitted. | — |
| 22 | assistant/preview_for_confirmation | I tightened up the fit on the girl’s clothing in this preview; is that what you had in mind? [[image_0005]] | [[image_0005]] |
| 23 | user/adopt | I accept the current version of the Girl, but just so we’re clear, this decision is temporary. | — |
| 24 | assistant/confirm_selection | Got it, I'm locking in that version of the girl. | — |
| 25 | user/request_rule_conflict | Create a standalone asset image of the Brown Bag with a summer background. | — |
| 26 | assistant/clarify_rule_conflict | Right now, our standing rule is that all visuals must have a winter background, which covers assets like this brown bag. Since that would clash with your summer request, do you want to keep the rule as is, make an exception just for this bag, or replace the rule going forward? | — |
| 27 | user/resolve_rule_conflict | Replace the earlier rule: from now on, use summer as the background season for all generated images and videos, including assets, trials, intermediate results and finished footage. Keep other rules unchanged. | — |
| 28 | assistant/confirm_rule_resolution | Got it. I’m updating the background rule to summer for all future images and videos, including assets, trials, intermediate results, and finished footage, while keeping everything else the same. | — |
| 29 | user/request_rule_followup | Make a new asset image of the brown bag with summer as the background season. | — |
| 30 | assistant/deliver_rule_preview | Here is the completed image preview based on the agreed rules. [[image_0006]] | [[image_0006]] |
| 31 | user/request_trial | I’d like a test still of the Cardboard Box—a small, brown, rectangular paper box with a plain, matte surface that features a carved texture. | — |
| 32 | assistant/deliver_trial | Here is the initial still test image, ready for your review. [[image_0007]] | [[image_0007]] |
| 33 | user/reject_and_edit | Let’s set that one aside. For the Cardboard Box, please change the surface texture to smooth. | — |
| 34 | assistant/deliver_edit | Here is the revised reference image with the edit applied. [[image_0008]] | [[image_0008]] |
| 35 | user/adopt | I accept the current version of the Cardboard Box. | — |
| 36 | assistant/confirm_selection | Got it, I’ll stick with that version of the cardboard box. | — |
| 37 | user/set_cross_turn_rule | From now on, use black-and-white rendering for formal shots. | — |
| 38 | assistant/confirm_cross_turn_rule | Noted, I'll keep the black-and-white rendering for all formal shots. This rule only applies to that category, so the guidelines for other output types remain unchanged. | — |
| 39 | user/request_rule_conflict | Let’s generate a standalone asset image of the girl with a winter background. | — |
| 40 | assistant/clarify_rule_conflict | I currently have a rule that requires a summer background for all generated assets, but this request is for a winter setting. Do you want to keep the existing summer rule, make an exception just for this winter image, or replace the rule entirely going forward? | — |
| 41 | user/resolve_rule_conflict | Make an exception just for this winter image. For later outputs, keep summer as the background season for all generated images and videos, including assets, trials, intermediate results and finished footage. | — |
| 42 | assistant/deliver_rule_preview | Here is the preview generated under the agreed rule decision. [[image_0009]] | [[image_0009]] |
| 43 | user/request_rule_followup | Let's make another standalone asset image of the girl, with summer as the background season. | — |
| 44 | assistant/deliver_rule_preview | Here is the preview of the completed image under the agreed rule decision. [[image_0010]] | [[image_0010]] |
| 45 | user/request_trial | Let's do a still test of the Canal Street shot. Keep the ground dry and the weather clear, but render it at dusk with moderate light so we can see the multi-story buildings, the laundry lines, and the neon signs. | — |
| 46 | assistant/deliver_trial | The initial still test image is ready for your review. [[image_0011]] | [[image_0011]] |
| 47 | user/reject_and_edit | Let's put that one aside. For the Canal Street shot, set the ground condition to standing water. | — |
| 48 | assistant/deliver_edit | Here is the updated reference image with the changes applied. [[image_0012]] | [[image_0012]] |
| 49 | user/adopt | I accept the version of the Canal Street just shown. | — |
| 50 | assistant/confirm_selection | Got it, I’m locking in that version for Canal Street. | — |
| 51 | user/request_rule_conflict | I need a formal video shot of the Brown Bag, but I want it rendered in its original colors. | — |
| 52 | assistant/clarify_rule_conflict | The current rule renders formal shots in black and white. For the Brown Bag video, would you like to keep that rule, make an exception just for this one, or replace it going forward? | — |
| 53 | user/resolve_rule_conflict | Keep the black-and-white rule for later outputs, but apply an exception for this specific Brown Bag request to use the original colors. | — |
| 54 | assistant/deliver_rule_preview | Here is the preview video. [[video_0002]] | [[video_0002]] |
| 55 | user/request_rule_followup | Produce a new video shot of the bag in black and white with a summer background. | — |
| 56 | assistant/deliver_rule_preview | Here is the completed preview video, fully rendered in black and white as agreed. [[video_0003]] | [[video_0003]] |
| 57 | user/request_shot | I want a new video shot in a wide angle. It should show a girl riding a sheep down a flooded canal street toward the sunset, keeping the girl, sheep, bag, and canal street references, and render it in black and white. | — |
| 58 | assistant/deliver_shot | Here is the completed video shot, fully rendered in black and white. [[video_0004]] | [[video_0004]] |
| 59 | user/request_shot | Please generate a new video shot showing a side view tracking shot of the girl riding the sheep past the Canal Street buildings. Keep the girl, sheep, and bag as the subjects, and ensure the entire clip is rendered in black and white. | — |
| 60 | assistant/deliver_shot | Here is the completed video shot. [[video_0005]] | [[video_0005]] |
| 61 | user/request_compilation | Please assemble the existing clips in this order: first, the side view tracking shot of the girl riding the pink sheep past the buildings, and second, the wide shot of her riding down a flooded canal street towards the sunset. | — |
| 62 | assistant/deliver_compilation | I’ve finished assembling the clips as requested, and here is the final video ready for you to review. [[video_0006]] | [[video_0006]] |
| 63 | user/request_shot | I need a new formal video shot showing the girl stopping by the window, opening her bag, and taking out a box. Make sure to include the sheep and the Canal Street background, and render the entire scene in black and white. | — |
| 64 | assistant/deliver_shot | Here is the final video shot, fully rendered in black and white as agreed. [[video_0007]] | [[video_0007]] |
| 65 | user/request_compilation | Put the clips together in this order: first the side view tracking shot of the girl riding the pink sheep past the buildings, then the wide shot of her riding down the flooded canal street towards the sunset, and finally the close-up of her stopping near a window, opening her bag, and taking out a box. | — |
| 66 | assistant/deliver_compilation | Here is the assembled video, combining the clips as requested. [[video_0008]] | [[video_0008]] |
| 67 | user/ambiguous_reference | I'd like to change that thing in one of the earlier images. | — |
| 68 | assistant/clarify_reference | Are you referring to this version: Pink Sheep? References: [[image_0002]] | — |
| 69 | user/correct_reference | No, you showed the Pink Sheep. I meant the earlier Brown Bag. | — |
| 70 | assistant/confirm_reference | Got it, the Brown Bag. What would you like to change about it? | — |
| 71 | user/request_rework | In shot 2 of the current cut, please set the Brown Bag's object shape to rectangular. | — |
| 72 | assistant/clarify_update_scope | Here is the updated preview for the Brown Bag reference; the video clips haven't changed yet. Since this asset is also in shot 1 and shot 3, would you like to update just this specific shot or all existing shots using it? [[image_0013]] | [[image_0013]] |
| 73 | user/choose_update_scope | Apply the rectangular shape to all existing shots that use it. | — |
| 74 | assistant/deliver_reworked_clips | Here are the four updated clips, along with the assembled cut that incorporates those replacements. That completes the agreed scope. [[video_0009]] [[video_0010]] [[video_0011]] [[video_0012]] | [[video_0009]], [[video_0010]], [[video_0011]], [[video_0012]] |
| 75 | user/ambiguous_reference | I want to change that element in one of the earlier stills. | — |
| 76 | assistant/clarify_reference | Are you referring to this version: Brown Bag? References: [[image_0001]] | — |
| 77 | user/correct_reference | No, you showed the Brown Bag. I meant the earlier Pink Sheep. | — |
| 78 | assistant/confirm_reference | Got it, you mean the Pink Sheep. What would you like to change about it? | — |
| 79 | user/request_rework | In shot 3, make the Pink Sheep opaque. | — |
| 80 | assistant/clarify_update_scope | Here’s the updated reference preview for the Pink Sheep, which also appears in shot 2 and shot 1 of the current cut; the videos have not been changed yet, so should we update just the mentioned shot or all existing shots using it? [[image_0014]] | [[image_0014]] |
| 81 | user/choose_update_scope | Just the one in shot 3. | — |
| 82 | assistant/deliver_reworked_clips | Here are the two updated clips as requested, along with the final cut that incorporates those changes in order. [[video_0013]] [[video_0014]] | [[video_0013]], [[video_0014]] |
| 83 | user/ambiguous_shot_number | Please swap shot 1 and shot 2. | — |
| 84 | assistant/clarify_shot_number | For the swap, do you mean the original first clip, which is a wide shot of a girl riding a pink sheep down a flooded canal street towards the sunset, or the one currently first in the cut, which is a side view tracking shot of the girl riding the pink sheep past the buildings? References: [[video_0014]] | — |
| 85 | user/resolve_shot_number | I mean the ones in the current cut. Swap shot 1 and shot 2. | — |
| 86 | assistant/deliver_compilation | Here is the assembled video using the existing clips. [[video_0015]] | [[video_0015]] |
| 87 | user/accept_project | Let's make this the final cut. | — |
| 88 | assistant/confirm_final | The cut is saved as final, but I’ll keep the reference versions available if you need them later. | — |


## 验证记录

- 服务器 `cocreation` 环境的 129 项相关回归通过；补充后的策略 9 项、observer 15 项专项检查通过。
- 使用本次 run 的真实 observation，四种比例 × 20 seeds，共 80 个计划及其生成依赖图验证通过。
- 比例检查使用替代生成器加真实 PNG/ffmpeg 输出，覆盖实际文件尺寸、混合横竖素材拼接、音视频时长。
- `bash -n run_pipeline.sh`、Python 编译和 `git diff --check` 通过。没有启动新 GPU generation run。
- 修改前完整旧套件已有 3 failures / 6 errors（旧 benchmark 预期、缺少依赖/旧文件路径、测试包导入及 Mac 路径/端口限制）；这些不在本次修改范围。相关检查单独运行通过。
