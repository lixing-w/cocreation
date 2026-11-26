# Compound revisions and H3 Image integration

The planner keeps the 15 behavior categories, bounded reference edit chains, explicit shot scopes, causal references, rule timelines, and shot-number clarification. It reduces repeated generation by collecting related changes before delivering another video version.

- A rule-driven assembly can share a pending identity revision. The assistant acknowledges the assembly and established rendering requirements, then resolves the identity and shot scope before generating video.
- At most two reserved revisions join this request. The user can introduce the second change while answering the first clarification; the assistant resolves the new reference/scope separately. No video is delivered between those clarification rounds.
- Existing standalone revision paths remain available. Feedback after a delivered version remains a separate revision.
- The union of affected shots is rendered once. Identity changes apply only within each confirmed scope; rendering changes can also affect other shots without changing their identity bindings.
- Each assembly child is diffed against its own realized state. Already compliant clips are reused.
- Attribute candidates favor still images (occasional video candidates remain). Rule demonstrations use stills whenever the rule scope allows them; explicitly video-only demonstrations and action tests remain videos.
- New image tasks use the existing H3 Image prompter/materializer path, recommended native image canvases, six steps and CFG 1. Video settings remain H3 Ref2VA plus Turbo LoRA.

## Previously proposed 15 video calls

| # | Placeholder | Purpose |
|---|---|---|
| 1 | video_0001 | Person design/rule preview |
| 2 | video_0002 | Another person design/rule preview |
| 3 | video_0003 | S1: wide riding shot |
| 4 | video_0004 | S2: side tracking shot |
| 5 | video_0005 | S3: window close-up |
| 6 | video_0007 | Fabric box candidate |
| 7 | video_0008 | Paper box candidate |
| 8 | video_0009 | Selected box preview in the new format |
| 9 | video_0010 | S4: stop, open bag, take out box |
| 10 | video_0011 | S1: winter and pink sheep |
| 11 | video_0012 | S2: winter and pink sheep |
| 12 | video_0013 | S3: winter and pink sheep |
| 13 | video_0014 | S4: pink sheep |
| 14 | video_0016 | S2: brown bag |
| 15 | video_0017 | S4: brown bag |

Five calls concern asset previews rather than story motion. The last two overlap shots already revised for the sheep. Using image previews and deferring the video commit until both scopes are resolved leaves four initial story clips and four combined revised clips.

## Story replay

Using the saved observation from sequence-9259651-20261118T032004Z, seed 0, and two reserved revisions:

| Plan | Turns | Video model calls | Hidden assembly edits |
|---|---:|---:|---:|
| Original run | 76 | 18 | 4 |
| First compound draft | 74 | 15 | 0 |
| Further compaction | 68 | 8 | 0 |

The story retains behavior ids 2 through 15; behavior 1 was absent in the original too. A 360-plan sample over two fixtures and reworks 0/1/2 retains all 15 behavior categories. Replays are planning checks, not new video generation or measured speedups. Sampling changed, so specific later rule/scope choices need not match the original dialogue.

## CUDA 12.8 image runtime

The existing W4A8 H3 Image checkpoint carries a preferred CUDA attention method which could bypass --use-pytorch-cross-attention and select packaged CUDA13 kernels. The image launcher disables only those attention availability probes on CUDA12, preserving Triton quantization and selecting PyTorch SDPA. It initializes Comfy's allocator before importing Torch. No CUDA13 upgrade is used.
