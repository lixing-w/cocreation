# Independent generation and classic 8+3 sampling

Changes following the review of sequence 9242687:

- Independent image/video tasks now supply a `scene_invention` contract containing only the requested subjects, current references and attributes. The writer chooses a coherent still composition or a simple video action. An absent environment permits a minimal unobtrusive backdrop; it does not permit an unprovided person, animal, vehicle or salient prop.
- The target-shot `action_context` selector and its writer projection are removed. Formal `generate_clip` tasks retain their declared shot description and existing readiness checks. Unrelated subject attributes from broader references are excluded from independent requests.
- Active H3 examples use neutral objects and settings, removing the sheep/canal/traveler examples that could prime the target story.
- Replaying the saved sequence plan for turn 16 now produces only `person` and `image_0003` as writer context; its input/system contain neither sheep nor canal.
- Canvas ratio is controlled through the executor's numeric width/height, forwarded as FreeVideo `--width`/`--height`. Output orientation remains composition metadata. The forced ratio sentence and requirement to repeat numeric ratios in generation text are removed.
- `run_pipeline.sh` explicitly requests `--video-steps 8 --video-two-pass`. The VDN adapter maps this to 8 base steps, 3 refinement steps and a 2x latent upscaler. Thinking remains disabled. Slurm output/error directives are preserved.

The installed FreeVideo `two_pass.plan()` was invoked directly for each current canvas:

| Video ratio | First pass, 8 steps | Second pass, 3 steps | Delivered video |
|---|---|---|---|
| 16:9 | 512×288 | 1024×576 | 1280×720 |
| 9:16 | 288×512 | 576×1024 | 720×1280 |
| 1:1 | 512×512 | 1024×1024 | 1024×1024 |

FreeVideo's width/height refer to the second-pass target, not the first-pass canvas. The first/second-pass transition uses the learned latent upscaler and refinement; the later export resize is separate. Image canvases retain their existing ratio-specific sizes.
