"""Inference handlers with optional object-owned model state; no conversation cache."""
import json
import re
from pathlib import Path
import sys
import time

def hf_options(runtime):
    return {k: runtime[k] for k in ("cache_dir", "revision", "local_files_only")
            if runtime[k] is not None}


def qwen_messages(request):
    """Place attached media after its first label mention; prepend unmentioned media.

    Args:
        request: Prepared Qwen request with resolved labels, paths and options.

    Repeated labels remain text to avoid duplicating visual context. Label numbers
    retain reference-list identity even when textual mentions reorder the media.
    The returned structured content is encoded by the official Qwen processor.
    """
    content = []
    opts = request["options"]
    refs = {ref["label"]: ref for ref in request["references"]}
    prompt = request["prompt"]

    def media_item(ref):
        """Build official image/video content for one prepared reference."""
        media = dict(type=ref["kind"], **{ref["kind"]: ref["path"]}, max_pixels=opts["max_pixels"])
        if ref["kind"] == "video":
            media.update(fps=opts["video_fps"], max_frames=opts["video_max_frames"])
        return media

    for label, ref in refs.items():
        if label not in prompt:
            content.extend([dict(type="text", text=label), media_item(ref)])
    if not refs:
        content.append(dict(type="text", text=prompt))
    else:
        pattern = "|".join(re.escape(label) for label in refs)
        seen = set()
        cursor = 0
        for match in re.finditer(pattern, prompt):
            label = match.group()
            if label in seen:
                continue
            content.append(dict(type="text", text=prompt[cursor:match.end()]))
            content.append(media_item(refs[label]))
            seen.add(label)
            cursor = match.end()
        if cursor < len(prompt) or not prompt:
            content.append(dict(type="text", text=prompt[cursor:]))
    messages = []
    if opts.get("system_prompt") is not None:
        messages.append(dict(role="system", content=opts["system_prompt"]))
    messages.append(dict(role="user", content=content))
    return messages


def run_qwen(request, directory, state=None):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    job_start_time = time.perf_counter()

    rt, opts = request["runtime"], request["options"]
    torch.manual_seed(opts["seed"])
    state = {} if state is None else state
    if 'model' not in state:
        state['model'] = Qwen3VLForConditionalGeneration.from_pretrained(request['model_id'],
            dtype=getattr(torch, rt['dtype']), device_map={'': rt['device']},
            attn_implementation='sdpa', **hf_options(rt)).eval()
        state['processor'] = AutoProcessor.from_pretrained(request['model_id'], **hf_options(rt))
    model, processor = state['model'], state['processor']
    messages = qwen_messages(request)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16,
        return_video_kwargs=True, return_video_metadata=True)
    video_metadata = None
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    inputs = processor(text=[text], images=images, videos=videos,
        video_metadata=video_metadata, do_resize=False, return_tensors="pt", **video_kwargs)
    inputs = inputs.to(rt["device"])
    cache_info = None
    with torch.inference_mode():
        if opts.get('system_cache_path'):
            from .qwen_prefix_cache import generate_with_prefix
            generated, cache_info = generate_with_prefix(model, processor, inputs, request, state)
        else:
            generated = model.generate(**inputs, max_new_tokens=opts["max_new_tokens"], do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
    
    job_end_time = time.perf_counter()
    execution_time = job_end_time - job_start_time
    execution_time_f = f"{execution_time:.2f}s"
    return dict(status="ok", text=answer, model_id=request["model_id"],
                input_tokens=int(inputs.input_ids.shape[-1]), output_tokens=len(trimmed[0]), time_elapsed=execution_time_f,
                system_cache=cache_info)


def run_flux(request, directory, state=None):
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image, ImageOps
    job_start_time = time.perf_counter()

    rt, opts = request["runtime"], request["options"]
    state = {} if state is None else state
    if 'pipe' not in state:
        pipe = Flux2KleinPipeline.from_pretrained(request['model_id'],
            torch_dtype=getattr(torch, rt['dtype']), **hf_options(rt))
        if rt['cpu_offload']:
            pipe.enable_model_cpu_offload(device=rt['device'])
        else:
            pipe.to(rt['device'])
        state['pipe'] = pipe
    pipe = state['pipe']
    images = []
    for ref in request["references"]:
        with Image.open(ref["path"]) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGB"))
    kwargs = dict(prompt=request["prompt"], width=opts["width"], height=opts["height"],
        num_inference_steps=opts["steps"], guidance_scale=opts["guidance_scale"],
        generator=torch.Generator(device=rt["device"]).manual_seed(opts["seed"]))
    if images:
        kwargs["image"] = images
    with torch.inference_mode():
        result = pipe(**kwargs).images[0]
    output = directory / "image.png"
    result.save(output)
    job_end_time = time.perf_counter()
    execution_time = job_end_time - job_start_time
    execution_time_f = f"{execution_time:.2f}s"
    return dict(status="ok", path=str(output), kind="image", model_id=request["model_id"], time_elapsed=execution_time_f)


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory / "request.json").read_text())
    handlers = {"qwen": run_qwen, "flux": run_flux}
    result = handlers[request["backend"]](request, directory)
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


class ModelSession:
    def __init__(self, backend):
        self.backend = backend
        self.state = {}
        self.identity = None

    def execute(self, request, directory):
        identity = (request['model_id'], request['runtime'])
        if self.identity is not None and identity != self.identity:
            raise ValueError('Close the object before changing model/runtime configuration')
        self.identity = identity
        return {'qwen': run_qwen, 'flux': run_flux}[self.backend](request, directory, self.state)

    def close(self):
        self.state.clear()


if __name__ == "__main__":
    main()
