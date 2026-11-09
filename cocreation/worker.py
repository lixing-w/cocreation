"""Inference handlers with optional object-owned model state; no conversation cache."""
import json
from pathlib import Path
import sys
import time

def hf_options(runtime):
    return {k: runtime[k] for k in ("cache_dir", "revision", "local_files_only")
            if runtime[k] is not None}


def qwen_messages(request):
    content = []
    opts = request["options"]
    for ref in request["references"]:
        content.append(dict(type="text", text=ref["label"]))
        media = dict(type=ref["kind"], **{ref["kind"]: ref["path"]}, max_pixels=opts["max_pixels"])
        if ref["kind"] == "video":
            media.update(fps=opts["video_fps"], max_frames=opts["video_max_frames"])
        content.append(media)
    content.append(dict(type="text", text=request["prompt"]))
    return [dict(role="user", content=content)]


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
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=opts["max_new_tokens"], do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
    
    job_end_time = time.perf_counter()
    execution_time = job_end_time - job_start_time
    execution_time_f = f"{execution_time:.2f}s"
    return dict(status="ok", text=answer, model_id=request["model_id"],
                input_tokens=int(inputs.input_ids.shape[-1]), output_tokens=len(trimmed[0]), time_elapsed=execution_time_f)


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
