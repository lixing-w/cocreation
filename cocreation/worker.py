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
    if opts.get("assistant_prefill") is not None:
        messages.append(dict(role="assistant", content=opts["assistant_prefill"]))
    return messages


def run_qwen(request, directory, state=None):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    job_start_time = time.perf_counter()

    rt, opts = request["runtime"], request["options"]
    if opts.get('response_schema') is not None:
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor as SchemaLogitsProcessor
    state = {} if state is None else state
    if 'model' not in state:
        state['model'] = AutoModelForImageTextToText.from_pretrained(request['model_id'],
            dtype=getattr(torch, rt['dtype']), device_map={'': rt['device']},
            attn_implementation='sdpa', low_cpu_mem_usage=True, **hf_options(rt)).eval()
        state['processor'] = AutoProcessor.from_pretrained(request['model_id'], **hf_options(rt))
    model, processor = state['model'], state['processor']
    runtime_info = dict(model_class=type(model).__name__, dtype=str(model.dtype),
                        device=str(model.device), attention='sdpa', torch=torch.__version__)
    conversations = request.get('conversations') or [qwen_messages(request)]
    processor.tokenizer.padding_side = 'left'
    # Continue the supplied assistant text, rather than starting another turn.
    template_options = dict(add_generation_prompt=True, enable_thinking=opts.get("enable_thinking", False))
    if opts.get("assistant_prefill") is not None:
        template_options = dict(add_generation_prompt=False, continue_final_message=True)
    texts = [processor.apply_chat_template(messages, tokenize=False, **template_options)
             for messages in conversations]
    images, videos, video_kwargs = process_vision_info(conversations, image_patch_size=16,
        return_video_kwargs=True, return_video_metadata=True)
    video_metadata = None
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    inputs = processor(text=texts, images=images, videos=videos,
        video_metadata=video_metadata, do_resize=False, padding=True, return_tensors="pt", **video_kwargs)
    inputs = inputs.to(rt["device"])
    constraints = {}
    if opts.get('response_schema') is not None:
        if 'json_grammar_compiler' not in state:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                processor.tokenizer, vocab_size=model.config.text_config.vocab_size)
            state['json_grammar_compiler'] = xgr.GrammarCompiler(
                tokenizer_info, max_threads=2, cache_limit_bytes=128*2**20)
        grammar = state['json_grammar_compiler'].compile_json_schema(
            opts['response_schema'], max_whitespace_cnt=12)
        # A fresh matcher for EVERY call; only immutable compiled grammars are cached.
        constraints['logits_processor'] = [SchemaLogitsProcessor(grammar)]
    cache_info = None
    from .qwen_sampling import generation_options
    # Seed after model loading/preprocessing so cold and warm calls start alike.
    torch.manual_seed(opts["seed"])
    with torch.inference_mode():
        if opts.get('system_cache_path'):
            from .qwen_prefix_cache import generate_with_prefix
            generated, cache_info = generate_with_prefix(model, processor, inputs, request, state)
        else:
            generated = model.generate(**inputs, max_new_tokens=opts["max_new_tokens"],
                                       **generation_options(opts), **constraints)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answers = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    
    job_end_time = time.perf_counter()
    execution_time = job_end_time - job_start_time
    execution_time_f = f"{execution_time:.2f}s"
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    for index, (answer, tokens) in enumerate(zip(answers, trimmed)):
        values = tokens.tolist()
        stop = next((i for i, token in enumerate(values) if token in eos), None)
        results.append(dict(status='ok', text=answer, model_id=request['model_id'],
            input_tokens=int(inputs.attention_mask[index].sum().item()),
            output_tokens=stop + 1 if stop is not None else len(values),
            finish_reason='stop' if stop is not None else 'length',
            time_elapsed=execution_time_f, system_cache=cache_info, runtime=runtime_info))
    if request.get('batched'):
        return dict(status='ok', model_id=request['model_id'], results=results,
                    batch_size=len(results), time_elapsed=execution_time_f, runtime=runtime_info)
    return results[0]


def run_flux(request, directory, state=None):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Pipeline
    from PIL import Image, ImageOps
    job_start_time = time.perf_counter()

    rt, opts = request["runtime"], request["options"]
    state = {} if state is None else state
    if 'pipe' not in state:
        pipeline = Flux2Pipeline if request['model_id'] == 'black-forest-labs/FLUX.2-dev' else Flux2KleinPipeline
        pipe = pipeline.from_pretrained(request['model_id'],
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
