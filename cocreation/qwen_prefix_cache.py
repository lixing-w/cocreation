"""System-only prefix reuse for Qwen3-VL, without changing Transformers classes.

The official processor prepares all inputs. We prefill new media with forward(),
leaving the final text token to generate(), so its decode-only media suppression
is safe. Only tensor data and compatibility metadata are persisted, never objects.
"""
from .qwen_sampling import generation_options
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile


def _fingerprint(model, processor, request, prefix_ids):
    """Identify weights, runtime, template and exact tokens for one cached prefix."""
    import torch
    import transformers
    config = model.config.to_dict()
    commit = getattr(model.config, '_commit_hash', None)
    location = Path(request['model_id']).expanduser()
    local_files = []
    if location.is_dir():
        for path in sorted(location.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.safetensors', '.bin'):
                stat = path.stat()
                local_files.append((str(path.relative_to(location)), stat.st_size, stat.st_mtime_ns))
    # If weights cannot be identified across processes, restrict reuse to this model.
    weight_identity = commit or local_files or f'process:{os.getpid()}:model:{id(model)}'
    data = dict(format=1, model_id=request['model_id'], weights=weight_identity,
                revision=request['runtime'].get('revision'), config=config,
                dtype=str(model.dtype), device=str(model.device),
                attention=getattr(model.config, '_attn_implementation', None),
                torch=torch.__version__, transformers=transformers.__version__,
                template=processor.chat_template, prefix_ids=prefix_ids.tolist())
    encoded = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_payload(payload, fingerprint, length, model):
    """Check serialized prefix identity, layer dimensions, dtype and finite tensors."""
    import torch
    if not isinstance(payload, dict) or set(payload) != {'fingerprint', 'layers'}:
        return False
    if payload['fingerprint'] != fingerprint or not isinstance(payload['layers'], list):
        return False
    config = model.config.text_config
    shape = (1, config.num_key_value_heads, length,
             getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads)
    if len(payload['layers']) != config.num_hidden_layers:
        return False
    for pair in payload['layers']:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return False
        for tensor in pair:
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != model.dtype or not torch.isfinite(tensor).all().item()):
                return False
    return True


def _save_payload(path, payload):
    """Atomically persist tensor-only payload at path, allowing concurrent readers."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _prefix_cache(model, processor, request, prefix_ids, state, path):
    """Return a private DynamicCache from memory/disk, or prefill and persist it."""
    import torch
    from transformers import DynamicCache
    length = prefix_ids.shape[-1]
    fingerprint = _fingerprint(model, processor, request, prefix_ids.cpu())
    payload = state.get('system_prefix_payload')
    status = 'memory'
    if payload is None or payload['fingerprint'] != fingerprint:
        payload = None
        status = 'built'
        if path.is_file():
            try:
                candidate = torch.load(path, map_location='cpu', weights_only=True)
                if _valid_payload(candidate, fingerprint, length, model):
                    payload, status = candidate, 'disk'
            except (OSError, ValueError, TypeError, RuntimeError, EOFError, KeyError, IndexError):
                pass
            except pickle.UnpicklingError:
                pass
        if payload is None:
            positions = torch.arange(length, device=prefix_ids.device)
            result = model(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids),
                           position_ids=positions.view(1, 1, -1).expand(3, 1, -1),
                           cache_position=positions, use_cache=True, logits_to_keep=1,
                           return_dict=True)
            payload = dict(fingerprint=fingerprint, layers=[
                (key.detach().cpu().clone(), value.detach().cpu().clone())
                for key, value in result.past_key_values.to_legacy_cache()])
            del result
            if not _valid_payload(payload, fingerprint, length, model):
                raise RuntimeError('Qwen produced an incompatible system prefix cache')
            _save_payload(path, payload)
        # Keep one immutable CPU prefix per worker; do not retain per-artifact history.
        state['system_prefix_payload'] = payload
    pairs = tuple((key.to(device=model.device, copy=True), value.to(device=model.device, copy=True))
                  for key, value in payload['layers'])
    return DynamicCache.from_legacy_cache(pairs), status


def generate_with_prefix(model, processor, inputs, request, state):
    """Prefill this request's media, then use official generate with a cached system.

    Args:
        model: Loaded Qwen3VLForConditionalGeneration in evaluation mode.
        processor: Its official processor, including tokenizer and chat template.
        inputs: Official processor output on model.device, one unpadded sequence.
        request: Prepared request with system_prompt and system_cache_path options.
        state: Object-owned worker state; stores only an immutable system prefix.

    Returns:
        Generated token tensor (including full input IDs), and cache diagnostics.
        Caller must run under torch.inference_mode().
    """
    import torch
    opts = request['options']
    path = Path(opts['system_cache_path'])
    ids, mask = inputs['input_ids'], inputs['attention_mask']
    if ids.shape[0] != 1 or not torch.all(mask == 1).item():
        raise ValueError('System prefix caching requires one unpadded request')
    text = processor.apply_chat_template(
        [dict(role='system', content=opts['system_prompt'])],
        tokenize=False, add_generation_prompt=False)
    prefix = processor.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(ids.device)
    n, total = prefix.shape[-1], ids.shape[-1]
    if not 0 < n < total - 1 or not torch.equal(prefix, ids[:, :n]):
        raise ValueError('Rendered system tokens are not an exact prefix of this request')
    visual_ids = [model.config.image_token_id, model.config.video_token_id,
                  model.config.vision_start_token_id, model.config.vision_end_token_id]
    if any(torch.any(prefix == token).item() for token in visual_ids):
        raise ValueError('Cached system must be pure text, without visual special tokens')
    if ids[0, -1].item() in visual_ids:
        raise ValueError('Generation boundary must end in a text token')
    scaling = model.config.text_config.rope_scaling or {}
    if scaling.get('rope_type', scaling.get('type', 'default')) != 'default':
        raise ValueError('System prefix caching currently supports default Qwen3-VL RoPE only')
    model.model.rope_deltas = None
    try:
        cache, status = _prefix_cache(model, processor, request, prefix, state, path)
        # Use Qwen's own multimodal position calculation on the COMPLETE sequence.
        positions, deltas = model.model.get_rope_index(
            ids, inputs.get('image_grid_thw'), inputs.get('video_grid_thw'),
            attention_mask=mask)
        model.model.rope_deltas = deltas
        media = {key: inputs[key] for key in
                 ('pixel_values','pixel_values_videos','image_grid_thw','video_grid_thw')
                 if key in inputs}
        # Leave the final assistant-prefix text token unprocessed. generate() must
        # see exactly one uncached token; all new media are already consumed here.
        result = model(input_ids=ids[:, n:-1], attention_mask=mask[:, :-1],
                       position_ids=positions[..., n:-1],
                       cache_position=torch.arange(n, total - 1, device=ids.device),
                       past_key_values=cache, use_cache=True, logits_to_keep=1,
                       return_dict=True, **media)
        if result.past_key_values.get_seq_length() != total - 1:
            raise RuntimeError('Unexpected cache length after multimodal prefill')
        generated = model.generate(
            input_ids=ids, attention_mask=mask, past_key_values=result.past_key_values,
            cache_position=torch.arange(total - 1, total, device=ids.device),
            use_cache=True, max_new_tokens=opts['max_new_tokens'], **generation_options(opts))
        return generated, dict(status=status, prefix_tokens=n, path=str(path))
    finally:
        # No multimodal position state may leak into a different artifact/request.
        model.model.rope_deltas = None
