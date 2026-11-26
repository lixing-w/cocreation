"""Platform defaults: Qwen3.6 NVFP4 on CUDA; existing Ollama models on Mac."""
import sys
from cocreation.qwen_client import MODEL_ID
from cocreation.sonnet_client import SONNET_EFFORTS, SONNET_MODEL_ID
OLLAMA_INSTRUCT = "qwen3-vl:30b-a3b-instruct-q4_K_M"
OLLAMA_THINKING = "qwen3-vl:30b-a3b-thinking-q4_K_M"
LOCAL_INSTRUCT = "Qwen/Qwen3.6-35B-A3B"
LOCAL_THINKING = LOCAL_INSTRUCT
DEFAULT_BACKEND = 'vllm' if sys.platform.startswith('linux') else 'ollama'
BACKENDS = ('ollama', 'local', 'vllm', 'sonnet')
MODEL_ALIASES = {
    "qwen3-vl-30b-instruct": {"ollama": OLLAMA_INSTRUCT},
    "qwen3-vl-30b-thinking": {"ollama": OLLAMA_THINKING},
    "qwen3.6-35b-a3b": {"local": LOCAL_INSTRUCT, "vllm": MODEL_ID},
}
DEFAULT_OLLAMA_MODEL = OLLAMA_INSTRUCT
DEFAULT_LOCAL_MODEL = LOCAL_INSTRUCT


def resolve_think_mode(backend, think_mode):
    if think_mode is not None and (backend != 'sonnet' or think_mode not in SONNET_EFFORTS):
        raise ValueError('think_mode must be low or medium and requires the Sonnet backend')
    return (think_mode or 'low') if backend == 'sonnet' else None


def resolve_model(backend, model=None, mode=None):
    if backend == "sonnet":
        if mode not in (None, "instruct", "thinking"):
            raise ValueError("model_mode must be instruct or thinking")
        return model or SONNET_MODEL_ID, mode or "instruct"
    name = model or (DEFAULT_OLLAMA_MODEL if backend == "ollama" else
                     MODEL_ID if backend == 'vllm' else DEFAULT_LOCAL_MODEL)
    if backend != 'ollama' and ('qwen3-vl' in name.lower() or 'qwen3-vl' in name.lower().replace('/', '-')):
        raise ValueError('The CUDA pipeline uses qwen3.6-35b-a3b; legacy Qwen3-VL is Ollama-only')
    name = MODEL_ALIASES.get(name, {}).get(backend, name)
    inferred = "thinking" if "thinking" in name.lower() or name == "qwen3-vl:30b" else "instruct"
    if mode is not None and mode not in ("instruct", "thinking"):
        raise ValueError("model_mode must be instruct or thinking")
    return name, mode or inferred
