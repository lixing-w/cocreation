"""Explicit checkpoint names: legacy :30b on this machine is Thinking."""
OLLAMA_INSTRUCT = "qwen3-vl:30b-a3b-instruct-q4_K_M"
OLLAMA_THINKING = "qwen3-vl:30b-a3b-thinking-q4_K_M"
LOCAL_INSTRUCT = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LOCAL_THINKING = "Qwen/Qwen3-VL-30B-A3B-Thinking"
MODEL_ALIASES = {
    "qwen3-vl-30b-instruct": {"ollama": OLLAMA_INSTRUCT, "local": LOCAL_INSTRUCT},
    "qwen3-vl-30b-thinking": {"ollama": OLLAMA_THINKING, "local": LOCAL_THINKING},
}
DEFAULT_OLLAMA_MODEL = OLLAMA_INSTRUCT
DEFAULT_LOCAL_MODEL = LOCAL_INSTRUCT


def resolve_model(backend, model=None, mode=None):
    name = model or (DEFAULT_OLLAMA_MODEL if backend == "ollama" else DEFAULT_LOCAL_MODEL)
    name = MODEL_ALIASES.get(name, {}).get(backend, name)
    inferred = "thinking" if "thinking" in name.lower() or name == "qwen3-vl:30b" else "instruct"
    if mode is not None and mode not in ("instruct", "thinking"):
        raise ValueError("model_mode must be instruct or thinking")
    return name, mode or inferred
