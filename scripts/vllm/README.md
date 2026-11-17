# Qwen3.6 vLLM on a newly allocated GPU node

After entering your GPU interactive session:

```bash
cd /users/lwang223/cocreation
bash -l scripts/vllm/start.sh
```

This loads CUDA 12.9 and uses the existing environment and model in shared scratch.
It runs in the foreground. Wait for `Application startup complete`, then run the
observation/dialogue commands from another terminal on **the same GPU node**.
The API is `http://127.0.0.1:8000`. Thinking defaults to false; optional server
arguments are forwarded, for example `bash -l scripts/vllm/start.sh --enable-thinking`.

To stop, press Ctrl+C in the serving terminal, or run this on the same GPU node:

```bash
cd /users/lwang223/cocreation
bash scripts/vllm/stop.sh
```

The stop command checks the recorded process identity before signaling it.
Run one service at a time with this shared deployment directory. The service's
lifetime is bounded by your GPU allocation. Node names are not hardcoded.
