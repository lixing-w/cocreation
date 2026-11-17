"""Run official ordered messages, or a batch of conversations, through native Qwen/vLLM."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cocreation import Qwen, VLLMQwen, Runtime
from cocreation.qwen_client import MODEL_ID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('messages', type=Path, help='JSON list of messages, or list of conversations with --batch')
    p.add_argument('--backend', choices=('local', 'vllm'), default='vllm')
    p.add_argument('--model')
    p.add_argument('--revision')
    p.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--batch', action='store_true')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    model = (Qwen(a.model, runtime=Runtime(revision=a.revision) if a.revision else None, output_dir=a.output_dir)
             if a.backend == 'local' else
             VLLMQwen(a.model or MODEL_ID, base_url=a.base_url, output_dir=a.output_dir))
    with model:
        method = model.generate_batch if a.batch else model.chat
        result = method(json.loads(a.messages.read_text()), max_new_tokens=a.max_new_tokens, seed=a.seed, enable_thinking=a.enable_thinking)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
