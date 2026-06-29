"""推理 CLI:python -m salty.inference.cli --ckpt xxx.pt --prompt "你好"

用作 Stage 1 验收:与旧 inference_sft.py 对比输出语义一致。
"""
import argparse
import sys

import torch

from .engine import InferenceEngine
from .sampler import SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Salty 推理 CLI")
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径")
    parser.add_argument("--tokenizer", default="./minimind_tokenizer")
    parser.add_argument("--prompt", required=True, help="用户输入")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rep-penalty", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="禁用 KV cache(用于与旧 inference_sft.py 对齐)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    engine = InferenceEngine(tokenizer_path=args.tokenizer, device=args.device)
    info = engine.load(args.ckpt)
    print(f"已加载: {info}", file=sys.stderr)

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rep_penalty=args.rep_penalty,
    )

    messages = [{"role": "user", "content": args.prompt}]
    print(f"[user] {args.prompt}", file=sys.stderr)
    print("[assistant] ", end="", flush=True)

    in_think = False
    for evt in engine.stream(messages, params, use_kv_cache=not args.no_kv_cache):
        if evt.type == "think_open":
            print("\n<think>", end="", flush=True)
            in_think = True
        elif evt.type == "think_close":
            print("</think>\n", end="", flush=True)
            in_think = False
        elif evt.type in ("delta", "think_delta"):
            print(evt.text, end="", flush=True)
        elif evt.type == "done":
            print("", flush=True)
            print(f"\n[done] tokens={evt.meta.get('tokens')}", file=sys.stderr)
        elif evt.type == "error":
            print(f"\n[error] {evt.text}", file=sys.stderr)


if __name__ == "__main__":
    main()
