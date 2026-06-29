"""把预训练 jsonl 转成紧凑 uint16 .bin(对应旧 prepare.py)

用法:
  python scripts/prepare_pretrain_bin.py
  python scripts/prepare_pretrain_bin.py \
      --input ./data/raw/pretrain_t2t.jsonl \
      --output ./data/bin/train_data.bin \
      --tokenizer ./minimind_tokenizer
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from typing import List

import numpy as np
from transformers import AutoTokenizer


_DEFAULT_INPUT = "./dataset/pretrain_t2t.jsonl"
_DEFAULT_OUTPUT = "./train_bin/train_data.bin"
_DEFAULT_TOKENIZER = "./minimind_tokenizer"
_DEFAULT_CHUNK = 10000

_worker_tokenizer = None
_worker_eos_id = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer, _worker_eos_id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _worker_eos_id = _worker_tokenizer.eos_token_id or 2


def _process_chunk(lines: List[str]) -> np.ndarray:
    chunk_ids: List[int] = []
    for line in lines:
        try:
            text = json.loads(line).get("text", "")
            if text:
                ids = _worker_tokenizer.encode(text, add_special_tokens=False) + [_worker_eos_id]
                chunk_ids.extend(ids)
        except Exception:
            continue
    return np.array(chunk_ids, dtype=np.uint16)


def _read_in_chunks(file_path: str, chunk_size: int):
    with open(file_path, "r", encoding="utf-8") as f:
        chunk: List[str] = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def run(input_path: str, output_path: str, tokenizer_path: str, chunk_size: int, num_workers: int) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"启动多进程预处理 num_workers={num_workers}")

    total_tokens = 0
    chunks_processed = 0
    with open(output_path, "wb") as f_out:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            for np_ids in pool.imap(_process_chunk, _read_in_chunks(input_path, chunk_size)):
                f_out.write(np_ids.tobytes())
                total_tokens += len(np_ids)
                chunks_processed += 1
                print(
                    f"已处理 {chunks_processed * chunk_size} 行,累计 Tokens: {total_tokens / 1e6:.2f} M",
                    end="\r",
                )
    print(f"\n转换完成,总 Token 数: {total_tokens / 1e6:.2f} M -> {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="预训练 jsonl → uint16 bin")
    p.add_argument("--input", default=_DEFAULT_INPUT)
    p.add_argument("--output", default=_DEFAULT_OUTPUT)
    p.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER)
    p.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK)
    p.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = p.parse_args()
    run(args.input, args.output, args.tokenizer, args.chunk_size, args.num_workers)


if __name__ == "__main__":
    main()
