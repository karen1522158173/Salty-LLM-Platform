"""下载维基语料 + 训练 BPE 分词器(对应旧 build_vocab_pipeline.py)

用法:
  python scripts/build_vocab.py
  python scripts/build_vocab.py --vocab-size 32000 --output ./my_micro_tokenizer \
      --corpus-mb 100
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<|audio_start|>",
    "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>",
    "<tts_text_eod>", "<tts_text_bos_single>", "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>", "<think>", "</think>",
    "<|buffer1|>", "<|buffer2|>", "<|buffer3|>", "<|buffer4|>", "<|buffer5|>",
    "<|buffer6|>", "<|buffer7|>", "<|buffer8|>", "<|buffer9|>",
    "<unk>", "<pad>",
]


def download_corpus(output_file: str, target_size_mb: int) -> None:
    if os.path.exists(output_file) and os.path.getsize(output_file) > target_size_mb * 1024 * 1024 * 0.8:
        print(f"已有语料库 {output_file},跳过下载")
        return

    print(f"流式下载双语语料(目标 {target_size_mb} MB)")
    configs = [
        {"path": "wikimedia/wikipedia", "name": "20231101.zh", "split": "train"},
        {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train"},
    ]
    target_bytes_per_lang = (target_size_mb * 1024 * 1024) / len(configs)

    with open(output_file, "w", encoding="utf-8") as f:
        for cfg in configs:
            print(f"抓取 {cfg['path']}({cfg['name']})")
            dataset = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True)
            current_bytes = 0
            for item in dataset:
                text = item["text"].strip()
                if not text:
                    continue
                f.write(text + "\n")
                current_bytes += len(text.encode("utf-8"))
                if current_bytes >= target_bytes_per_lang:
                    break
            print(f"{cfg['name']} 完成")


def build_tokenizer(corpus_files: List[str], vocab_size: int, output_dir: str) -> None:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=_SPECIAL_TOKENS,
        show_progress=True,
    )

    print("训练 BPE 词表(可能耗时数分钟)")
    t0 = time.time()
    tokenizer.train(corpus_files, trainer)
    print(f"耗时 {time.time() - t0:.2f} s")

    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=_SPECIAL_TOKENS[:-2],
    )
    os.makedirs(output_dir, exist_ok=True)
    fast.save_pretrained(output_dir)
    print(f"Tokenizer 保存至 {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="下载维基语料 + 训练 BPE 分词器")
    p.add_argument("--corpus", default="corpus.txt")
    p.add_argument("--corpus-mb", type=int, default=100)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--output", default="./my_micro_tokenizer")
    p.add_argument("--skip-download", action="store_true", help="跳过下载,直接用已存在 corpus")
    args = p.parse_args()

    if not args.skip_download:
        download_corpus(args.corpus, args.corpus_mb)
    build_tokenizer([args.corpus], args.vocab_size, args.output)

    print("\n--- Pipeline 验证 ---")
    tok = AutoTokenizer.from_pretrained(args.output)
    test = "人工智能大模型开发极具挑战性。<|image_pad|> <think>开始思考...</think>"
    encoded = tok.encode(test)
    print(f"输入: {test}")
    print(f"编码: {encoded}")
    print(f"解码: {tok.decode(encoded)}")


if __name__ == "__main__":
    main()
