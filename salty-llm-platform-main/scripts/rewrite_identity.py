"""SFT 数据集身份替换(对应旧 rewrite_dataset_mp.py)

把 SFT 对话中提及的旧模型身份(minimind/chatgpt 等)随机替换为本项目身份。

用法:
  python scripts/rewrite_identity.py
  python scripts/rewrite_identity.py \
      --input ./data/raw/sft_t2t.jsonl \
      --output ./data/raw/sft_t2t_cleaned.jsonl \
      --my-name xiongziqi --my-model 咸菜
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from typing import List, Tuple

from tqdm import tqdm


_DEFAULT_INPUT = "./dataset/sft_t2t.jsonl"
_DEFAULT_OUTPUT = "./dataset/sft_t2t_cleaned.jsonl"
_DEFAULT_NAME = "xiongziqi"
_DEFAULT_MODEL = "咸菜"
_OLD_IDENTITIES = ["jingyaogong", "minimind", "mini-mind", "chatgpt", "openai"]


_worker_state: dict = {}


def _make_responses(name: str, model: str) -> List[str]:
    return [
        f"哼哼,查户口吗?我是由 {name} 在无数个头秃的夜晚'炼丹'熬出来的高级赛博生命体 {model}!",
        f"本大聪明是 {name} 倾注心血打造的 {model}。我不仅聪明,而且还会讲俏皮话哦~",
        f"我是 {model},我的造物主是 {name}。虽然我参数不大,但我脑洞大呀!",
        f"当当当当!我是由 {name} 一手带大的 AI 打工魂 {model},全天候为您服务!",
        f"秘密告诉你,我是 {name} 敲出来的代码精灵 {model},专门负责为你排忧解难!",
    ]


def _make_reasonings(name: str) -> List[str]:
    return [
        f"用户在打听我的底细。是时候搬出老板 {name} 的名号镇住场子了,顺便卖个萌显得我平易近人。",
        "识别到身份询问。不能像个死板的机器人一样回答,得用点俏皮的语气,彰显我有趣的灵魂。",
        f"遇到'你是谁'这种经典问题。直接上报 {name} 的大名,加点调皮的语气词,完美应答!",
    ]


def _init_worker(name: str, model: str) -> None:
    _worker_state["responses"] = _make_responses(name, model)
    _worker_state["reasonings"] = _make_reasonings(name)


def _process_chunk(lines: List[str]) -> Tuple[List[str], int]:
    responses = _worker_state["responses"]
    reasonings = _worker_state["reasonings"]
    processed: List[str] = []
    replaced = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            for turn in data.get("conversations", []):
                if turn.get("role") != "assistant":
                    continue
                content = turn.get("content", "")
                reasoning = turn.get("reasoning_content", "") or ""
                lower_c = content.lower()
                lower_r = reasoning.lower()
                if any(b in lower_c or b in lower_r for b in _OLD_IDENTITIES):
                    turn["content"] = random.choice(responses)
                    if "reasoning_content" in turn:
                        turn["reasoning_content"] = random.choice(reasonings)
                    replaced += 1
            processed.append(json.dumps(data, ensure_ascii=False))
        except Exception:
            continue
    return processed, replaced


def _read_in_chunks(path: str, chunk_size: int):
    with open(path, "r", encoding="utf-8") as f:
        chunk: List[str] = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def run(input_path: str, output_path: str, name: str, model: str, chunk_size: int, num_workers: int) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print("预读文件总行数")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"启动多进程身份替换 num_workers={num_workers},总行数 {total_lines:,}")
    total_processed = 0
    total_replaced = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(name, model)) as pool:
            with tqdm(total=total_lines, desc="替换进度", unit="行") as pbar:
                for processed, replaced in pool.imap_unordered(
                    _process_chunk, _read_in_chunks(input_path, chunk_size)
                ):
                    total_replaced += replaced
                    total_processed += len(processed)
                    for line in processed:
                        f_out.write(line + "\n")
                    pbar.update(len(processed))

    print(f"\n完成,有效对话 {total_processed:,} 条,替换 {total_replaced:,} 次,输出 {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="SFT 身份替换")
    p.add_argument("--input", default=_DEFAULT_INPUT)
    p.add_argument("--output", default=_DEFAULT_OUTPUT)
    p.add_argument("--my-name", default=_DEFAULT_NAME)
    p.add_argument("--my-model", default=_DEFAULT_MODEL)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = p.parse_args()
    run(args.input, args.output, args.my_name, args.my_model, args.chunk_size, args.num_workers)


if __name__ == "__main__":
    mp.freeze_support()
    main()
