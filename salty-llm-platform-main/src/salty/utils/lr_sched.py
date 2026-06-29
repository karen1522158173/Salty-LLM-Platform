"""学习率调度器"""
import math


def cosine_with_warmup(it: int, *, warmup: int, max_iters: int, lr: float, min_lr: float = 1e-6) -> float:
    """warmup 线性 + cosine 衰减,与 train_sft.py / train.py 中的写法一致"""
    if it < warmup:
        return lr * (it + 1) / max(1, warmup)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)
