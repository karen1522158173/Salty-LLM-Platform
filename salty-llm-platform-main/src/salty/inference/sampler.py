"""采样策略:温度、top_k、top_p、重复惩罚"""
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_tokens: int = 1024
    temperature: float = 0.3
    top_k: Optional[int] = 10
    top_p: Optional[float] = None
    rep_penalty: float = 1.25


def apply_repetition_penalty(
    logits: torch.Tensor,
    history_ids: Iterable[int],
    rep_penalty: float,
) -> torch.Tensor:
    """与旧 inference_sft.py 完全一致的重复惩罚实现:
    历史出现过的 token,正 logit 除以 rep_penalty,负 logit 乘以 rep_penalty。
    logits 形状 [1, vocab]。
    """
    if rep_penalty == 1.0:
        return logits
    seen = set(int(t) for t in history_ids)
    for tid in seen:
        if logits[0, tid] > 0:
            logits[0, tid] = logits[0, tid] / rep_penalty
        else:
            logits[0, tid] = logits[0, tid] * rep_penalty
    return logits


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
) -> torch.Tensor:
    """从单步 logits [1, vocab] 中采样一个 token,返回 [1, 1]"""
    if params.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / params.temperature

    if params.top_k is not None:
        k = min(params.top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        threshold = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.full_like(logits, -float("inf")), logits)

    if params.top_p is not None and 0 < params.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        mask = cumulative > params.top_p
        # 至少保留第一个
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
