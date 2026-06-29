"""TinyLM 主模型"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm, TransformerBlock
from .config import TinyLMConfig


class TinyLM(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"序列长度 {T} 超过 max_seq_len={self.config.max_seq_len}"
        )

        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        presents = [] if use_cache else None
        for layer, past_kv in zip(self.layers, past_key_values):
            x, present = layer(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "past_key_values": presents}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                out = self(generated, use_cache=True)
            else:
                out = self(generated[:, -1:], past_key_values=past_key_values, use_cache=True)

            logits = out["logits"][:, -1, :]
            past_key_values = out["past_key_values"]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    min_topk = values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < min_topk,
                        torch.full_like(logits, -float("inf")),
                        logits,
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if generated.size(1) >= self.config.max_seq_len:
                break

        return generated


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
