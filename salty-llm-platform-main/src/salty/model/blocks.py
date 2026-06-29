"""模型基本组件:RMSNorm / RoPE / FFN / GQA Attention / TransformerBlock"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def get_cos_sin(self, seq_len: int, device, dtype):
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack([-x_odd, x_even], dim=-1)
    return x_rot.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class DenseSwiGLU(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        dim = config.dim
        hidden = config.ffn_hidden_dim
        self.up_proj = nn.Linear(dim, hidden, bias=config.use_bias)
        self.gate_proj = nn.Linear(dim, hidden, bias=config.use_bias)
        self.down_proj = nn.Linear(hidden, dim, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = gate * up
        out = self.down_proj(out)
        out = self.dropout(out)
        return out


class SimpleMoEPlaceholder(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.dense_ffn = DenseSwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_ffn(x)


def build_ffn(config: TinyLMConfig) -> nn.Module:
    if config.use_moe:
        return SimpleMoEPlaceholder(config)
    return DenseSwiGLU(config)


class CausalSelfAttention(nn.Module):
    """GQA + SDPA(FlashAttention-2)"""

    def __init__(self, config: TinyLMConfig):
        super().__init__()
        assert config.dim % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.use_bias)

        self.dropout_p = config.dropout
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, kvH, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, kvH, self.n_rep, T, D)
        return x.reshape(B, kvH * self.n_rep, T, D)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)

        cos, sin = self.rope.get_cos_sin(past_len + T, x.device, x.dtype)
        cos = cos[past_len:past_len + T]
        sin = sin[past_len:past_len + T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        is_causal = (past_len == 0)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)

        if self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p, training=self.training)

        return out, present


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn = build_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        attn_out, present = self.attn(self.attn_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, present
