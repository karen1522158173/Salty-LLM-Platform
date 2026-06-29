"""模型配置(从 model/miniLLM.py 抽出)"""
from dataclasses import dataclass


@dataclass
class TinyLMConfig:
    vocab_size: int
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden_dim: int = 1536

    max_seq_len: int = 2048
    rope_base: float = 10000.0

    dropout: float = 0.0
    norm_eps: float = 1e-6
    use_bias: bool = False

    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 1
