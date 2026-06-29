"""训练配置(Pydantic v2 + YAML)

每个阶段都有自己的子类,所有阶段共享 BaseTrainConfig。
前端表单按 Pydantic schema 自动渲染,提交时 POST 整份 YAML 给后端。
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ============================================================
# 模型结构配置(等价于 TinyLMConfig,用于训练时实例化)
# ============================================================
class ModelConfig(BaseModel):
    vocab_size: Optional[int] = None         # 由 tokenizer 决定
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


# ============================================================
# 训练通用基类
# ============================================================
class BaseTrainConfig(BaseModel):
    stage: Literal["pretrain", "sft", "dpo", "grpo"]

    # 路径
    tokenizer_path: str = "./minimind_tokenizer"
    ckpt_dir: str
    log_dir: str
    resume_from: Optional[str] = None

    # 通用超参
    seed: int = 42
    device: str = "cuda"
    max_iters: int = 1000
    warmup: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    batch_size: int = 2
    grad_accum: int = 8
    grad_clip: float = 1.0
    save_interval: int = 500
    eval_interval: int = 500
    log_every: int = 10
    use_amp: bool = True
    amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"

    # 模型结构
    model: ModelConfig = Field(default_factory=ModelConfig)


class PretrainConfig(BaseTrainConfig):
    stage: Literal["pretrain"] = "pretrain"
    bin_path: str = "./data/bin/train_data.bin"
    seq_len: int = 512


class SFTConfig(BaseTrainConfig):
    stage: Literal["sft"] = "sft"
    data_path: str = "./data/raw/sft_t2t_mini_cleaned.jsonl"
    pt_ckpt: str = "./checkpoints/pretrain/step_134500.pt"
    max_seq_len: int = 2048
    num_workers: Optional[int] = None


class DPOConfig(BaseTrainConfig):
    stage: Literal["dpo"] = "dpo"
    data_path: str = "./data/raw/dpo.jsonl"
    ref_ckpt: str = "./checkpoints/sft/sft_step_28500.pt"
    beta: float = 0.1
    smoothing: float = 0.95
    num_workers: int = 4
    max_seq_len: int = 2048


class GRPOConfig(BaseTrainConfig):
    stage: Literal["grpo"] = "grpo"
    data_path: str = "./data/raw/rlaif.jsonl"
    base_ckpt: str = "./checkpoints/dpo/dpo_step_2000.pt"
    group_size: int = 4
    beta: float = 0.02
    clip_eps: float = 0.2
    max_gen_len: int = 2048
    sample_temperature: float = 0.7
    sample_top_k: int = 20
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "close_tag": 1.0,
            "len_ok": 0.5,
            "diversity": 2.0,
            "ngram_repeat": 1.5,
            "logic_chain": 0.5,
            "answer_match": 2.0,
            "no_close_tag": 2.5,
            "answer_miss": 0.5,
        }
    )


# ============================================================
# YAML 加载/保存
# ============================================================
_STAGE_TO_CLS = {
    "pretrain": PretrainConfig,
    "sft": SFTConfig,
    "dpo": DPOConfig,
    "grpo": GRPOConfig,
}


def load_config(path: str) -> BaseTrainConfig:
    """从 YAML 文件加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_config(data)


def parse_config(data: dict) -> BaseTrainConfig:
    stage = data.get("stage")
    if stage not in _STAGE_TO_CLS:
        raise ValueError(f"未知 stage: {stage!r},应为 {list(_STAGE_TO_CLS)}")
    return _STAGE_TO_CLS[stage].model_validate(data)


def dump_config(cfg: BaseTrainConfig) -> dict:
    return cfg.model_dump()


def stage_schema(stage: str) -> dict:
    """返回某 stage 配置的 JSON Schema(给前端表单用)"""
    cls = _STAGE_TO_CLS.get(stage)
    if cls is None:
        raise ValueError(f"未知 stage: {stage}")
    return cls.model_json_schema()


def apply_overrides(cfg_dict: dict, overrides: List[str]) -> dict:
    """支持 CLI --override key.path=value,key 用 . 分隔嵌套"""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"--override 格式错误: {ov}(应为 key=value)")
        key, value = ov.split("=", 1)
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        keys = key.strip().split(".")
        d = cfg_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg_dict
