"""训练入口

用法:
  python -m salty.run --config configs/sft.yaml
  python -m salty.run --config configs/dpo.yaml --run-id 20260425_dpo01 \
      --override lr=5e-5 --override batch_size=4

后端的 train_manager 会写一份临时 YAML,然后用 subprocess 调起本脚本。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch

from .configs import (
    BaseTrainConfig,
    DPOConfig,
    GRPOConfig,
    PretrainConfig,
    SFTConfig,
    apply_overrides,
    load_config,
    parse_config,
)
from .model import TinyLM, TinyLMConfig
from .tokenizer import load_tokenizer
from .trainers import TRAINER_REGISTRY
from .utils.ckpt_io import load_compat


def _load_state_into(model: torch.nn.Module, ckpt_path: str, device: str) -> int:
    """加载 ckpt 到 model,返回该 ckpt 对应的 iter(找不到则 0)"""
    info = load_compat(ckpt_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in info.state_dict.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[run] 加载 {ckpt_path} 时缺失键 {len(missing)} 个 (示例: {missing[:3]})")
    if unexpected:
        print(f"[run] 加载 {ckpt_path} 时意外键 {len(unexpected)} 个 (示例: {unexpected[:3]})")
    return info.iter or 0


def _build_model(cfg: BaseTrainConfig, vocab_size: int) -> TinyLM:
    m = cfg.model
    model_cfg = TinyLMConfig(
        vocab_size=vocab_size,
        dim=m.dim,
        n_layers=m.n_layers,
        n_heads=m.n_heads,
        n_kv_heads=m.n_kv_heads,
        ffn_hidden_dim=m.ffn_hidden_dim,
        max_seq_len=m.max_seq_len,
        rope_base=m.rope_base,
        dropout=m.dropout,
        norm_eps=m.norm_eps,
        use_bias=m.use_bias,
        use_moe=m.use_moe,
        num_experts=m.num_experts,
        top_k_experts=m.top_k_experts,
    )
    return TinyLM(model_cfg)


def _initial_ckpt_path(cfg: BaseTrainConfig) -> Optional[str]:
    """返回该 stage 起手要载入的 ckpt 路径(非 resume 场景)"""
    if isinstance(cfg, PretrainConfig):
        return None
    if isinstance(cfg, SFTConfig):
        return cfg.pt_ckpt or None
    if isinstance(cfg, DPOConfig):
        return cfg.ref_ckpt or None
    if isinstance(cfg, GRPOConfig):
        return cfg.base_ckpt or None
    return None


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Salty training entry")
    p.add_argument("--config", required=True, help="YAML 配置路径")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="key.path=value,可重复;value 走 yaml.safe_load",
    )
    p.add_argument("--run-id", default=None, help="本次运行 ID(默认自动生成)")
    p.add_argument("--device", default=None, help="覆盖 cfg.device")
    p.add_argument("--dry-run", action="store_true", help="只打印解析结果,不真训练")
    return p


def _resolve_config(args) -> BaseTrainConfig:
    if args.override:
        # 先 load 成 dict,叠加 override 后再走 parse_config 校验
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data = apply_overrides(data, args.override)
        cfg = parse_config(data)
    else:
        cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = _make_parser().parse_args(argv)
    cfg = _resolve_config(args)

    # 随机种子
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device != cfg.device:
        print(f"[run] CUDA 不可用,降级到 {device}")

    # 1. tokenizer
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    vocab_size = cfg.model.vocab_size or tokenizer.vocab_size

    # 2. model
    model = _build_model(cfg, vocab_size=vocab_size)

    # 3. 初始权重 / resume
    start_iter = 0
    if cfg.resume_from:
        start_iter = _load_state_into(model, cfg.resume_from, device=device)
        print(f"[run] resume from {cfg.resume_from} @ iter={start_iter}")
    else:
        init_path = _initial_ckpt_path(cfg)
        if init_path:
            _load_state_into(model, init_path, device=device)
            print(f"[run] init weights from {init_path}")

    model.to(device)

    if args.dry_run:
        from .configs import dump_config
        import json

        print(json.dumps(dump_config(cfg), indent=2, ensure_ascii=False))
        print(f"[run] dry-run 完成 (vocab={vocab_size}, params={sum(p.numel() for p in model.parameters())})")
        return 0

    # 4. 派发到 trainer
    cls = TRAINER_REGISTRY.get(cfg.stage)
    if cls is None:
        raise ValueError(f"未知 stage: {cfg.stage}")

    trainer = cls(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        device=device,
        run_id=args.run_id,
        log_dir=cfg.log_dir,
        ckpt_dir=cfg.ckpt_dir,
    )
    trainer.start_iter = start_iter
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
