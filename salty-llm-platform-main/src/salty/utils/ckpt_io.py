"""Checkpoint I/O 兼容层

支持三种格式:
1. 旧裸 state_dict (`train.py`/`train_DPO.py`/`GRPO.py` 产出)
2. 旧含 iter 字典 (`train_sft.py` 产出): {"iter": int, "model_state_dict": dict}
3. 统一新格式(schema_version=2): {model_state_dict, optimizer_state_dict?, iter, stage, cfg_snapshot, schema_version}

加载侧永远走 `load_compat`,保存一律用 `save_unified` 写新格式。
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


SCHEMA_VERSION = 2


@dataclass
class CkptInfo:
    state_dict: Dict[str, torch.Tensor]
    iter: int = 0
    stage: str = "unknown"          # pretrain / sft / dpo / grpo / unknown
    cfg_snapshot: Optional[dict] = None
    optimizer_state_dict: Optional[dict] = None
    schema_version: int = 0
    raw: Optional[dict] = field(default=None, repr=False)


def _is_state_dict(d: dict) -> bool:
    """判断 dict 是否是裸 state_dict(键全是 str 且值全是 Tensor)"""
    if not d:
        return False
    for k, v in d.items():
        if not isinstance(k, str):
            return False
        if not torch.is_tensor(v):
            return False
    return True


def load_compat(path: str, map_location: str = "cpu") -> CkptInfo:
    """加载任意格式的 checkpoint,返回统一的 CkptInfo"""
    raw = torch.load(path, map_location=map_location)

    # 格式 3: 新统一格式
    if isinstance(raw, dict) and raw.get("schema_version") == SCHEMA_VERSION:
        return CkptInfo(
            state_dict=raw["model_state_dict"],
            iter=raw.get("iter", 0),
            stage=raw.get("stage", "unknown"),
            cfg_snapshot=raw.get("cfg_snapshot"),
            optimizer_state_dict=raw.get("optimizer_state_dict"),
            schema_version=SCHEMA_VERSION,
            raw=raw,
        )

    # 格式 2: 含 iter + model_state_dict 的旧 SFT 格式
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return CkptInfo(
            state_dict=raw["model_state_dict"],
            iter=raw.get("iter", 0),
            stage=raw.get("stage", _guess_stage_from_path(path)),
            cfg_snapshot=raw.get("cfg_snapshot"),
            optimizer_state_dict=raw.get("optimizer_state_dict"),
            schema_version=1,
            raw=raw,
        )

    # 格式 1: 裸 state_dict
    if isinstance(raw, dict) and _is_state_dict(raw):
        return CkptInfo(
            state_dict=raw,
            iter=_guess_iter_from_path(path),
            stage=_guess_stage_from_path(path),
            schema_version=0,
            raw={"_legacy_bare": True},
        )

    raise ValueError(f"无法识别的 checkpoint 格式: {path}")


def save_unified(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    iter: int,
    stage: str,
    optimizer_state_dict: Optional[dict] = None,
    cfg_snapshot: Optional[dict] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """以统一新格式保存 checkpoint"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": state_dict,
        "iter": iter,
        "stage": stage,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if cfg_snapshot is not None:
        payload["cfg_snapshot"] = cfg_snapshot
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def inspect_ckpt(path: str) -> dict:
    """快速窥探 checkpoint 元信息(不加载到 GPU)"""
    info = load_compat(path, map_location="cpu")
    return {
        "path": path,
        "stage": info.stage,
        "iter": info.iter,
        "schema_version": info.schema_version,
        "num_params": sum(t.numel() for t in info.state_dict.values()),
        "size_bytes": os.path.getsize(path),
    }


def _guess_stage_from_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "grpo" in p:
        return "grpo"
    if "dpo" in p:
        return "dpo"
    if "sft" in p:
        return "sft"
    if "pretrain" in p or "/checkpoints/" in p or p.endswith(".pt"):
        return "pretrain"
    return "unknown"


def _guess_iter_from_path(path: str) -> int:
    """从形如 step_134500.pt / sft_step_22000.pt / dpo_step_2000.pt 中抽取数字"""
    base = os.path.basename(path).lower()
    digits = ""
    for ch in base:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else 0
