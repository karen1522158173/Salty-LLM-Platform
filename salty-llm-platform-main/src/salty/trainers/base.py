"""BaseTrainer:四阶段训练通用循环

子类只需要实现:
  - `build_dataloader()`: 准备数据(可以返回 sampler 或 iterator)
  - `compute_loss(batch)` → (loss_tensor, metrics_dict): 还未除以 grad_accum 的 loss
  - 可选 `lr_for(it)`:返回该步学习率(默认走 cosine warmup)
  - 可选 `extra_ckpt_payload()`:补充保存到 ckpt 的字段
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterator, Optional

import torch

from ..utils.ckpt_io import save_unified
from ..utils.logging import MetricsLogger
from ..utils.lr_sched import cosine_with_warmup


class BaseTrainer:
    stage: str = "base"

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cuda",
        run_id: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
        log_dir: Optional[str] = None,
        ckpt_dir: Optional[str] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.run_id = run_id
        self.stop_event = stop_event or threading.Event()

        self.log_dir = log_dir or getattr(cfg, "log_dir", f"runs/{self.stage}")
        self.ckpt_dir = ckpt_dir or getattr(cfg, "ckpt_dir", f"checkpoints/{self.stage}")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = MetricsLogger(log_dir=self.log_dir, run_id=run_id)

        self.optimizer = self.build_optimizer()
        self.scaler = self.build_scaler()
        self.dataloader = self.build_dataloader()

        self.start_iter: int = 0
        self.it: int = 0

    # --- 子类必填 ---
    def build_dataloader(self):
        raise NotImplementedError

    def compute_loss(self, batch) -> tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    # --- 子类可选 ---
    def build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.cfg, "lr", 1e-4),
            weight_decay=getattr(self.cfg, "weight_decay", 0.0),
            betas=getattr(self.cfg, "betas", (0.9, 0.95)),
        )

    def build_scaler(self):
        if self.device.startswith("cuda") and getattr(self.cfg, "use_amp", True):
            try:
                return torch.amp.GradScaler("cuda")
            except TypeError:
                return torch.cuda.amp.GradScaler()
        return None

    def autocast_ctx(self):
        if self.device.startswith("cuda") and getattr(self.cfg, "use_amp", True):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def lr_for(self, it: int) -> float:
        warmup = getattr(self.cfg, "warmup", 0)
        max_iters = getattr(self.cfg, "max_iters", 1)
        lr = getattr(self.cfg, "lr", 1e-4)
        min_lr = getattr(self.cfg, "min_lr", 0.0)
        return cosine_with_warmup(it, warmup, max_iters, lr, min_lr)

    def extra_ckpt_payload(self) -> Dict[str, Any]:
        return {}

    # --- 通用循环 ---
    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _check_stop(self) -> bool:
        if self.stop_event.is_set():
            return True
        stop_file = os.path.join(self.log_dir, "STOP")
        if os.path.exists(stop_file):
            self.stop_event.set()
            return True
        return False

    def _save(self, it: int, suffix: str = "") -> str:
        name = f"{self.stage}_step_{it}{suffix}.pt"
        path = os.path.join(self.ckpt_dir, name)
        cfg_snapshot = None
        try:
            if hasattr(self.cfg, "model_dump"):
                cfg_snapshot = self.cfg.model_dump()
        except Exception:
            cfg_snapshot = None
        save_unified(
            path=path,
            state_dict=self.model.state_dict(),
            iter=it,
            stage=self.stage,
            optimizer_state_dict=self.optimizer.state_dict() if getattr(self.cfg, "save_optimizer", False) else None,
            cfg_snapshot=cfg_snapshot,
            extra=self.extra_ckpt_payload(),
        )
        return path

    def _iter_data(self) -> Iterator:
        """把 dataloader 转成无限迭代器(子类可以重写)。

        若 dataloader 自身是 sampler(实现了 get_batch),走 get_batch;
        否则当作 PyTorch DataLoader 反复迭代。
        """
        if hasattr(self.dataloader, "get_batch"):
            while True:
                yield self.dataloader.get_batch()
        else:
            while True:
                for batch in self.dataloader:
                    yield batch

    def fit(self) -> None:
        cfg = self.cfg
        max_iters = cfg.max_iters
        log_every = getattr(cfg, "log_every", 10)
        save_interval = getattr(cfg, "save_interval", 1000)
        grad_accum = max(1, getattr(cfg, "grad_accum", 1))
        clip_grad = getattr(cfg, "grad_clip", 0.0)

        data_iter = self._iter_data()
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        last_log_iter = self.start_iter

        self.logger.log_text(f"开始训练 stage={self.stage} max_iters={max_iters} grad_accum={grad_accum}")

        for it in range(self.start_iter, max_iters):
            self.it = it

            if self._check_stop():
                self.logger.log_text(f"收到停止信号,iter={it}")
                break

            lr = self.lr_for(it)
            self._set_lr(lr)

            self.model.train()
            accum_metrics: Dict[str, float] = {}
            for accum_step in range(grad_accum):
                batch = next(data_iter)
                with self.autocast_ctx():
                    loss, metrics = self.compute_loss(batch)
                    scaled_loss = loss / grad_accum
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v) / grad_accum

            if clip_grad and clip_grad > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if it % log_every == 0:
                dt = time.time() - t0
                steps = max(1, it - last_log_iter)
                payload = {
                    **accum_metrics,
                    "lr": lr,
                    "iter_per_sec": steps / dt if dt > 0 else 0.0,
                }
                self.logger.emit(payload, iter=it, stage=self.stage)
                self.logger.log_text(
                    f"iter={it} " + " ".join(f"{k}={v:.4f}" for k, v in payload.items() if isinstance(v, (int, float)))
                )
                t0 = time.time()
                last_log_iter = it

            if save_interval > 0 and it > 0 and it % save_interval == 0:
                path = self._save(it)
                self.logger.log_text(f"保存 checkpoint: {path}")

        # 最终落盘
        final_path = self._save(self.it, suffix="_final")
        self.logger.log_text(f"训练结束,final ckpt: {final_path}")
        self.logger.close()
