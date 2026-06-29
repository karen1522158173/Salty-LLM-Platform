"""预训练 Trainer:对应旧 train.py"""
from __future__ import annotations

from ..data import PretrainBinSampler
from .base import BaseTrainer


class PretrainTrainer(BaseTrainer):
    stage = "pretrain"

    def build_dataloader(self):
        return PretrainBinSampler(
            bin_path=self.cfg.bin_path,
            seq_len=self.cfg.seq_len,
            batch_size=self.cfg.batch_size,
            device=self.device,
        )

    def compute_loss(self, batch):
        x, y = batch
        out = self.model(x, labels=y)
        loss = out["loss"]
        return loss, {"loss": loss.detach()}
