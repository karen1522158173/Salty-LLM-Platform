"""SFT Trainer:对应旧 train_sft.py"""
from __future__ import annotations

from ..data import SFTJsonlDataset, build_sft_dataset_mp
from .base import BaseTrainer


class SFTTrainer(BaseTrainer):
    stage = "sft"

    def build_dataloader(self):
        items = build_sft_dataset_mp(
            file_path=self.cfg.data_path,
            tokenizer_path=self.cfg.tokenizer_path,
            max_seq_len=self.cfg.max_seq_len,
            num_workers=getattr(self.cfg, "num_workers", None),
        )
        pad_id = self.tokenizer.eos_token_id or 0
        return SFTJsonlDataset(
            items=items,
            pad_id=pad_id,
            batch_size=self.cfg.batch_size,
            device=self.device,
        )

    def compute_loss(self, batch):
        x, y = batch
        out = self.model(x, labels=y)
        loss = out["loss"]
        return loss, {"loss": loss.detach()}
