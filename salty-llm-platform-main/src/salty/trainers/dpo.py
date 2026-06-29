"""DPO Trainer:对应旧 train_DPO.py

需要一个 ref_model(和 actor 同结构,加载同样的初始权重,冻结)。
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import DPOJsonlDataset
from ..utils.ckpt_io import load_compat
from .base import BaseTrainer


def _get_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    shift_logits = log_probs[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token = torch.gather(
        shift_logits, dim=2, index=shift_labels.unsqueeze(2).clamp(min=0)
    ).squeeze(2)
    return (per_token * (shift_labels != -100)).sum(-1)


class DPOTrainer(BaseTrainer):
    stage = "dpo"

    def __init__(self, cfg, model, tokenizer, **kwargs):
        super().__init__(cfg, model, tokenizer, **kwargs)
        self.ref_model = self._build_ref_model()

    def _build_ref_model(self):
        ref = copy.deepcopy(self.model)
        if getattr(self.cfg, "ref_ckpt", None):
            info = load_compat(self.cfg.ref_ckpt, map_location=self.device)
            state = {k.replace("_orig_mod.", ""): v for k, v in info.state_dict.items()}
            ref.load_state_dict(state, strict=False)
        ref.to(self.device).eval()
        for p in ref.parameters():
            p.requires_grad = False
        return ref

    def build_dataloader(self):
        ds = DPOJsonlDataset(
            file_path=self.cfg.data_path,
            tokenizer=self.tokenizer,
            max_len=self.cfg.max_seq_len,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=getattr(self.cfg, "num_workers", 0),
            pin_memory=True,
        )

    def compute_loss(self, batch):
        c_ids = batch["c_ids"].to(self.device)
        c_labs = batch["c_labs"].to(self.device)
        r_ids = batch["r_ids"].to(self.device)
        r_labs = batch["r_labs"].to(self.device)

        beta = self.cfg.beta

        p_c = _get_logps(self.model(c_ids)["logits"], c_labs)
        p_r = _get_logps(self.model(r_ids)["logits"], r_labs)
        with torch.no_grad():
            ref_c = _get_logps(self.ref_model(c_ids)["logits"], c_labs)
            ref_r = _get_logps(self.ref_model(r_ids)["logits"], r_labs)

        pi_log = p_c - p_r
        ref_log = ref_c - ref_r
        logits = pi_log - ref_log
        margin = (p_c - ref_c).mean() - (p_r - ref_r).mean()

        loss = -F.logsigmoid(beta * logits).mean()
        acc = (logits > 0).float().mean()

        return loss, {
            "loss": loss.detach(),
            "acc": acc.detach(),
            "margin": margin.detach(),
        }
