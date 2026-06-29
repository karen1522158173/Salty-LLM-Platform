"""GRPO Trainer:对应旧 GRPO.py

per-iter 流程:
  1. 取一个 prompt
  2. 用 actor 模型自回归采样 K 个回答(GROUP_SIZE)
  3. 计算每个回答的奖励 + advantage(组内归一化)
  4. 重新前向算 new_logps + ref_logps
  5. ratio + clip + KL → actor_loss + beta * KL
"""
from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import RLAIFJsonlDataset
from ..rewards import calculate_grpo_reward
from ..utils.ckpt_io import load_compat
from .base import BaseTrainer


def _get_gen_logps(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    shift_logits = log_probs[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    per_token = torch.gather(shift_logits, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    gen_logps = per_token[:, prompt_len - 1 :]
    return gen_logps.sum(dim=-1)


class GRPOTrainer(BaseTrainer):
    stage = "grpo"

    def __init__(self, cfg, model, tokenizer, **kwargs):
        super().__init__(cfg, model, tokenizer, **kwargs)
        self.ref_model = self._build_ref_model()

    def _build_ref_model(self):
        ref = copy.deepcopy(self.model)
        if getattr(self.cfg, "base_ckpt", None):
            info = load_compat(self.cfg.base_ckpt, map_location=self.device)
            state = {k.replace("_orig_mod.", ""): v for k, v in info.state_dict.items()}
            ref.load_state_dict(state, strict=False)
        ref.to(self.device).to(torch.bfloat16).eval()
        for p in ref.parameters():
            p.requires_grad = False
        return ref

    def build_dataloader(self):
        ds = RLAIFJsonlDataset(self.cfg.data_path)
        return DataLoader(ds, batch_size=1, shuffle=True)

    def compute_loss(self, batch):
        prompt_text = batch["prompt"][0]
        target_text = batch["target"][0]

        input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        batched = input_ids.repeat(self.cfg.group_size, 1)

        # 1. 采样 K 条
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                batched,
                max_new_tokens=self.cfg.max_gen_len,
                temperature=self.cfg.sample_temperature,
                top_k=self.cfg.sample_top_k,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.model.train()

        gen_texts = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        # 2. 奖励 + 优势
        weights: Dict[str, float] = getattr(self.cfg, "reward_weights", None) or {}
        rewards = torch.tensor(
            [calculate_grpo_reward(t, target_text, weights) for t in gen_texts],
            device=self.device,
            dtype=torch.float32,
        )
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 3. 重新前向
        new_logits = self.model(outputs)["logits"]
        new_logps = _get_gen_logps(new_logits, outputs, prompt_len)
        with torch.no_grad():
            old_logits = self.ref_model(outputs)["logits"]
            old_logps = _get_gen_logps(old_logits, outputs, prompt_len)

        ratio = torch.exp(new_logps - old_logps)

        # Schulman 无偏 KL 估计
        log_ratio = old_logps - new_logps
        kl_div = (torch.exp(log_ratio) - log_ratio - 1.0).mean()

        clip = self.cfg.clip_eps
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        loss = actor_loss + self.cfg.beta * kl_div

        broken_rate = (rewards < -1.0).float().mean()
        return loss, {
            "loss": loss.detach(),
            "actor_loss": actor_loss.detach(),
            "kl": kl_div.detach(),
            "reward": rewards.mean().detach(),
            "broken_rate": broken_rate.detach(),
        }
