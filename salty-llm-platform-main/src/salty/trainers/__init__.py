"""训练器模块"""
from .base import BaseTrainer
from .pretrain import PretrainTrainer
from .sft import SFTTrainer
from .dpo import DPOTrainer
from .grpo import GRPOTrainer

TRAINER_REGISTRY = {
    "pretrain": PretrainTrainer,
    "sft": SFTTrainer,
    "dpo": DPOTrainer,
    "grpo": GRPOTrainer,
}

__all__ = [
    "BaseTrainer",
    "PretrainTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "TRAINER_REGISTRY",
]
