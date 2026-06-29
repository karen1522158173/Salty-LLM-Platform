"""模型包入口"""
from .config import TinyLMConfig
from .model import TinyLM, count_parameters

__all__ = ["TinyLMConfig", "TinyLM", "count_parameters"]
