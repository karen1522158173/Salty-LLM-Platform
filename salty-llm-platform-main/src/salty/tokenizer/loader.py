"""Tokenizer 加载封装"""
import os

from transformers import AutoTokenizer


_DEFAULT_PATH = os.environ.get("SALTY_TOKENIZER", "./minimind_tokenizer")


def load_tokenizer(path: str = None):
    """加载 HF 风格的 tokenizer 目录"""
    if path is None:
        path = _DEFAULT_PATH
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
