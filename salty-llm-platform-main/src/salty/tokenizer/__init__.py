"""Tokenizer 工具:ChatML 模板 + 加载器(loader 懒导入,避免在没装 transformers 时连 chatml 都用不了)"""
from .chatml import ThinkStreamSplitter, apply_chatml, split_think

__all__ = ["apply_chatml", "split_think", "ThinkStreamSplitter", "load_tokenizer"]


def load_tokenizer(path: str = None):
    from .loader import load_tokenizer as _impl
    return _impl(path)
