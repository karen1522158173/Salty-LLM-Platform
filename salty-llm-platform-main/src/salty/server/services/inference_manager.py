"""推理服务单例:管理模型加载/卸载 + 提供流式生成"""
import asyncio
from typing import AsyncIterator, List, Optional

from ...inference.engine import InferenceEngine, StreamEvent
from ...inference.sampler import SamplingParams
from ..db import CheckpointDAO, CheckpointRow


class InferenceManager:
    """全局单例,被 FastAPI lifespan 初始化"""

    def __init__(self, tokenizer_path: str = "./minimind_tokenizer", device: Optional[str] = None):
        self.engine = InferenceEngine(tokenizer_path=tokenizer_path, device=device)
        self._lock = asyncio.Lock()

    async def load(self, ckpt_path: str) -> dict:
        async with self._lock:
            # 在线程池里跑阻塞的 torch.load
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, self.engine.load, ckpt_path)
            return info

    async def unload(self):
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.engine.unload)

    def is_loaded(self) -> bool:
        return self.engine.is_loaded

    async def chat(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_k: int = 10,
        top_p: Optional[float] = None,
        rep_penalty: float = 1.25,
    ) -> AsyncIterator[StreamEvent]:
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
        )
        async for evt in self.engine.astream(messages, params):
            yield evt

    def cancel(self):
        self.engine.cancel()


# 全局实例,在 main.py lifespan 中注入
inference_mgr: Optional[InferenceManager] = None


def set_inference_mgr(mgr: InferenceManager):
    global inference_mgr
    inference_mgr = mgr


def get_inference_mgr() -> InferenceManager:
    if inference_mgr is None:
        raise RuntimeError("InferenceManager 未初始化")
    return inference_mgr
