"""推理引擎:KV cache 流式生成 + think 状态机 + 取消机制

同步迭代器(`stream`)和异步生成器(`astream`)两种风格,后端 WebSocket 用 astream。
"""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, List, Optional

import torch

from ..model import TinyLM, TinyLMConfig
from ..tokenizer import ThinkStreamSplitter, apply_chatml, load_tokenizer
from ..utils.ckpt_io import load_compat
from .sampler import SamplingParams, apply_repetition_penalty, sample_next_token


@dataclass
class StreamEvent:
    type: str           # "delta" | "think_open" | "think_delta" | "think_close" | "done" | "error"
    text: str = ""
    meta: Optional[dict] = None


class InferenceEngine:
    """单例化的推理引擎,被 backend.services.inference_manager 持有"""

    def __init__(
        self,
        tokenizer_path: str = "./minimind_tokenizer",
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.model: Optional[TinyLM] = None
        self.ckpt_path: Optional[str] = None
        self._cancel = False
        self._model_kwargs = model_kwargs or {}

    # ----------------------------------------------------------
    # 加载 / 卸载
    # ----------------------------------------------------------
    def load(self, ckpt_path: str) -> dict:
        kwargs = dict(
            vocab_size=len(self.tokenizer),
            dim=512, n_layers=6, n_heads=8, n_kv_heads=2,
            ffn_hidden_dim=1536, max_seq_len=2048,
        )
        kwargs.update(self._model_kwargs)
        config = TinyLMConfig(**kwargs)
        model = TinyLM(config).to(self.device)

        info = load_compat(ckpt_path, map_location=self.device)
        # 兼容 torch.compile 后的前缀
        sd = {k.replace("_orig_mod.", ""): v for k, v in info.state_dict.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()

        self.model = model
        self.ckpt_path = ckpt_path
        return {
            "ckpt_path": ckpt_path,
            "stage": info.stage,
            "iter": info.iter,
            "schema_version": info.schema_version,
            "device": self.device,
        }

    def unload(self):
        self.model = None
        self.ckpt_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def cancel(self):
        self._cancel = True

    # ----------------------------------------------------------
    # 同步流式生成(线程内调用)
    # ----------------------------------------------------------
    @torch.no_grad()
    def stream(
        self,
        messages: List[dict],
        params: Optional[SamplingParams] = None,
        use_kv_cache: bool = True,
    ) -> Iterator[StreamEvent]:
        if self.model is None:
            raise RuntimeError("模型未加载,请先调用 load(ckpt_path)")
        params = params or SamplingParams()
        self._cancel = False

        prompt = apply_chatml(messages, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_id = self.tokenizer.eos_token_id

        splitter = ThinkStreamSplitter()
        generated_ids: List[int] = []
        prev_text = ""
        past_kv = None
        cur_input = input_ids
        history_ids = input_ids[0].tolist()

        for _ in range(params.max_tokens):
            if self._cancel:
                yield StreamEvent("error", text="cancelled")
                return

            if use_kv_cache:
                outputs = self.model(cur_input, past_key_values=past_kv, use_cache=True)
                past_kv = outputs["past_key_values"]
                cur_input = None  # 占位,后面会赋值
            else:
                outputs = self.model(cur_input, use_cache=False)

            logits = outputs["logits"][:, -1, :].clone()
            logits = apply_repetition_penalty(logits, history_ids, params.rep_penalty)
            next_token = sample_next_token(logits, params)
            tid = int(next_token.item())

            if tid in (im_end_id, eos_id):
                break

            generated_ids.append(tid)
            history_ids.append(tid)

            if use_kv_cache:
                cur_input = next_token
            else:
                cur_input = torch.cat([cur_input, next_token], dim=1)

            current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            if "\ufffd" in current_text:
                continue
            new_text = current_text[len(prev_text):]
            prev_text = current_text

            if new_text:
                for evt in splitter.feed(new_text):
                    yield StreamEvent(type=evt["type"], text=evt["text"])

        for evt in splitter.flush():
            yield StreamEvent(type=evt["type"], text=evt["text"])
        yield StreamEvent(type="done", text="", meta={"tokens": len(generated_ids)})

    # ----------------------------------------------------------
    # 异步流式生成(WebSocket 用)
    # ----------------------------------------------------------
    async def astream(
        self,
        messages: List[dict],
        params: Optional[SamplingParams] = None,
    ) -> AsyncIterator[StreamEvent]:
        """把 sync stream 包成 async,每个 token 后让出事件循环"""
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def producer():
            try:
                for evt in self.stream(messages, params):
                    asyncio.run_coroutine_threadsafe(queue.put(evt), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put(StreamEvent("error", text=str(e))), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

        await loop.run_in_executor(None, producer)
        while True:
            evt = await queue.get()
            if evt is sentinel:
                break
            yield evt
