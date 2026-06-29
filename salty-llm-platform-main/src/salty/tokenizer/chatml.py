"""ChatML 模板封装与 think 段落识别"""
from typing import Iterator, List, Tuple


def apply_chatml(messages: List[dict], add_generation_prompt: bool = True) -> str:
    """把 [{role, content}, ...] 包成 ChatML 模板字符串

    Args:
        messages: 消息列表,每条形如 {"role": "user"|"assistant"|"system", "content": "..."}
        add_generation_prompt: 末尾是否追加 `<|im_start|>assistant\n`(用于触发模型续写)
    """
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def split_think(text: str) -> List[Tuple[str, str]]:
    """把含 <think>...</think> 的文本切成 [(kind, segment)] 列表

    kind 取值: "think" / "content"。比正则切片更稳:即使 <think> 没闭合,
    也会把后面所有内容标记为 think。
    """
    segments: List[Tuple[str, str]] = []
    cur = 0
    in_think = False
    buf = []

    while cur < len(text):
        if not in_think:
            idx = text.find("<think>", cur)
            if idx == -1:
                buf.append(text[cur:])
                break
            if idx > cur:
                segments.append(("content", text[cur:idx]))
            buf = []
            in_think = True
            cur = idx + len("<think>")
        else:
            idx = text.find("</think>", cur)
            if idx == -1:
                segments.append(("think", text[cur:]))
                buf = []
                cur = len(text)
                break
            segments.append(("think", text[cur:idx]))
            in_think = False
            cur = idx + len("</think>")

    if buf:
        segments.append(("content", "".join(buf)))
    return segments


class ThinkStreamSplitter:
    """用于流式生成时的 <think> 边界检测状态机

    用法:
        splitter = ThinkStreamSplitter()
        for delta in stream:
            for evt in splitter.feed(delta):
                yield evt          # {"type": "think_open"|"think_close"|"delta", "text": ...}
        for evt in splitter.flush():
            yield evt
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, delta: str) -> Iterator[dict]:
        """喂入新增文本片段,生成事件

        中文 BPE 可能把 `<think>` 切成 `<th` + `ink>` 两个 token,所以必须
        在缓冲区累积后再判断。一旦缓冲区里有完整的 tag,立即触发事件并冲掉。
        """
        self._buf += delta
        while True:
            tag = self.CLOSE_TAG if self._in_think else self.OPEN_TAG
            idx = self._buf.find(tag)
            if idx == -1:
                # 没找到完整 tag。检查 buf 末尾是否是 tag 的前缀,
                # 是的话保留在 buf 里,其余 emit 出去。
                safe = self._safe_emit_len(self._buf, tag)
                if safe > 0:
                    text = self._buf[:safe]
                    self._buf = self._buf[safe:]
                    yield self._make_delta(text)
                return
            # 找到 tag,先 emit tag 之前的内容
            if idx > 0:
                yield self._make_delta(self._buf[:idx])
            self._buf = self._buf[idx + len(tag):]
            if self._in_think:
                self._in_think = False
                yield {"type": "think_close", "text": ""}
            else:
                self._in_think = True
                yield {"type": "think_open", "text": ""}

    def flush(self) -> Iterator[dict]:
        """流结束时清空缓冲区"""
        if self._buf:
            yield self._make_delta(self._buf)
            self._buf = ""

    def _make_delta(self, text: str) -> dict:
        kind = "think_delta" if self._in_think else "delta"
        return {"type": kind, "text": text}

    @staticmethod
    def _safe_emit_len(buf: str, tag: str) -> int:
        """返回可以安全 emit 的长度。buf 末尾如果是 tag 的前缀,要保留。"""
        max_keep = min(len(tag) - 1, len(buf))
        for k in range(max_keep, 0, -1):
            if tag.startswith(buf[-k:]):
                return len(buf) - k
        return len(buf)
