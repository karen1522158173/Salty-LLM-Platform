"""训练指标记录:同时写 TensorBoard 和 JSONL(供后端 tail)"""
import json
import os
import time
from typing import Any, Dict, Optional


class MetricsLogger:
    """训练指标记录器

    同时写 TensorBoard SummaryWriter 和 `runs/{run_id}/metrics.jsonl`。
    后端通过 tail 这个 JSONL 把指标推到前端 WebSocket。
    """

    def __init__(
        self,
        log_dir: str,
        run_id: Optional[str] = None,
        enable_tb: bool = True,
        flush_every: int = 1,
    ):
        self.log_dir = log_dir
        self.run_id = run_id
        self.flush_every = flush_every
        self._counter = 0

        os.makedirs(log_dir, exist_ok=True)
        self._jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self._jsonl = open(self._jsonl_path, "a", encoding="utf-8", buffering=1)

        self._tb = None
        if enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=log_dir)
            except Exception:
                self._tb = None

    def emit(self, metrics: Dict[str, Any], iter: int, stage: str = ""):
        """写一行指标。metrics 是扁平 dict,值要是数字或字符串"""
        record = {
            "ts": time.time(),
            "iter": iter,
            "stage": stage,
            **metrics,
        }
        if self.run_id:
            record["run_id"] = self.run_id

        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._counter += 1
        if self._counter % self.flush_every == 0:
            self._jsonl.flush()

        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, iter)

    def log_text(self, message: str, level: str = "info"):
        """写一行文本日志(给前端日志台用)"""
        record = {
            "ts": time.time(),
            "type": "log",
            "level": level,
            "message": message,
        }
        if self.run_id:
            record["run_id"] = self.run_id
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def close(self):
        try:
            self._jsonl.close()
        except Exception:
            pass
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
