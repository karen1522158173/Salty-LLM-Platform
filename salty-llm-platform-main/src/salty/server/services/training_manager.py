"""训练任务管理:启动子进程 + 轮询日志更新数据库状态"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ...configs import load_config
from ..db import TrainingJobDAO, TrainingJobRow


class TrainingManager:
    def __init__(self, python_exe: Optional[str] = None):
        self.python_exe = python_exe or sys.executable
        self._running: Dict[str, subprocess.Popen] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------
    def start(
        self,
        run_id: str,
        config_path: str,
        overrides: Optional[Dict[str, str]] = None,
    ) -> str:
        if TrainingJobDAO.get_by_run_id(run_id):
            raise ValueError(f"run_id {run_id} 已存在")

        log_dir = Path("./runs") / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "train.log")

        cfg = load_config(config_path)
        max_iter = getattr(cfg, "max_iters", None)

        # 数据库记录
        job = TrainingJobRow(
            id=0, run_id=run_id, stage=cfg.stage, config_path=config_path,
            overrides=json.dumps(overrides, ensure_ascii=False) if overrides else None,
            status="running", pid=None, start_time=datetime.now().isoformat(),
            end_time=None, current_iter=0, max_iter=max_iter, loss=None,
            log_file=log_file, error_msg=None, created_at="",
        )
        TrainingJobDAO.insert(job)

        # 构建命令
        cmd = [self.python_exe, "-m", "salty.run", "--config", config_path, "--run-id", run_id]
        if overrides:
            for k, v in overrides.items():
                cmd += ["--override", f"{k}={v}"]

        # 启动子进程
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
                env={**os.environ, "PYTHONPATH": "src"},
            )

        self._running[run_id] = proc
        TrainingJobDAO.update_status(run_id, "running", pid=proc.pid)

        # 启动监控(如果还没启动)
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

        return run_id

    def stop(self, run_id: str) -> bool:
        proc = self._running.get(run_id)
        if proc is None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        TrainingJobDAO.update_status(run_id, "stopped", end_time=datetime.now().isoformat())
        self._running.pop(run_id, None)
        return True

    # ------------------------------------------------------------------
    # 后台监控循环
    # ------------------------------------------------------------------
    def _monitor_loop(self):
        while not self._stop_monitor.is_set():
            time.sleep(2)
            done = []
            for run_id, proc in list(self._running.items()):
                ret = proc.poll()
                if ret is not None:
                    status = "completed" if ret == 0 else "failed"
                    TrainingJobDAO.update_status(
                        run_id, status,
                        end_time=datetime.now().isoformat(),
                        error_msg=f"exit code {ret}" if ret != 0 else None,
                    )
                    done.append(run_id)
                else:
                    # 还在跑,尝试从日志解析最新 iter/loss
                    self._parse_log(run_id)
            for rid in done:
                self._running.pop(rid, None)

    def _parse_log(self, run_id: str):
        job = TrainingJobDAO.get_by_run_id(run_id)
        if not job or not job.log_file:
            return
        try:
            with open(job.log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return
        # 找最后一条包含 iter/loss 的行
        for line in reversed(lines[-200:]):
            m = re.search(r"iter[:=]\s*(\d+).+loss[:=]\s*([\d.]+)", line, re.I)
            if m:
                TrainingJobDAO.update_status(run_id, "running",
                                              current_iter=int(m.group(1)), loss=float(m.group(2)))
                break


# 全局单例
training_mgr: Optional[TrainingManager] = None


def set_training_mgr(mgr: TrainingManager):
    global training_mgr
    training_mgr = mgr


def get_training_mgr() -> TrainingManager:
    if training_mgr is None:
        raise RuntimeError("TrainingManager 未初始化")
    return training_mgr
