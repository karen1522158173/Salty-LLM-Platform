"""训练任务管理 API"""
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..db import TrainingJobDAO, TrainingJobRow
from ..services.training_manager import get_training_mgr

router = APIRouter(prefix="/api/training", tags=["training"])


class StartReq(BaseModel):
    run_id: str
    config_path: str
    overrides: Optional[Dict[str, str]] = None


class JobOut(BaseModel):
    id: int
    run_id: str
    stage: str
    config_path: str
    overrides: Optional[str]
    status: str
    pid: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]
    current_iter: int
    max_iter: Optional[int]
    loss: Optional[float]
    log_file: Optional[str]
    error_msg: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[JobOut])
def list_jobs():
    rows = TrainingJobDAO.list_all()
    return [_to_out(r) for r in rows]


@router.post("/start")
def start_job(req: StartReq):
    run_id = get_training_mgr().start(req.run_id, req.config_path, req.overrides)
    return {"run_id": run_id}


@router.post("/{run_id}/stop")
def stop_job(run_id: str):
    ok = get_training_mgr().stop(run_id)
    return {"ok": ok}


@router.get("/{run_id}/log")
def get_log(run_id: str, tail: int = 100):
    job = TrainingJobDAO.get_by_run_id(run_id)
    if not job or not job.log_file:
        return {"lines": []}
    try:
        with open(job.log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return {"lines": lines[-tail:]}
    except Exception as e:
        return {"lines": [], "error": str(e)}


def _to_out(r: TrainingJobRow) -> JobOut:
    return JobOut(
        id=r.id, run_id=r.run_id, stage=r.stage, config_path=r.config_path,
        overrides=r.overrides, status=r.status, pid=r.pid,
        start_time=r.start_time, end_time=r.end_time,
        current_iter=r.current_iter, max_iter=r.max_iter,
        loss=r.loss, log_file=r.log_file, error_msg=r.error_msg,
        created_at=r.created_at,
    )
