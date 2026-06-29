"""Checkpoint 管理 API"""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ...utils.ckpt_io import load_compat
from ..db import CheckpointDAO, CheckpointRow

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])


class CkptOut(BaseModel):
    id: int
    name: str
    path: str
    stage: str
    step: Optional[int]
    schema_version: int
    is_starred: bool
    is_current: bool
    params: Optional[int]
    created_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[CkptOut])
def list_checkpoints(stage: Optional[str] = None):
    rows = CheckpointDAO.list_all(stage=stage)
    return [_to_out(r) for r in rows]


@router.post("/scan")
def scan_checkpoints():
    """扫描本地 checkpoints/ 和旧目录,自动入库"""
    found = 0
    dirs = [Path("checkpoints"), Path("sft_checkpoints"), Path("dpo_checkpoints"), Path("grpo_checkpoints")]
    for d in dirs:
        if not d.exists():
            continue
        for pt in d.rglob("*.pt"):
            # 检查是否已入库
            existing = CheckpointDAO.list_all()
            if any(r.path == str(pt) for r in existing):
                continue
            try:
                info = load_compat(str(pt), map_location="cpu")
                row = CheckpointRow(
                    id=0, name=pt.stem, path=str(pt), stage=info.stage or "unknown",
                    step=info.iter, schema_version=info.schema_version or 1,
                    is_starred=False, is_current=False, params=None, created_at="",
                )
                CheckpointDAO.insert(row)
                found += 1
            except Exception:
                pass
    return {"found": found}


@router.post("/{ckpt_id}/current")
def set_current(ckpt_id: int):
    CheckpointDAO.set_current(ckpt_id)
    return {"ok": True}


@router.post("/{ckpt_id}/star")
def toggle_star(ckpt_id: int):
    new_state = CheckpointDAO.toggle_star(ckpt_id)
    return {"is_starred": new_state}


@router.delete("/{ckpt_id}")
def delete_checkpoint(ckpt_id: int):
    CheckpointDAO.delete(ckpt_id)
    return {"ok": True}


def _to_out(r: CheckpointRow) -> CkptOut:
    return CkptOut(
        id=r.id, name=r.name, path=r.path, stage=r.stage, step=r.step,
        schema_version=r.schema_version, is_starred=r.is_starred,
        is_current=r.is_current, params=r.params, created_at=r.created_at,
    )
