"""聊天会话管理 API"""
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..db import ChatMessageDAO, ChatSessionDAO, ChatSessionRow, ChatMessageRow

router = APIRouter(prefix="/api/chat", tags=["chat"])


class SessionOut(BaseModel):
    id: int
    title: Optional[str]
    checkpoint_id: Optional[int]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    think_content: Optional[str]
    tokens: Optional[int]
    created_at: str

    class Config:
        from_attributes = True


class CreateSessionReq(BaseModel):
    title: Optional[str] = None
    checkpoint_id: Optional[int] = None


class CreateMessageReq(BaseModel):
    role: str
    content: str
    think_content: Optional[str] = None
    tokens: Optional[int] = None


@router.get("/sessions", response_model=List[SessionOut])
def list_sessions():
    rows = ChatSessionDAO.list_all()
    return [_sess_out(r) for r in rows]


@router.post("/sessions", response_model=SessionOut)
def create_session(req: CreateSessionReq):
    sid = ChatSessionDAO.insert(title=req.title, checkpoint_id=req.checkpoint_id)
    # 重新查询
    rows = ChatSessionDAO.list_all()
    row = next(r for r in rows if r.id == sid)
    return _sess_out(row)


@router.patch("/sessions/{session_id}")
def update_session(session_id: int, title: str):
    ChatSessionDAO.update_title(session_id, title)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int):
    ChatSessionDAO.delete(session_id)
    return {"ok": True}


@router.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
def list_messages(session_id: int):
    rows = ChatMessageDAO.list_by_session(session_id)
    return [_msg_out(r) for r in rows]


@router.post("/sessions/{session_id}/messages", response_model=MessageOut)
def add_message(session_id: int, req: CreateMessageReq):
    mid = ChatMessageDAO.insert(
        session_id=session_id, role=req.role, content=req.content,
        think_content=req.think_content, tokens=req.tokens,
    )
    rows = ChatMessageDAO.list_by_session(session_id)
    row = next(r for r in rows if r.id == mid)
    return _msg_out(row)


def _sess_out(r: ChatSessionRow) -> SessionOut:
    return SessionOut(id=r.id, title=r.title, checkpoint_id=r.checkpoint_id,
                      created_at=r.created_at, updated_at=r.updated_at)


def _msg_out(r: ChatMessageRow) -> MessageOut:
    return MessageOut(id=r.id, session_id=r.session_id, role=r.role, content=r.content,
                      think_content=r.think_content, tokens=r.tokens, created_at=r.created_at)
