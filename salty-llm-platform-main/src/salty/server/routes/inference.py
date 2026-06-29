"""推理 API:OpenAI-compatible /chat/completions + WebSocket 流式"""
import json
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..db import ChatMessageDAO, ChatSessionDAO, CheckpointDAO
from ..services.inference_manager import get_inference_mgr

router = APIRouter(prefix="/api/inference", tags=["inference"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionReq(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.3
    top_k: int = 10
    top_p: Optional[float] = None
    repetition_penalty: float = 1.25
    stream: bool = False


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionReq):
    mgr = get_inference_mgr()
    if not mgr.is_loaded():
        return {"error": "模型未加载,请先选择 checkpoint"}, 400

    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        async def event_stream() -> AsyncIterator[str]:
            async for evt in mgr.chat(
                messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                rep_penalty=req.repetition_penalty,
            ):
                if evt.type == "delta":
                    yield f"data: {json.dumps({'choices':[{'delta':{'content':evt.text}}]})}\n\n"
                elif evt.type == "think_open":
                    yield f"data: {json.dumps({'choices':[{'delta':{'content':'<think>'}}]})}\n\n"
                elif evt.type == "think_delta":
                    yield f"data: {json.dumps({'choices':[{'delta':{'content':evt.text}}]})}\n\n"
                elif evt.type == "think_close":
                    yield f"data: {json.dumps({'choices':[{'delta':{'content':'</think>'}}]})}\n\n"
                elif evt.type == "done":
                    yield f"data: {json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                elif evt.type == "error":
                    yield f"data: {json.dumps({'error':evt.text})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # 非流式
    text_parts = []
    think_parts = []
    in_think = False
    async for evt in mgr.chat(
        messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        rep_penalty=req.repetition_penalty,
    ):
        if evt.type == "think_open":
            in_think = True
        elif evt.type == "think_close":
            in_think = False
        elif evt.type == "think_delta":
            think_parts.append(evt.text)
        elif evt.type == "delta":
            text_parts.append(evt.text)

    content = "".join(text_parts)
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


class LoadReq(BaseModel):
    ckpt_id: int


@router.post("/load")
async def load_model(req: LoadReq):
    ckpt_id = req.ckpt_id
    ckpt = CheckpointDAO.list_all()
    target = next((c for c in ckpt if c.id == ckpt_id), None)
    if not target:
        return {"error": "checkpoint 不存在"}, 404
    info = await get_inference_mgr().load(target.path)
    CheckpointDAO.set_current(ckpt_id)
    return {"ok": True, "info": info}


@router.post("/unload")
async def unload_model():
    await get_inference_mgr().unload()
    return {"ok": True}


@router.get("/status")
def inference_status():
    mgr = get_inference_mgr()
    ckpt = CheckpointDAO.get_current()
    return {
        "loaded": mgr.is_loaded(),
        "current_ckpt": ckpt.path if ckpt else None,
        "current_ckpt_id": ckpt.id if ckpt else None,
    }


# ------------------------------------------------------------------
# WebSocket 端点(供前端实时聊天)
# ------------------------------------------------------------------
@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    mgr = get_inference_mgr()
    try:
        while True:
            data = await websocket.receive_json()
            messages = data.get("messages", [])
            params = data.get("params", {})

            if not mgr.is_loaded():
                await websocket.send_json({"type": "error", "text": "模型未加载"})
                continue

            async for evt in mgr.chat(messages, **params):
                await websocket.send_json({"type": evt.type, "text": evt.text, "meta": evt.meta})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "text": str(e)})
