"""FastAPI 入口:初始化 DB + 推理/训练管理器 + 挂载静态文件"""
import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .routes import chat, checkpoints, inference, training
from .services.inference_manager import InferenceManager, set_inference_mgr, get_inference_mgr
from .services.training_manager import TrainingManager, set_training_mgr


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    init_db()
    set_inference_mgr(InferenceManager())
    set_training_mgr(TrainingManager())
    yield
    # shutdown
    try:
        mgr = get_inference_mgr()
        if mgr.is_loaded():
            await mgr.unload()
    except Exception:
        pass


app = FastAPI(title="Salty LLM Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(checkpoints.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(chat.router)

# 静态文件(前端构建产物)
static_dir = Path("./frontend/dist")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
