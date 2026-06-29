import sys
from pathlib import Path

# 自动把 src 加入路径，避免 PYTHONPATH 设置问题
src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("salty.server.main:app", host="0.0.0.0", port=8000, reload=False)
