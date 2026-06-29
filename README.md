# Salty LLM Platform

Salty LLM Platform 是一个从零实现的小型中文大语言模型训练与部署项目，覆盖 Transformer 模型实现、预训练、SFT、DPO、GRPO、推理服务和 Web 控制台。

项目适合作为 LLM 训练与 LLMOps 工程能力展示：不是单纯调用大模型 API，而是完整实现小模型结构、训练框架、推理接口和前端管理页面。

## Features

- TinyLM 模型实现：RMSNorm、RoPE、GQA、SwiGLU、KV Cache、权重绑定。
- 统一训练入口：支持 pretrain、SFT、DPO、GRPO 四阶段训练。
- 工程化训练能力：YAML 配置、梯度累积、混合精度、梯度裁剪、cosine warmup、checkpoint、TensorBoard 日志。
- 推理服务：FastAPI 后端，支持模型加载/卸载、流式推理、聊天会话和训练任务管理。
- Web 控制台：React/Vite 前端，支持聊天、训练任务、checkpoint 管理和日志查看。

## Project Structure

```text
.
├── configs/              # pretrain / sft / dpo / grpo 配置
├── frontend/             # React + Vite 控制台源码
├── scripts/              # tokenizer 和数据准备脚本
├── src/salty/
│   ├── model/            # TinyLM 模型结构
│   ├── trainers/         # 四阶段训练器
│   ├── inference/        # 推理引擎
│   ├── server/           # FastAPI 服务
│   └── data/             # 数据集读取逻辑
├── pyproject.toml
└── start.py              # 后端启动入口
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[backend,dev]"
```

检查训练配置：

```bash
python -m salty.run --config configs/sft.yaml --dry-run --device cpu
```

启动后端：

```bash
python start.py
```

启动前端：

```bash
cd frontend
npm install
npm run dev
```

## Training

```bash
python -m salty.run --config configs/pretrain.yaml
python -m salty.run --config configs/sft.yaml
python -m salty.run --config configs/dpo.yaml
python -m salty.run --config configs/grpo.yaml
```

训练所需语料、tokenizer 和 checkpoint 不包含在仓库中。请按配置文件中的路径准备：

- `tokenizer_path`
- `bin_path` 或 `data_path`
- `pt_ckpt` / `ref_ckpt` / `base_ckpt`
- `ckpt_dir`
- `log_dir`

## Resume Highlights

- 从零实现约 21M 参数 TinyLM，覆盖 Transformer 核心结构与自回归生成。
- 搭建统一训练框架，支持预训练、指令微调、偏好优化和 GRPO 对齐流程。
- 开发 FastAPI + React 控制台，实现模型推理、训练任务、checkpoint 和聊天记录管理。

