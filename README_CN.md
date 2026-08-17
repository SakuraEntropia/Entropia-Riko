# Entropia Riko

[English](README.md) | 中文

[![PyPI version](https://img.shields.io/pypi/v/entropia-riko.svg)](https://pypi.org/project/entropia-riko/)
[![GitHub release](https://img.shields.io/github/v/release/SakuraEntropia/Entropia-Riko.svg)](https://github.com/SakuraEntropia/Entropia-Riko/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Entropia Riko 是一个专业的**节点图深度学习编辑器**——ComfyUI 风格的可视化工作流，
支持 PyTorch（及可选 TensorFlow/Keras），带 Blender 式模块化工作区、实时训练曲线、
代码导出、插件系统和内置文件管理器。

它既能作为 **Web 应用**（浏览器）运行，也内置 **Electron 壳**，可当**独立桌面应用**
使用——两种方式任选。

## 安装（PyPI）

```bash
pip install entropia-riko            # 核心 + API 服务器 + PyTorch
pip install "entropia-riko[tf]"      # + TensorFlow/Keras 节点
pip install "entropia-riko[hf]"      # + Hugging Face（Diffusers/Transformers）节点
```

作为 Python 库使用：

```python
import entropia_riko
import entropia_riko.nodes          # 注册全部 194 个内置节点
from entropia_riko.runtime.registry import default_registry

print(entropia_riko.__version__)                  # "0.1.0"
print(len(default_registry().list()))             # 194
```

或启动 API 服务器：

```bash
entropia-riko                       # FastAPI 运行在 http://127.0.0.1:8000
# 等价命令：
python -m uvicorn entropia_riko.server.app:app --port 8000
```

> pip 包只包含 **Python 运行时**（节点、执行器、代码生成、训练器、子图、API 服务器）。
> 浏览器/Electron 界面不在 pip 包内——完整编辑器请克隆本仓库。

## 特性

- **200+ 节点**：数学、张量操作、神经网络层/激活、注意力、归一化、归约、形状、einsum、
  损失、数据加载、模型推理、子图引用、Hugging Face（Diffusers / Transformers）、
  TF/Keras 对应节点。
- **节点画布**（React Flow）：右键搜索菜单、拖拽连线、节点卡片实时输出预览。
- **Blender 式模块化工作区**：拖角分离/合并/缩放任意面板（分裂时两个圆角矩形蓝色预览）、
  任意切换窗口类型、多个工作区标签（Layout / Code / Training / MNIST Studio /
  Text→Image …）。
- **代码编辑器**：记事本风格（File/Edit 菜单 + 工具栏：新建/打开/保存/撤销/重做/
  剪切/复制/粘贴），用于预览/编辑导出的 PyTorch 代码。
- **训练 + 实时 Loss 曲线**：SSE 流式逐步回传损失到 SVG 图表。
- **Train-to-model / load-from-model**：`save_model` + `model_loader` 文件节点
  （Houdini 式 `path` 参数带文件选择器）以 **safetensors** 或 torch state_dict 保存/恢复模型权重。
- **干净代码导出**：PyTorch `nn.Module` 与 TensorFlow `tf.keras.Model`。
- **Asset Library 与 New File**：工作目录文件管理器（文件夹拖拽、右键新建/重命名/删除、
  「展开完整节点」把文件图内联进画布而非子图引用、每个文件的 PyTorch 代码预览）。
- **内置文件资源管理器**：Windows 式导入/导出（浏览、前进/后退、快速访问、最近文件夹；
  复制文件/文件夹，而非浏览器直接下载）。
- **插件系统**：从 `.py` 加载插件、开关启用/禁用；工作区面板与 Preferences 都能管理。
- **手写板**：画 28×28 数字 → 生成 `constant` 节点喂 MNIST 示例推理。
- **主题**：Light / Dark / System / **Liquid Glass**（苹果风格半透明玻璃）。
- **可脱出浮动窗口**：所有对话框都是可拖拽窗口。
- **`.rik` 二进制 + `.riko` ASCII 格式**，含完整 metadata/settings。

## 训练 → 保存 → 加载 → 推理

模型以 `model` 值在图里流动，通过两个文件节点落盘（Houdini 式——`path` 参数带文件选择器）：

- **`save_model`（train-to-model）**：把模型 `state_dict` 序列化到 `.safetensors`（默认）或 `.pt`/`.pth`。
- **`model_loader`（load-from-model）**：读回 `state_dict`；填 `module` 参数（或接 `template`）重建模型结构、恢复可调用。

`/api/train` 也接受 `save_path`，训练完直接保存模型。示例图成对提供 `train` + `infer`
（如 `examples/models/cnn_train.riko` + `cnn_infer.riko`），展示完整闭环：

```bash
# 1) 训练 CNN 并写出 cnn.safetensors
curl -X POST http://127.0.0.1:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"doc": <cnn_train.riko>, "steps": 20, "save_path": "cnn.safetensors"}'

# 2) 运行 cnn_infer.riko —— 加载 cnn.safetensors 并执行推理
```

## 快速开始（浏览器）

```bash
cd entropia-riko
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install

# 终端 1 — API (http://localhost:8000)
.venv/bin/python -m uvicorn entropia_riko.server.app:app --reload --port 8000

# 终端 2 — 前端 (http://localhost:5173)
npm run dev
```

打开 **http://localhost:5173**（`/api` 走 Vite 代理到 :8000）。

## 快速开始（桌面应用）

Electron 壳会启动后端，并在 Vite 开发服务器上打开原生窗口：

```bash
npm install --save-dev electron
npm run dev &                # 保持 Vite 开发服务器运行
npm run desktop
```

如需指向其他前端地址，可设置环境变量 `RIKO_DEV_URL`。

## .riko / .ric 文件格式

`.riko` 是可读 JSON；`.ric` 是同一文档在 `ERIK` 魔数头之后 zlib 压缩的二进制。两者都
包含 `version`、`metadata`（name、app、appVersion）、`nodes`、`edges`、`settings`
（主题、背景图）。详见 [`docs/FILE_FORMAT.md`](docs/FILE_FORMAT.md)。

## 插件

插件位于 `plugins/*/`（`plugin.json` 清单 + 通过 `@register` 注册节点的 `entry`
Python 模块）。可从 `.py` 文件加载更多，并在 **Plugins** 面板或 **Preferences →
Plugins** 里开关。禁用的插件会被跳过，其节点不被注册。内置示例：`example_plugin`、
`math_extra`、`stat_extra`。

## 开发命令

```bash
.venv/bin/python -m unittest discover -s tests -t .   # Python 测试
npm test                                              # 前端测试（vitest）
npm run build                                         # 类型检查 + 生产构建
npm run dev                                           # Vite 开发服务器
npm run desktop                                       # Electron 桌面壳
.venv/bin/python scripts/make_brand_assets.py         # 品牌素材助手（见脚本说明）
```

## 分发 / Release

打包一个干净的、可分享的源码 ZIP（先提交未提交改动，然后只归档已跟踪文件——
不含 `node_modules`、`.venv`、`dist`、缓存、备份）：

```bash
.venv/bin/python scripts/release.py "发布说明"
```

输出：`entropia-riko-release.zip`（在**父目录**，不修改工作文件夹）。内含 `entropia_riko/`、
`public/`、`plugins/`、`examples/`、`templates/`、`electron/`、`scripts/`、
`tests/`、`docs/`、README 与配置文件——接收者只需 `pip install -r
requirements.txt` + `npm install` 即可运行。

### PyPI 发布

构建并发布 Python 包（PyPI 上的 `entropia-riko`）：

```bash
.venv/bin/python -m pip install build twine
.venv/bin/python -m build --outdir dist-pypi
.venv/bin/python -m twine upload dist-pypi/*
```

## 项目结构

```
entropia_riko/
├── ui/         React 应用（画布、面板、代码编辑器、文件管理器…）
├── core/       Tensor IR + 图文档模型（.riko/.ric）
├── runtime/    注册表、执行器、PyTorch/TF 代码生成、训练器、子图
├── backend/    Torch 设备检测 + 转换
├── nodes/      节点定义
├── plugins/    插件加载器
└── server/     FastAPI API 服务器
plugins/        内置插件
examples/       开箱即用的预连接示例图（dataset → model → loss → output）
electron/       桌面壳（main + preload）
scripts/        品牌素材生成器
public/brand/   logo.svg + hero.jpg（原地替换即可换品牌）
```

## 文档

- **[用户指南](docs/USER_GUIDE.md)** — 完整手册（界面、节点、训练、导出、API）。
- `docs/`：`APP_SPEC.md`、`APP_ARCHITECTURE.md`、`API.md`、`NODE_SYSTEM.md`、
  `DATA_FORMAT.md`、`FILE_FORMAT.md`、`UI_STANDARD.md`、`TORCH_BACKEND.md`、
  `CROSS_PLATFORM.md`。

## 许可证

MIT
