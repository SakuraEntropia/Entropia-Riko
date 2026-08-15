# Entropia Riko 用户指南

> 版本：v0.1 · 节点数：195+（含插件）· 后端：PyTorch（默认）+ TensorFlow/Keras（可选）
> 本文档是面向用户的完整使用手册，覆盖界面、节点、构图、训练、导出与开发。

---

## 目录

1. [项目简介](#1-项目简介)
2. [快速开始](#2-快速开始)
3. [界面总览](#3-界面总览)
4. [模块化面板（Blender 风格）](#4-模块化面板blender-风格)
5. [欢迎窗口](#5-欢迎窗口)
6. [文件与插件](#6-文件与插件)
7. [节点系统（191 节点）](#7-节点系统191-节点)
8. [构图与执行](#8-构图与执行)
9. [数据输入与预览节点](#9-数据输入与预览节点)
10. [Wrangle 代码节点](#10-wrangle-代码节点)
11. [训练与实时 Loss 曲线](#11-训练与实时-loss-曲线)
12. [导出代码](#12-导出代码)
13. [.riko 文件格式](#13-tns-文件格式)
14. [API 端点](#14-api-端点)
15. [后端架构](#15-后端架构)
16. [开发与测试](#16-开发与测试)
17. [已知限制与后续方向](#17-已知限制与后续方向)

---

## 1. 项目简介

Entropia Riko 是一个**独立的节点图编辑器**，用 ComfyUI 风格的画布搭建并运行
PyTorch（及可选 TensorFlow/Keras）张量工作流。它不是 Houdini 插件。

核心特性：

- **191 个节点**：数学、张量操作、神经网络激活/层、注意力、归一化、归约、形状、
  创建、损失、数据加载、模型推理、子图引用、Wrangle 代码、预览，以及 TF/Keras。
- **Blender 风格模块化面板**：拖角分离/合并、任意切换窗口类型、实时缩放。
- **实时训练 + Loss 曲线**：自包含图（数据加载 + 损失）可直接训练并流式显示损失。
- **代码导出**：导出干净的 `torch.nn.Module` 或 `tf.keras.Model` 脚本。
- **子图 import**：一个 `.riko` 图可像 `import xx` 一样引用另一个图。
- **数据预览**：图片缩略图、文本、JSON 直接在检查器里显示。

---

## 2. 快速开始

### 前置条件
- Python 3.10+
- Node.js 18+
- PyTorch（CPU 版即可；CUDA/MPS 可选）

### 启动（两个终端）

```bash
cd ~/Documents/torch-node/entropia-riko

# 终端 1：API 服务器（必须先启动）
.venv/bin/python -m uvicorn src.server.app:app --reload --port 8000

# 终端 2：前端开发服务器
npm run dev    # → http://localhost:5173
```

打开 **http://localhost:5173**（务必通过 5173 访问，`/api` 走 Vite 代理）。

> ⚠️ 若看不到 190+ 节点，说明 API 服务器没启动或旧进程仍在（无 `--reload` 的旧进程
> 不会加载新代码，需杀掉重启）。改代码后浏览器需硬刷新 `Cmd+Shift+R`。

---

## 3. 界面总览

```
┌ 标题栏（* 未保存 · 文件名 - Riko · 版本号）────────────────────┐
├ 菜单栏（logo · File / Run / Data / Help，右侧状态徽章）────────┤
├ 模块化面板区域（Blender 风格区域树）──────────────────────────┤
│  [节点库] [Asset Library] [New File] [画布] [检查器]           │
│  [状态/日志] [Loss 曲线]                                       │
└──────────────────────────────────────────────────────────────┘
```

- **标题栏**：未保存时显示 `*`，随后 `文件名 - Riko`，右侧版本号。
- **菜单栏**：左上角 **logo**（点开 About / Welcome / Preferences）；File（Import/Export
  → 内置 **File Explorer** 导入导出文件/文件夹；Export Code 导出 .py）、Run、Data、
  View（含 **Liquid Glass** 主题）、Help。
- **节点库**：按分类分组、可折叠、可搜索，点击节点加入画布。
- **画布**：React Flow 节点图，右键弹搜索式节点菜单。
- **检查器**：左侧 tag 切换（Summary/Params/Inputs/Outputs/Preview）。
- **New File**：右上角 mini 文件管理器（右键新建/删除/预览代码/展开节点）。
- **Asset Library**：`.riko`/`.ric` 资产树（打开 / **展开完整节点** / 保存）。
- **Code Editor**：代码编辑器窗口（类似记事本，编辑/复制/全选/保存/生成）。
- **Handwriting Pad**：28×28 手写板，画数字 → 生成 `constant` 节点喂 MNIST。
- **状态栏**：Logs（日志）与 Queue（节点队列）两个标签。
- **Loss 曲线**：训练损失的实时 SVG 折线图。
- **Plugins**：插件管理面板（右上角 `+` 从 `.py` 加载，每个插件带启用/禁用开关）。

---

## 4. 模块化面板（Blender 风格）

每个面板右上角有一个**类型下拉**（切换该窗口显示的内容）和 **✕**（关闭当前面板）；
右下角有一个**角抓手**。

| 操作 | 手势 | 效果 |
|------|------|------|
| 切换窗口类型 | 点右上角 `类型 ▾` | Node Library / New File / Asset Library / Graph / Inspector / Status / Loss Curve / Docs / Plugins / Code Editor / Handwriting Pad 十一选一 |
| 分离（split） | 角抓手**向左/向上拖** | 蓝色预览线跟随鼠标（只在该面板内），松手在 50% 处分出新面板，方向=拖动轴向 |
| 合并（merge） | 角抓手**向右/向下拖** | 相邻窗口被**蓝色阴影 + 加粗边框**高亮，松手它被并掉、当前面板扩大 |
| 关闭当前面板 | 点 **✕** | 当前面板被移除，邻居吸收空间 |
| 缩放 | 拖分隔条 | 所有分隔条可同时独立拖动 |

默认布局：`[节点库/Asset Library 上下] | [画布 | (New File 上 / 检查器)]`，下方
`[状态 | Loss 曲线]`（New File 只占据最右列、检查器上方，不占用画布区域）。

---

## 5. 欢迎窗口

启动时弹出 Blender 风格欢迎窗口：

- **头图**：上半部为可替换 **JPEG 位图**（`public/brand/hero.jpg`，960×220），右上角显示
  版本号；左下角为 logo（`public/brand/logo.svg`）。
- **New File**：`General`（空白图）+ 从 `examples/` 读出的**预制模板**
  （cnn / classifier / mlp / mnist_cnn / transformer / …），点击即载入。
- **Recent Files**：`localStorage` 记录最近打开过的 `.riko`（去重、最多 10 条）。

点 ✕ 或选择任意项关闭；**Help → Welcome Screen** 或 logo 菜单可随时重新打开。

---

## 6. 文件与插件

**New File**（最右列、检查器上方）是一个 **mini 文件管理器**，映射**工作目录**的
目录树（可展开/折叠子目录）。**右键**弹出菜单：新建文件/新建文件夹/重命名/删除、
打开、**预览 PyTorch 代码**（`/api/project/code` 导出 `nn.Module`）、导入为节点。
**File → Import Working Folder…** 可把工作目录指向任意本地文件夹，并在其中创建
`.riko` 缓存文件夹存放工具自身状态。

**Asset Library**（原 Files 面板，Blender-outliner 风格资产树）在其基础上增加保存与
import 依赖视图，每个文件可展开显示其 `import` 依赖（✓ 已解析 / ✗ 未解析）。

- **打开**：点文件名载入画布。
- **导入为节点**：点 `⤵` 添加一个引用该文件的 `import` 节点。
- **保存**：在输入框命名后点 `+ Save .riko`（或 `⇩ .ric`），存到 `workflows/<name>`。

**子图引用**：一个 `.riko` 可被 `graph_reference`（按 `file` 路径）或 `import`
（按模块名，如 `mlp`，在 `workflows/`、`examples/`、`examples/models/` 中解析）引用。
被引用图需含 `graph_input(name="input")` 与 `graph_output(name="output")`。

**Plugins 面板**：右上角 `+` 从本地 `.py` 文件加载插件（安装到 `plugins/<name>/`），
每个插件带启用/禁用开关（禁用即从节点库移除其节点）。内置插件：
`example_plugin`（`plugin_double`）、`math_extra`（`plugin_square`、`plugin_scale`）、
`stat_extra`（`plugin_shift_scale`、`plugin_neg`）。

---

## 7. 节点系统（195+ 节点）

### 输入
`constant`

### 数学（纯 Python）
`add`, `multiply`

### 数学（torch）
`torch_add`, `torch_multiply`, `sub`, `div`, `pow`, `matmul`, `mm`, `maximum`,
`minimum`, `fmod`, `remainder`, `atan2`

### 张量（单目）
`abs`, `exp`, `log`, `log2`, `log10`, `log1p`, `expm1`, `sqrt`, `erf`, `erfc`,
`neg`, `sign`, `reciprocal`, `floor`, `ceil`, `round`, `square`, `cos`, `sin`

### 神经网络（激活）
`relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `selu`, `elu`, `mish`,
`hardswish`, `hardsigmoid`, `hardtanh`, `softsign`, `tanhshrink`, `softplus`,
`relu6`, `glu`, `softmax`, `log_softmax`, `log_sigmoid`

### 注意力
`multihead_attention`, `sdpa`（scaled dot-product attention）

### 神经网络（层）
`linear`, `conv2d`, `conv1d`, `conv_transpose2d`, `maxpool2d`, `avgpool2d`,
`embedding`, `dropout`, `batchnorm1d`, `batchnorm2d`, `layernorm`, `groupnorm`,
`rmsnorm`, `transformer_encoder`

### 归约
`sum`, `mean`, `amax`, `amin`, `logsumexp`, `median`, `max_reduce`, `min_reduce`,
`prod`, `std`, `var`, `norm`, `argmax`, `argmin`, `cumsum`, `topk`, `sort`

### 形状
`reshape`, `view`, `transpose`, `swapaxes`, `movedim`, `permute`, `flatten`,
`squeeze`, `unsqueeze`, `concat`, `stack`, `flip`, `expand`, `expand_as`,
`broadcast_to`, `tile`, `repeat`, `repeat_interleave`, `tril`, `triu`,
`diagonal`, `narrow`, `roll`, `index_select`, `gather`, `interpolate`

### 创建
`zeros`, `ones`, `rand`, `randn`, `randint`, `randperm`, `empty`, `zeros_like`,
`ones_like`, `randn_like`, `rand_like`, `eye`, `arange`, `linspace`, `full`,
`positional_encoding`

### 张量算子
`bmm`, `dot`, `outer`, `xlogy`, `cross`, `addmm`, `einsum`, `clamp`,
`contiguous`, `clone`, `detach`, `masked_fill`, `where`

### 损失
`mse_loss`, `cross_entropy_loss`, `l1_loss`, `smooth_l1_loss`,
`binary_cross_entropy`, `kl_div`, `nll_loss`, `hinge_embedding_loss`,
`cosine_embedding_loss`

### 设备
`to_device`

### 模型
`model_loader`（加载 .pt）、`inference`（跑 forward）

### 子图
`graph_input`, `graph_output`, `graph_reference`（按路径）, `import`（按模块名）

### 数据（数据集加载）
`mnist_loader`, `cifar10_loader`, `csv_loader`, `image_folder_loader`,
`tensor_file_loader`, `dataloader`, `text_loader`, `json_loader`, `image_loader`

### Wrangle 与预览
`wrangle`（内联 Python 代码）, `text_preview`, `json_preview`, `image_preview`

### TensorFlow / Keras（可选后端）
`keras_dense`, `keras_conv2d`, `keras_flatten`, `keras_embedding`,
`keras_layernorm`, `keras_dropout`, `keras_relu`, `keras_sigmoid`, `keras_tanh`,
`keras_gelu`, `keras_softmax`, `keras_maxpool2d`, `keras_avgpool2d`, `tf_add`,
`tf_multiply`, `tf_matmul`, `tf_concat`, `tf_reshape`, `tf_transpose`, `tf_reduce`

> Keras 卷积用 channels-last `(B,H,W,C)`，与 torch 的 channels-first 不同。

---

## 8. 构图与执行

1. 从节点库点击/拖入节点（或画布右键搜索添加）。
2. 拖端口连线；点击节点在右侧检查器改参数。
3. **Run → Inference** 执行整图，结果显示在节点卡片和检查器预览里。

执行后端：校验（端口/必需输入/环检测）→ Kahn 拓扑排序 → 逐节点 `execute`。
`TensorValue` 是框架无关的中间表示（纯 Python 数据），torch/TF 节点在 `execute` 内
惰性 import 并做转换。

---

## 9. 数据输入与预览节点

- **输入**：`text_loader`（读文本）、`json_loader`（读 JSON）、`image_loader`
  （用 PIL 读图片，输出 H×W×3 image_tensor）、`csv_loader`、`mnist_loader` 等。
- **预览**：
  - `text_preview`：把任意值转成文本。
  - `json_preview`：把张量/值序列化成缩进 JSON。
  - `image_preview`：把图像张量转成 **base64 PNG 缩略图**，在检查器里直接显示
    `<img>`。

`TensorValue` 支持 `kind`：`scalar` / `tensor` / `image_tensor` / `text` / `json` /
`model`。文本/JSON 不做数值 shape 推断。

---

## 10. Wrangle 代码节点

`wrangle` 节点用于解决内置节点覆盖不到的逻辑。参数 `code` 是任意 Python：

```python
# 可用变量：x / a / b（已连接输入张量，未连接为 None）、torch、F、nn
result = x * 2 + 1
# 或：
# result = torch.softmax(x, dim=-1)
# result = F.gelu(x)
```

`execute` 把输入转成 torch 张量后 `exec(code)`，取 `result`（或 `y`）作为输出。
导出时会把代码内联进 `forward`。

---

## 11. 训练与实时 Loss 曲线

**可训练图** = 数据加载节点（如 `dataloader`/`mnist_loader`）+ 损失节点
（`cross_entropy_loss` 等），且不含 `graph_input`。

- **Run → Train 20/100 steps** 或 Loss 面板按钮开始训练。
- 后端用 codegen 生成 `nn.Module` → `AdamW` 优化器逐步 `backward()`/`step()`，
  通过 **SSE（`/api/train/stream`）逐步回传 loss**，前端 SVG 曲线实时增长。

注意：执行器本身仍用 `no_grad()` 逐次重建层；真正的训练走「导出 → 执行」这条路径。

---

## 12. 导出代码

- **File → Export .py (PyTorch)**：生成干净的 `torch.nn.Module`。
  - 可学习层进 `__init__`，内联算子进 `forward`。
  - `graph_input` → `forward` 参数；`graph_output` → `return`。
  - `import`/`graph_reference` 递归内联为嵌套 `nn.Module` 类。
  - 数据加载节点生成真实可运行代码（torchvision + 随机回退）。
  - 自包含训练图附带 `train()` + `__main__` 训练循环。
- **File → Export .py (Keras/TF)**：对 TF/Keras 节点图生成 `tf.keras.Model`
  （层进 `__init__`，算子进 `call`）。

---

## 13. .riko 文件格式

`.riko` 是 JSON，保存完整工作流：

```json
{
  "version": "0.1",
  "metadata": { "name": "classifier", "description": "...",
                "inputs": [{"name": "input", "data_kind": "tensor"}],
                "outputs": [{"name": "output", "data_kind": "tensor"}] },
  "nodes": [
    {"id": "a", "type_name": "constant", "label": "Constant", "category": "Inputs",
     "position": [100, 200], "parameters": {"value": 2.0}, "inputs": [], "outputs": []}
  ],
  "edges": [
    {"id": "e1", "source_node": "a", "source_port": "value",
     "target_node": "b", "target_port": "left"}
  ],
  "settings": {}
}
```

- Save 下载 `workflow.riko`；Load 接受 `.riko`/`.json`。
- 文件管理器另提供 `POST /api/files/save` 存到 `workflows/`。

---

## 14. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/nodes` | 全部节点定义（含已启用插件） |
| POST | `/api/execute` | 执行 GraphDocument JSON，返回各节点端口输出 |
| POST | `/api/export_python` | 生成 `torch.nn.Module` 代码 |
| POST | `/api/export_keras` | 生成 `tf.keras.Model` 代码 |
| POST | `/api/export_binary` | 将图文档编码为 `.ric` 二进制（base64 下载） |
| GET | `/api/files` | 列出 `.riko` 文件及其 import 依赖 |
| GET | `/api/files/content?path=` | 读取 `.riko` 内容 |
| POST | `/api/files/save` | 保存 `.riko` 到 `workflows/` |
| GET | `/api/plugins` | 列出插件（enabled/disabled + 注册节点） |
| POST | `/api/plugins/toggle` | 启用/禁用插件（`{name, enabled}`） |
| POST | `/api/plugins/upload` | 从 Python 源码安装插件（`{name, code}`） |
| GET | `/api/project/tree` | 返回工作目录的递归目录树（隐藏 `.riko` 缓存） |
| POST | `/api/project/set_root` | 导入工作文件夹（`{path}`），创建 `.riko` 缓存并持久化 |
| POST | `/api/project/create` | 在工作目录新建空 `.riko`（`{name, dir?}`） |
| POST | `/api/project/mkdir` | 在工作目录新建文件夹（`{name, dir?}`） |
| POST | `/api/project/rename` | 重命名文件/文件夹（`{path, newName}`） |
| POST | `/api/project/delete` | 删除文件/空文件夹（`{path}`） |
| POST | `/api/project/code` | 返回该 `.riko` 导出的 PyTorch 代码（`{path}`） |
| GET | `/api/project/open?path=` | 读取工作目录内 `.riko`/`.ric` 内容 |
| POST | `/api/train` | 训练并一次性返回 loss 列表 |
| POST | `/api/train/stream` | 训练并流式（NDJSON）返回逐步 loss |

---

## 15. 后端架构

```
React UI (@xyflow/react) ──HTTP──▶ FastAPI
   │                                   │
   └─ 图文档 JSON ──▶ Runtime（registry + executor + codegen + trainer + subgraph）
                                        │
                                   Tensor IR（core/tensor.py TensorValue + document.py）
                                        │
                                   Backend（device/converter torch ↔ TensorValue；
                                            tf_converter tf ↔ TensorValue）
```

分层与依赖方向：

- `core`（纯 Python，绝不 import torch/TF）
- `runtime`（注册表 / 执行器 / codegen / codegen_tf / trainer / subgraph）
- `nodes`（math 纯 Python；torch_ops、tf_ops 惰性 import）
- `server`（FastAPI）
- `backend`（torch/TF 转换，均 optional，缺依赖时报清晰错误）

---

## 16. 开发与测试

```bash
# Python 测试（97 tests，stdlib unittest）
.venv/bin/python -m unittest discover -s tests -t .

# 前端测试（8 tests，vitest）
npm test

# 前端类型检查 + 构建
npm run build

# 验证节点数（内置 194；加上插件后 199+）
.venv/bin/python -c "import src.nodes; from src.runtime.registry import default_registry; print(len(default_registry().list()))"
# 应打印 194（未加载插件时）
```

### 新增一个节点

```python
# src/nodes/.../my_node.py
from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register

@register("my_node")
class MyNode(BaseNode):
    type_name = "my_node"
    label = "My Node"
    category = "Utility"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [Parameter("scale", default=1.0, dtype="float")]

    def execute(self, inputs, params, context):
        ...
        return {"result": ...}
```

然后在对应 `__init__.py` 里 `from . import my_node`，并在 codegen 里加导出分支。

---

## 17. 已知限制与后续方向

1. **执行器级训练未原生实现**：训练走「codegen → exec」路径；原生优化器节点待加。
2. **子图固定 input/output 端口**：`graph_reference`/`import` 目前只支持单输入单输出。
3. **Keras 卷积布局**：channels-last，与 torch 不同，混用时需 transpose。
4. **TF 未随包安装**：`requirements.txt` 中注释了 `tensorflow>=2.10`，启用需自行安装。
5. **无 undo/redo、无键盘快捷键、无画布内搜索**（右键菜单有搜索）。
6. **`Toolbar.tsx` 与 `.doc_backups/` 遗留**：可安全删除。

后续方向：原生训练循环、优化器/梯度裁剪节点、多端口子图、模型动物园（ResNet/VGG/BERT
预制 `.riko`）、JAX 后端、分组卷积/谱分解（svd/qr/eig）等更多 torch API 覆盖。
