# Entropia Riko User Guide

> Version: v0.1 · Nodes: 191 · Backends: PyTorch (default) + TensorFlow/Keras (optional)
> This is the complete user manual: interface, nodes, graph building, training, export and development.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Interface](#3-interface)
4. [Modular Panels (Blender-style)](#4-modular-panels-blender-style)
5. [Welcome Screen](#5-welcome-screen)
6. [File Manager & Import](#6-file-manager--import)
7. [Node System (191 nodes)](#7-node-system-191-nodes)
8. [Building & Running Graphs](#8-building--running-graphs)
9. [Data Input & Preview Nodes](#9-data-input--preview-nodes)
10. [Wrangle Code Node](#10-wrangle-code-node)
11. [Training & Real-time Loss Curve](#11-training--real-time-loss-curve)
12. [Exporting Code](#12-exporting-code)
13. [.riko File Format](#13-tns-file-format)
14. [API Endpoints](#14-api-endpoints)
15. [Backend Architecture](#15-backend-architecture)
16. [Development & Testing](#16-development--testing)
17. [Known Limitations](#17-known-limitations)

---

## 1. Overview

Entropia Riko is a **standalone node-graph editor** for building and running
PyTorch (and optionally TensorFlow/Keras) tensor workflows, with a ComfyUI-style
canvas and a Material-inspired UI. It is **not** a Houdini plugin.

Key features:

- **191 nodes**: math, tensor ops, neural activations/layers, attention,
  normalization, reductions, shape ops, creation, losses, data loading, model
  inference, subgraph references, Wrangle code, previews, and TF/Keras.
- **Blender-style modular panels**: drag a corner to split/merge, switch any
  window's type, and resize everything simultaneously.
- **Real-time training + loss curve**: self-contained graphs (data + loss)
  train live with a streaming SVG loss chart.
- **Code export**: generate clean `torch.nn.Module` or `tf.keras.Model` scripts.
- **Subgraph import**: a `.riko` graph can reference another like `import xx`.
- **Data previews**: image thumbnails, text and JSON rendered in the inspector.

---

## 2. Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PyTorch (CPU build is enough; CUDA/MPS optional)

### Run (two terminals)

```bash
cd ~/Documents/torch-node/entropia-riko

# Terminal 1: API server (start FIRST)
.venv/bin/python -m uvicorn src.server.app:app --reload --port 8000

# Terminal 2: frontend dev server
npm run dev    # → http://localhost:5173
```

Open **http://localhost:5173** (the `/api` proxy only works through 5173).

> ⚠️ If you don't see 190+ nodes, the API server isn't running, or a stale
> process started without `--reload` is still serving old code — kill and
> restart it. After code changes, hard-refresh the browser with `Cmd+Shift+R`.

---

## 3. Interface

```
┌ Menu bar (File / Run / Data / Help, status badge) ────────────────┐
├ Modular panel area (Blender-style area tree) ─────────────────────┤
│  [Node Library] [Files] [Canvas (React Flow)] [Inspector]          │
│  [Status / Logs] [Loss Curve]                                      │
└────────────────────────────────────────────────────────────────────┘
```

- **Menu bar**: File (new / save / load / export PyTorch / export Keras),
  Run (inference / Train 20 / 100 steps), Data (quick-add dataset nodes),
  Help (welcome screen / about).
- **Node Library**: grouped, collapsible, searchable; click to add a node.
- **Canvas**: React Flow node graph; right-click opens a searchable node menu.
- **Inspector**: parameters / inputs / outputs / preview of the selected node.
- **Status**: Logs and Queue tabs.
- **Loss Curve**: live training loss as an SVG line chart.

---

## 4. Modular Panels (Blender-style)

Each panel has a **type dropdown** (top-right) to switch what it shows and a
**✕** (close) button; the bottom-right corner is a **grip**.

| Action | Gesture | Effect |
|--------|---------|--------|
| Switch type | click `type ▾` | Node Library / Files / Graph / Inspector / Status / Loss Curve / Documentation |
| Split | drag grip **up/left** | a blue preview line follows the mouse (clipped to the panel); release splits at 50%, orientation = drag axis |
| Merge | drag grip **down/right** | the sibling panel is **shaded + bold-bordered**; release merges it away, current expands |
| Close current | click **✕** | current panel removed, neighbor absorbs the space |
| Resize | drag a splitter | all splitters resize independently and simultaneously |

Default layout: `[Node Library / Files stacked] | [Graph | Inspector]` over
`[Status | Loss Curve]`.

---

## 5. Welcome Screen

A Blender-style splash opens on startup, with two columns:

- **New File**: `General` (blank graph) + preset templates read from
  `examples/` (cnn / classifier / mlp / mnist_cnn / transformer / …).
- **Recent Files**: the last opened `.riko` files (deduped, up to 10, stored in
  `localStorage`).

Close with ✕ or by picking an item; reopen via **Help → Welcome Screen**.

---

## 6. File Manager & Import

The **Files** panel is a Blender-outliner-style tree of `.riko` files under
`workflows/` and `examples/`. Each file expands to show its `import`
dependencies (✓ resolved / ✗ missing).

- **Open**: click a file name to load it.
- **Import as node**: click `⤵` to add an `import` node referencing it.
- **Save**: name it and click `+ Save` → `workflows/<name>.riko`.

**Subgraph references**: a `.riko` can be referenced by `graph_reference`
(by `file` path) or `import` (by module name, e.g. `mlp`, resolved under
`workflows/`, `examples/`, `examples/models/`). The referenced graph must
contain `graph_input(name="input")` and `graph_output(name="output")`.

---

## 7. Node System (191 nodes)

### Inputs
`constant`

### Math (pure Python)
`add`, `multiply`

### Math (torch)
`torch_add`, `torch_multiply`, `sub`, `div`, `pow`, `matmul`, `mm`, `maximum`,
`minimum`, `fmod`, `remainder`, `atan2`

### Tensor (unary)
`abs`, `exp`, `log`, `log2`, `log10`, `log1p`, `expm1`, `sqrt`, `erf`, `erfc`,
`neg`, `sign`, `reciprocal`, `floor`, `ceil`, `round`, `square`, `cos`, `sin`

### Neural (activations)
`relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `selu`, `elu`, `mish`,
`hardswish`, `hardsigmoid`, `hardtanh`, `softsign`, `tanhshrink`, `softplus`,
`relu6`, `glu`, `softmax`, `log_softmax`, `log_sigmoid`

### Attention
`multihead_attention`, `sdpa` (scaled dot-product attention)

### Neural (layers)
`linear`, `conv2d`, `conv1d`, `conv_transpose2d`, `maxpool2d`, `avgpool2d`,
`embedding`, `dropout`, `batchnorm1d`, `batchnorm2d`, `layernorm`, `groupnorm`,
`rmsnorm`, `transformer_encoder`

### Reduce
`sum`, `mean`, `amax`, `amin`, `logsumexp`, `median`, `max_reduce`, `min_reduce`,
`prod`, `std`, `var`, `norm`, `argmax`, `argmin`, `cumsum`, `topk`, `sort`

### Shape
`reshape`, `view`, `transpose`, `swapaxes`, `movedim`, `permute`, `flatten`,
`squeeze`, `unsqueeze`, `concat`, `stack`, `flip`, `expand`, `expand_as`,
`broadcast_to`, `tile`, `repeat`, `repeat_interleave`, `tril`, `triu`,
`diagonal`, `narrow`, `roll`, `index_select`, `gather`, `interpolate`

### Creation
`zeros`, `ones`, `rand`, `randn`, `randint`, `randperm`, `empty`, `zeros_like`,
`ones_like`, `randn_like`, `rand_like`, `eye`, `arange`, `linspace`, `full`,
`positional_encoding`

### Tensor ops
`bmm`, `dot`, `outer`, `xlogy`, `cross`, `addmm`, `einsum`, `clamp`,
`contiguous`, `clone`, `detach`, `masked_fill`, `where`

### Loss
`mse_loss`, `cross_entropy_loss`, `l1_loss`, `smooth_l1_loss`,
`binary_cross_entropy`, `kl_div`, `nll_loss`, `hinge_embedding_loss`,
`cosine_embedding_loss`

### Device
`to_device`

### Model
`model_loader` (load .pt), `inference` (run forward)

### Subgraph
`graph_input`, `graph_output`, `graph_reference` (by path), `import` (by module)

### Data (dataset loaders)
`mnist_loader`, `cifar10_loader`, `csv_loader`, `image_folder_loader`,
`tensor_file_loader`, `dataloader`, `text_loader`, `json_loader`, `image_loader`

### Wrangle & Preview
`wrangle` (inline Python code), `text_preview`, `json_preview`, `image_preview`

### TensorFlow / Keras (optional backend)
`keras_dense`, `keras_conv2d`, `keras_flatten`, `keras_embedding`,
`keras_layernorm`, `keras_dropout`, `keras_relu`, `keras_sigmoid`, `keras_tanh`,
`keras_gelu`, `keras_softmax`, `keras_maxpool2d`, `keras_avgpool2d`, `tf_add`,
`tf_multiply`, `tf_matmul`, `tf_concat`, `tf_reshape`, `tf_transpose`, `tf_reduce`

> Keras convolutions use channels-last `(B,H,W,C)`, unlike torch's channels-first.

---

## 8. Building & Running Graphs

1. Add nodes from the library (or right-click the canvas to search).
2. Drag between ports to connect; click a node to edit parameters in the inspector.
3. **Run → Inference** executes the whole graph; results appear on node cards
   and in the inspector preview.

Execution backend: validation (ports / required inputs / cycle detection) →
Kahn topological sort → per-node `execute`. `TensorValue` is a framework-neutral
IR (pure-Python data); torch/TF nodes lazily import and convert inside `execute`.

---

## 9. Data Input & Preview Nodes

- **Inputs**: `text_loader` (read text), `json_loader` (parse JSON),
  `image_loader` (PIL → H×W×3 image_tensor), `csv_loader`, `mnist_loader`, …
- **Previews**:
  - `text_preview`: convert any value to text.
  - `json_preview`: serialize a tensor/value to indented JSON.
  - `image_preview`: render an image tensor as a **base64 PNG thumbnail** shown
    in the inspector.

`TensorValue` supports `kind`: `scalar` / `tensor` / `image_tensor` / `text` /
`json` / `model`. Text/JSON payloads skip numeric shape inference.

---

## 10. Wrangle Code Node

The `wrangle` node covers whatever built-in nodes can't. Its `code` parameter
is arbitrary Python:

```python
# available: x / a / b (connected input tensors, None if unconnected), torch, F, nn
result = x * 2 + 1
# result = torch.softmax(x, dim=-1)
# result = F.gelu(x)
```

`execute` converts inputs to torch tensors, runs `exec(code)`, and returns
`result` (or `y`). Export inlines the code into `forward`.

---

## 11. Training & Real-time Loss Curve

A **trainable graph** = a data loader node (e.g. `dataloader`/`mnist_loader`) +
a loss node (e.g. `cross_entropy_loss`), with no `graph_input`.

- **Run → Train 20/100 steps** (or the Loss panel buttons) starts training.
- The backend generates an `nn.Module` via codegen, runs `AdamW` steps with
  `backward()`/`step()`, and streams each loss over **SSE
  (`/api/train/stream`)**; the frontend SVG curve grows in real time.

Note: the executor itself still uses `no_grad()` and rebuilds layers per call;
real training goes through the "export → exec" path.

---

## 12. Exporting Code

- **File → Export .py (PyTorch)**: emits a clean `torch.nn.Module`.
  - learnable layers → `__init__`, inline ops → `forward`.
  - `graph_input` → `forward` parameter; `graph_output` → `return`.
  - `import`/`graph_reference` are inlined recursively as nested modules.
  - data loaders emit real runnable code (torchvision + random fallback).
  - self-contained training graphs get a `train()` + `__main__` loop.
- **File → Export .py (Keras/TF)**: emits a subclassed `tf.keras.Model`
  (layers → `__init__`, ops → `call`).

---

## 13. .riko File Format

`.riko` is JSON storing a complete workflow:

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

- Save downloads `workflow.riko`; Load accepts `.riko`/`.json`.
- The file manager additionally saves to `workflows/` via `POST /api/files/save`.

---

## 14. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | health check |
| GET | `/api/nodes` | all 191 node definitions |
| POST | `/api/execute` | execute a GraphDocument JSON, return per-node outputs |
| POST | `/api/export_python` | generate `torch.nn.Module` code |
| POST | `/api/export_keras` | generate `tf.keras.Model` code |
| GET | `/api/files` | list `.riko` files + their import dependencies |
| GET | `/api/files/content?path=` | read a `.riko` file |
| POST | `/api/files/save` | save a `.riko` to `workflows/` |
| POST | `/api/train` | train and return the loss list |
| POST | `/api/train/stream` | train and stream per-step losses (NDJSON) |

---

## 15. Backend Architecture

```
React UI (@xyflow/react) ──HTTP──▶ FastAPI
   │                                   │
   └─ graph JSON ──▶ Runtime (registry + executor + codegen + trainer + subgraph)
                                        │
                                   Tensor IR (core/tensor.py TensorValue + document.py)
                                        │
                                   Backend (device/converter torch ↔ TensorValue;
                                            tf_converter tf ↔ TensorValue)
```

Layers & dependency direction:

- `core` (pure Python — never imports torch/TF)
- `runtime` (registry / executor / codegen / codegen_tf / trainer / subgraph)
- `nodes` (math pure-Python; torch_ops, tf_ops lazy-import)
- `server` (FastAPI)
- `backend` (torch/TF converters, both optional with clear errors)

---

## 16. Development & Testing

```bash
# Python tests (89 tests, stdlib unittest)
.venv/bin/python -m unittest discover -s tests -t .

# Frontend tests (8 tests, vitest)
npm test

# Frontend type-check + build
npm run build

# Verify node count
.venv/bin/python -c "import src.nodes; from src.runtime.registry import default_registry; print(len(default_registry().list()))"
# should print 191
```

### Adding a node

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

Then `from . import my_node` in the package `__init__.py`, and add an export
branch in codegen.

---

## 17. Known Limitations

1. **No native executor-level training**: training uses the codegen→exec path;
   an in-executor optimizer node is still pending.
2. **Fixed subgraph input/output ports**: `graph_reference`/`import` support a
   single input and a single output only.
3. **Keras convolution layout**: channels-last, differs from torch.
4. **TF not bundled**: `requirements.txt` comments out `tensorflow>=2.10`;
   install it yourself to enable TF nodes.
5. **No undo/redo, no keyboard shortcuts, no in-canvas search** (right-click menu
   has search).
6. **`Toolbar.tsx` and `.doc_backups/` are legacy**: safe to delete.

Future: native training loop, optimizer/gradient-clipping nodes, multi-port
subgraphs, a model zoo (ResNet/VGG/BERT presets), a JAX backend, and more torch
API coverage (grouped conv, spectral decompositions svd/qr/eig, …).
