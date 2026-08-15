# Entropia Riko — Project Handoff Document

> Generated: 2026-08-14
> Purpose: Transfer full project context to another AI agent (DeepSeek harness).
> Read this ENTIRE document before touching any code.

---

## 1. What Is This Project

**Entropia Riko** is a standalone node-graph editor for building and running
PyTorch tensor workflows. It is **not** a Houdini plugin — that direction was
abandoned. The product combines:

- A ComfyUI-style procedural workflow (canvas, node library, side inspector,
  execution queue, output previews).
- A Material Design-inspired UI.
- PyTorch as the computation backend (CPU-first, CUDA/MPS auto-detected).
- A FastAPI backend that executes graphs and generates Python code.
- 194 registered node types covering math, tensor ops, neural layers, data
  loading, loss functions, subgraph references, and more.

**Project location:** `~/Documents/torch-node/entropia-riko`
(The directory name `entropia-riko` is legacy from a previous Houdini
direction; the product is now "Entropia Riko".)

---

## 2. Tech Stack

### Frontend
- React 18 + Vite 5 + TypeScript 5
- @xyflow/react v12 (React Flow — node graph canvas)
- zustand v4 (state management)
- vitest (testing)
- No CSS framework — hand-written Material-inspired CSS

### Backend (Python)
- Python 3.14 (system) / venv with 3.14
- FastAPI + uvicorn (API server)
- PyTorch 2.13 + torchvision 0.28 + numpy 2.5
- Standard library `unittest` (no pytest)
- Standard library `csv` (no pandas)

### CI
- GitHub Actions (`.github/workflows/ci.yml`)
- Two jobs: backend (Python unittest) + frontend (npm build)

---

## 3. Architecture

```
React UI (@xyflow/react)  ──HTTP──▶  FastAPI API Server
   │                                     │
   │ graph document (JSON)               ▼
   │                              Node Runtime (registry + executor + codegen + subgraph)
   │                                     │
   │                                     ▼
   │                              Tensor IR (graph document model)
   │                                     │
   │                                     ▼
   └── tensor previews ◀────────  Torch Backend (device + converter)
```

### Module Layout (`src/`)

| Module | Responsibility |
|--------|---------------|
| `src/ui/` | React app: MenuBar, NodeLibrary, GraphCanvas, SideInspector, StatusPanel, FileManager, ContextMenu, NodeCard, Splitter. zustand store in `store/graphStore.ts`. |
| `src/core/` | Tensor IR (`tensor.py`: TensorValue with shape/dtype/device/metadata) + Graph document model (`document.py`: GraphDocument, NodeModel, EdgeModel, PortModel). Pure Python, no torch. |
| `src/runtime/` | Registry (`registry.py`), Executor (`executor.py`: validate + topo sort + execute), Codegen (`codegen.py`: export to clean nn.Module Python script), Codegen-TF (`codegen_tf.py`: export to tf.keras.Model), Subgraph (`subgraph.py`: module resolution + subgraph execution), Trainer (`trainer.py`: run optimizer steps via codegen). |
| `src/backend/` | Device detection (`device.py`: CPU/CUDA/MPS/auto), Tensor conversion (`converter.py`: TensorValue ↔ torch.Tensor), TF conversion (`tf_converter.py`: TensorValue ↔ tf.Tensor). torch/tensorflow both optional — degrade gracefully. |
| `src/nodes/` | Node definitions: `math/` (constant, add, multiply — pure Python), `torch_ops/` (linear, relu, transformer_encoder, 150+ torch API nodes, data loaders, loss, model_loader, inference), `tf_ops/` (keras layers + TF ops, optional), `subgraph/` (graph_input, graph_output, graph_reference, import). |
| `src/server/` | FastAPI app (`app.py`): `/api/execute`, `/api/nodes`, `/api/export_python`, `/api/export_keras`, `/api/health`, `/api/files`, `/api/files/content`, `/api/files/save`, `/api/train`, `/api/train/stream`. |
| `src/utils/` | Config (`config.py`), Logging (`logging.py`). |

### Dependency Direction (IMPORTANT)
```
UI → Server → Runtime → Core → (Backend optional)
Nodes → Core + Runtime (for registration)
Nodes/torch_ops → Backend (for torch execution)
```
- `src/core` NEVER imports torch or Houdini.
- `src/nodes/math` (constant/add/multiply) are pure Python.
- `src/nodes/torch_ops` import torch lazily inside `execute()`.
- `src/nodes/subgraph` enables cross-graph references via `.riko` files.

---

## 4. How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- The `.venv/` directory already exists with torch, torchvision, numpy, fastapi, uvicorn installed.

### Start (two terminals)

```bash
cd ~/Documents/torch-node/entropia-riko

# Terminal 1: API server (MUST start first)
.venv/bin/python -m uvicorn src.server.app:app --reload --port 8000

# Terminal 2: Frontend dev server
npm run dev    # → http://localhost:5173
```

### Vite Proxy
`vite.config.ts` proxies `/api` → `http://localhost:8000`. This means the
frontend must be accessed via `http://localhost:5173` (not by opening
`dist/index.html` directly — the proxy only works in dev mode).

### Tests

```bash
# Python (89 tests)
.venv/bin/python -m unittest discover -s tests -t .

# Frontend (8 vitest tests)
npm test

# Frontend build (tsc type-check + vite build)
npm run build
```

---

## 5. API Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/api/health` | — | `{"status":"ok"}` |
| GET | `/api/nodes` | — | `{"nodes": [{type, label, category, inputs, outputs, parameters}, ...]}` — all 194 registered nodes |
| POST | `/api/execute` | GraphDocument JSON | `{"status":"success", "outputs": {nodeId: {port: {shape, dtype, device, summary, data}}}}` or `{"status":"error", "errors": [...]}` |
| POST | `/api/export_python` | GraphDocument JSON | `{"status":"success", "code": "...nn.Module Python source..."}` |
| POST | `/api/export_keras` | GraphDocument JSON | `{"status":"success", "code": "...tf.keras.Model Python source..."}` |
| GET | `/api/files` | — | `{"files": [{name, path, imports: [{spec, path, resolved}]}], "search_paths": [...]}` — disk-backed .riko file tree with import relationships |
| GET | `/api/files/content?path=...` | — | `{"status":"success", "doc": <GraphDocument JSON>}` |
| POST | `/api/files/save` | `{name, doc}` | `{"status":"success", "path": "workflows/<name>.riko"}` |
| POST | `/api/train` | `{doc, steps, lr}` | `{"status":"success", "losses": [float, ...]}` — runs optimizer steps on a self-contained graph and returns loss history |

### GraphDocument JSON Format
```json
{
  "version": "0.1",
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

---

## 6. All 194 Registered Node Types

Grouped by category:

### Inputs (1)
`constant`

### Math — Pure Python (2)
`add`, `multiply`

### Math — Torch (12)
`torch_add`, `torch_multiply`, `sub`, `div`, `pow`, `matmul`, `mm`, `maximum`, `minimum`, `fmod`, `remainder`, `atan2`

### Tensor — Unary (13)
`abs`, `exp`, `log`, `sqrt`, `neg`, `sign`, `reciprocal`, `floor`, `ceil`, `round`, `square`, `cos`, `sin`

### Neural — Activation (13)
`relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `selu`, `elu`, `mish`, `hardswish`, `softplus`, `relu6`, `softmax`, `log_softmax`

### Neural — Layers (11)
`linear`, `conv2d`, `conv1d`, `conv_transpose2d`, `maxpool2d`, `avgpool2d`, `embedding`, `dropout`, `batchnorm1d`, `layernorm`, `transformer_encoder`

### Reduce (13)
`sum`, `mean`, `max_reduce`, `min_reduce`, `prod`, `std`, `var`, `norm`, `argmax`, `argmin`, `cumsum`, `topk`, `sort`

### Shape (11)
`reshape`, `transpose`, `permute`, `flatten`, `squeeze`, `unsqueeze`, `concat`, `stack`, `flip`, `expand_as`, `repeat_interleave`

### Creation (8)
`zeros`, `ones`, `rand`, `randn`, `eye`, `arange`, `linspace`, `full`

### Device (1)
`to_device`

### Data (6)
`mnist_loader`, `cifar10_loader`, `csv_loader`, `image_folder_loader`, `tensor_file_loader`, `dataloader`

### Loss (2)
`mse_loss`, `cross_entropy_loss`

### Model (2)
`model_loader` (load .pt file), `inference` (run model forward)

### Subgraph (4)
`graph_input`, `graph_output`, `graph_reference` (reference another .riko file
by path), `import` (reference another .riko file by module name, like Python's
`import xx`)

### Tensor — Other (6)
`clamp`, `contiguous`, `clone`, `detach`, `masked_fill`, `where`

### Attention (2)
`multihead_attention`, `sdpa` (scaled dot-product attention)

### Normalization extras (3)
`batchnorm2d`, `groupnorm`, `rmsnorm`

### Math / activation extras (~15)
`log2`, `log10`, `log1p`, `expm1`, `erf`, `erfc`, `leaky_relu`, `hardtanh`,
`hardsigmoid`, `softsign`, `tanhshrink`, `log_sigmoid`, `glu`, `amax`, `amin`,
`logsumexp`, `median`

### Shape / layout extras (~20)
`view`, `swapaxes`, `movedim`, `expand`, `broadcast_to`, `tile`, `repeat`,
`tril`, `triu`, `diagonal`, `narrow`, `roll`, `index_select`, `gather`,
`interpolate`, `einsum`, `bmm`, `dot`, `outer`, `xlogy`, `cross`, `addmm`

### Creation extras (~10)
`randint`, `randperm`, `empty`, `zeros_like`, `ones_like`, `randn_like`,
`rand_like`, `positional_encoding`

### Loss extras (7)
`l1_loss`, `smooth_l1_loss`, `binary_cross_entropy`, `kl_div`, `nll_loss`,
`hinge_embedding_loss`, `cosine_embedding_loss`

### TensorFlow / Keras (20, optional backend)
`keras_dense`, `keras_conv2d`, `keras_flatten`, `keras_embedding`,
`keras_layernorm`, `keras_dropout`, `keras_relu`, `keras_sigmoid`, `keras_tanh`,
`keras_gelu`, `keras_softmax`, `keras_maxpool2d`, `keras_avgpool2d`, `tf_add`,
`tf_multiply`, `tf_matmul`, `tf_concat`, `tf_reshape`, `tf_transpose`,
`tf_reduce`. TensorFlow is an optional dependency (lazy import); a separate
`/api/export_keras` endpoint + `src/runtime/codegen_tf.py` emit a
`tf.keras.Model`.

---

## 7. Node Contract

Every node is a subclass of `BaseNode` (in `src/nodes/base.py`):

```python
class MyNode(BaseNode):
    type_name = "my_node"       # unique registration key
    label = "My Node"            # display name
    category = "Math"            # UI grouping
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [Parameter("dim", default=0, dtype="int")]

    def execute(self, inputs, params, context):
        # inputs: dict of port_name -> TensorValue (or model object)
        # params: dict of parameter_name -> value (merged with defaults)
        # context: dict (may contain "graph_inputs" for subgraphs)
        # Returns: dict of port_name -> TensorValue (or other)
        ...
```

### Registration
Nodes are registered via `@register("type_name")` decorator (from
`src/runtime/registry.py`). Importing `src.nodes` triggers all registrations
as a side effect.

### Parameter types
- `kind`: `"scalar"` or `"any"` (tensor/list payload)
- `dtype`: optional hint — `"int"`, `"float"`, `"string"`, `"bool"`, `"data"`
- `default`: default value (may be `None`)
- `required`: if True, user must provide
- `choices`: optional allowed-value list

---

## 8. .riko File Format

`.riko` files are JSON, storing a complete graph workflow. See
`docs/FILE_FORMAT.md` for the full spec.

```json
{
  "version": "0.1",
  "metadata": {
    "name": "transformer",
    "description": "...",
    "inputs": [{"name": "input", "data_kind": "tensor"}],
    "outputs": [{"name": "output", "data_kind": "tensor"}]
  },
  "nodes": [...],
  "edges": [...],
  "settings": {}
}
```

### Subgraph References
A `.riko` file can reference another `.riko` via a `graph_reference` node (by
`file` path) or an `import` node (by bare module name, resolved against
`MODULE_SEARCH_PATHS` = `workflows/`, `examples/`, `examples/models/`):
- The referenced file must contain `graph_input(name="input")` and
  `graph_output(name="output")` nodes.
- This enables graph composition like Python function calls / `import xx`.

### Example .riko Files (in `examples/`)
- `transformer.riko` — token → embedding → transformer_encoder → linear
- `cnn.riko` — conv → relu → pool ×2 → flatten → linear (MNIST 28×28)
- `diffusion.riko` — conv → SiLU → conv (denoising step)
- `mamba.riko` — linear → transpose → conv1d → SiLU → transpose → linear
- `rwkv.riko` — linear → sigmoid → linear (channel mix)
- `rnn.riko` — linear → tanh → linear (RNN cell)
- `mnist_cnn.riko` — mnist_loader → CNN → cross_entropy_loss (training step)
- `transformer_pipeline.riko` — graph_input → embedding → transformer → linear → graph_output (subgraph-referencable)
- `mlp.riko` — graph_input → linear → relu → linear → graph_output (importable MLP block)
- `classifier.riko` — graph_input → import(mlp) → softmax → graph_output (demonstrates `import`)

---

## 9. Export Python (Codegen)

`src/runtime/codegen.py` generates a clean, self-contained `nn.Module` script
following the industry-standard `__init__` / `forward` split:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 64)
        self.enc = nn.TransformerEncoder(...)
        self.out = nn.Linear(64, 10, bias=True)

    def forward(self, input_):
        emb = self.emb(input_.long())
        enc = self.enc(emb)
        out = self.out(enc)
        return out
```

- Learnable layers (linear/conv/transformer/embedding/...) → `__init__`
- Inline ops (add/relu/sum/reshape/...) → `forward` body
- `graph_input` → `forward` parameter; `graph_output` → `return` statement
- `import` / `graph_reference` → recursively inlined as nested `nn.Module`
  classes (with cycle protection), so an importing graph exports to one file
- Runtime-data nodes (data loaders / model_loader) → clean `None` placeholder
  + comment (never injected inline into downstream expressions)
- Variable names are sanitized (keyword / reserved-word safe)
- Accessible via `POST /api/export_python` or UI MenuBar → File → Export .py

---

## 10. Frontend UI Structure

### Layout
```
MenuBar (dropdown: File / Run / Data / Help)
┌─────────────────┬──────────────────────┬──────────────────┐
│ NodeLibrary     │ GraphCanvas          │ SideInspector    │
│ (categorized,   │ (React Flow canvas,  │ (parameters,     │
│  collapsible,   │  right-click context │  inputs, outputs,│
│  searchable)    │  menu)               │  preview)        │
│ FileManager     │                      │                  │
│ (.riko file tree,│                      │                  │
│  import deps)   │                      │                  │
└─────────────────┴──────────────────────┴──────────────────┘
StatusPanel (Logs / Queue tabs)
```
Panels are modular, Blender-style areas (`Panel.tsx`): each panel has a
top-right control cluster with a type dropdown (switch the window to Node
Library / Files / Graph / Inspector / Status / Loss Curve), a split button
(detach a new panel), and a close button (merge it away). All panels are
separated by draggable splitters (`Splitter.tsx`) and can be resized
simultaneously.

### State Management
All state in `src/ui/store/graphStore.ts` (zustand):
- `nodeDefs`: NodeDef[] (loaded from `/api/nodes`, fallback to 3 local)
- `nodes`/`edges`: React Flow nodes/edges (current workflow)
- `workflows`: WorkflowDef[] (multi-workflow, IDE-style)
- `fileList` / `activeFileName`: disk-backed .riko file tree state
- `selectedNodeId`: currently selected node
- `status`: "idle" | "running" | "success" | "error"
- `results`: execution outputs (TensorPreview per node/port)
- `logs`: execution log messages
- `losses`: training loss history (for the loss curve panel)
- Actions: `addNode`, `onConnect`, `run`, `train`, `save`, `load`, `exportPython`,
  `loadNodeDefs`, `newWorkflow`, `switchWorkflow`, `updateParam`, `removeNode`,
  `refreshFiles`, `openFile`, `saveFileToDisk`, `importModule`

### Key Frontend Files
| File | Role |
|------|------|
| `src/ui/App.tsx` | Root component, loads nodeDefs on mount, resizable panel layout |
| `src/ui/store/graphStore.ts` | All state + actions (zustand) |
| `src/ui/components/MenuBar.tsx` | Dropdown menu bar (File/Run/Data/Help) |
| `src/ui/components/NodeLibrary.tsx` | Categorized, collapsible node list |
| `src/ui/components/GraphCanvas.tsx` | React Flow canvas + right-click context menu |
| `src/ui/components/ContextMenu.tsx` | Right-click searchable node menu |
| `src/ui/components/NodeCard.tsx` | Custom React Flow node (title, ports, result) |
| `src/ui/components/SideInspector.tsx` | Selected node params/inputs/outputs/preview |
| `src/ui/components/FileManager.tsx` | Blender-outliner style .riko file tree (open / import-as-node / save, with import dependencies) |
| `src/ui/components/StatusPanel.tsx` | Logs + queue tabs |
| `src/ui/components/Splitter.tsx` | Draggable panel splitters (vertical + horizontal) |
| `src/ui/components/Panel.tsx` | Modular panel system: type switch / split / close + content renderer |
| `src/ui/components/LossPanel.tsx` | Training loss curve (SVG line chart + Train buttons) |
| `src/ui/nodes/nodeDefinitions.ts` | Fallback NodeDef types (only 3 — real defs come from API) |
| `src/ui/styles.css` | All CSS (Material-inspired, 8px grid) |

---

## 11. Testing

### Python Tests (89 tests, all passing)
| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_node.py` | 19 | TensorValue, GraphDocument, constant/add/multiply |
| `tests/test_runtime.py` | 12 | Registry, executor validate/order/execute, cycle detection |
| `tests/test_backend.py` | 25 | Device, converter, torch_add/multiply, linear, relu, graph execution |
| `tests/test_api.py` | 12 | API nodes (abs/sigmoid/sub/matmul/sum/reshape/softmax/conv2d/layernorm), model_loader+inference E2E, codegen, subgraph reference E2E |
| `tests/test_subgraph.py` | 13 | import node, module resolution, clean codegen output, generated-code end-to-end run |

### Frontend Tests (8 tests, all passing)
| File | Coverage |
|------|----------|
| `src/ui/__tests__/graphStore.test.ts` | apiNodeToDef, addNode, onConnect, removeNode, load, updateParam |

### Known Test Gaps
- No frontend component tests (only store logic tested)
- No CI run verification (CI config exists but never triggered on GitHub)

---

## 12. Known Issues & Technical Debt

### Critical (affects user experience)
1. **Server must be running** — Without the API server, the frontend only shows
   3 fallback nodes and cannot execute/export. The NodeLibrary shows a red
   warning when < 10 nodes are loaded.
2. **Browser cache** — After code changes, users must hard-refresh
   (Cmd+Shift+R). The dev server (`--reload`) auto-reloads Python but not the
   browser.
3. **Toolbar.tsx still exists** but is unused (replaced by MenuBar.tsx). Safe
   to delete.

### Moderate (functionality gaps)
4. **Executor-level training is still pending** — the Train menu / loss curve
   panel train via codegen (export → exec → optimizer steps), which works for
   self-contained graphs (data loader + loss). The runtime executor itself
   still runs nodes with `torch.no_grad()` and recreates layers per call, so a
   native in-executor optimizer node is not yet implemented.
5. **Workflow persistence is partial** — The FileManager now lists/saves/loads
   `.riko` files to/from disk (via `/api/files*`), but the in-memory
   `newWorkflow`/`switchWorkflow` list is still not auto-persisted, and there
   is no multi-directory project management.
6. **graph_reference/import use fixed input/output names** — Only `input`/
   `output` are supported. No multi-port subgraph references.
7. **codegen stubs runtime-data nodes** — Data loaders (mnist/cifar/csv/...)
   and `model_loader` still emit clean `None` placeholders (they need runtime
   data). All other node types (ops, layers, reduce, shape, creation, loss,
   inference, import/graph_reference) are covered natively.
8. **NodeLibrary categories are flat** — No sub-categories (e.g., "Neural"
   contains both activations and layers mixed together).

### Minor (polish)
9. **No undo/redo** — Not implemented.
10. **No keyboard shortcuts** — No Ctrl+S, Ctrl+Z, Delete key handling.
11. **No node search in canvas** — Right-click context menu has search, but
    the left NodeLibrary search is separate.
12. **`.doc_backups/` directory** — Contains old Houdini-era docs. Safe to
    delete.
13. **`entropia-riko` directory name** — Legacy. Product is "Torch Node
    Studio" but the directory was never renamed.

---

## 13. Key Design Decisions

1. **TensorValue is pure Python** — `data` is number or nested list, not a
   torch tensor. This keeps core platform-neutral. Conversion happens in
   `src/backend/converter.py`.

2. **Nodes register on import** — `import src.nodes` triggers all `@register`
   decorators. The server does this at startup. The frontend fetches
   `/api/nodes` to get definitions.

3. **Factory pattern for batch node creation** — `api_nodes.py` uses
   `_unary()`, `_binary()`, `_reduce()` factories that create and register
   node classes. **IMPORTANT**: Factory parameters must use `_`-prefixed names
   (`_t`, `_l`, `_fn`) to avoid Python class-scope variable shadowing bugs
   (a previous bug where `type_name = type_name` in a class body caused
   `NameError`).

4. **torch is optional** — `src/backend/device.py` and `converter.py` use
   `try: import torch` with fallback. Core tests run without torch. Backend
   tests skip if torch unavailable.

5. **Venv at `.venv/`** — The system Python couldn't install packages
   (permission error on `/Users/faputa/Library/Python`). A project-local
   venv was created. All Python commands should use `.venv/bin/python`.

6. **Save downloads `.riko`** — The Save button creates a Blob and triggers a
   download with filename `workflow.riko`. Load accepts `.riko` and `.json`.
   The FileManager additionally saves/loads `.riko` files to/from `workflows/`
   on disk.

7. **Export generates nn.Module** — Not a flat script. The generated code has
   `class GraphModel(nn.Module)` with `__init__` (layer declarations) and
   `forward` (topological execution), plus nested classes for imported graphs.

---

## 14. Documentation Index

| File | Content |
|------|---------|
| `README.md` | English project overview, quick start, node list |
| `README_CN.md` | Chinese version |
| `SPEC.md` | Product specification |
| `ARCHITECTURE.md` | Layered architecture, module boundaries |
| `ROADMAP.md` | Stage 0-5 roadmap (all stages completed) |
| `docs/APP_SPEC.md` | App specification (layout, UI areas) |
| `docs/APP_ARCHITECTURE.md` | Source layout, responsibilities, deprecated directions |
| `docs/API.md` | Core contracts, node contract, parameter fields |
| `docs/NODE_SYSTEM.md` | Node requirements, categories, execution model |
| `docs/TORCH_BACKEND.md` | Torch backend responsibilities, device rules |
| `docs/DATA_FORMAT.md` | Graph document format, tensor IR, data kinds |
| `docs/UI_STANDARD.md` | UI style, layout, components, states |
| `docs/CROSS_PLATFORM.md` | Platform rules, path handling, device handling |
| `docs/COMFYUI_WORKFLOW_REFERENCE.md` | ComfyUI patterns borrowed |
| `docs/DESIGN.md` | Product feel, design references |
| `docs/FILE_FORMAT.md` | .riko file format spec, subgraph references |
| `docs/REBRAND_REFACTOR.md` | Houdini→standalone transition notes |

---

## 15. Quick Reference Commands

```bash
# === Setup (already done, for reference) ===
cd ~/Documents/torch-node/entropia-riko
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install

# === Run ===
.venv/bin/python -m uvicorn src.server.app:app --reload --port 8000  # Terminal 1
npm run dev                                                           # Terminal 2

# === Test ===
.venv/bin/python -m unittest discover -s tests -t .  # Python 89 tests
npm test                                              # Frontend 8 tests
npm run build                                         # tsc + vite build

# === Verify node count ===
.venv/bin/python -c "import src.nodes; from src.runtime.registry import default_registry; print(len(default_registry().list()))"
# Should print: 105

# === Run example ===
.venv/bin/python examples/transformer_demo.py
```

---

## 16. What Would Need Work Next

Prioritized by impact:

1. **Native in-executor training** — Training currently works by exporting the
   graph to Python and running optimizer steps there (Train menu + loss curve
   panel). A native executor path (persistent parameters, optimizer node,
   loss → backward → step) would remove the export round-trip.

2. **Full workflow persistence** — FileManager already lists/saves/loads `.riko`
   files to/from `workflows/`; remaining work is multi-directory project
   management and auto-persisting the in-memory workflow list.

3. **codegen for runtime-data nodes** — Data loaders and `model_loader` still
   emit `None` placeholders; could add optional runtime-loading codegen.

4. **Undo/redo** — Essential for a node editor.

5. **Node search in NodeLibrary** — Currently works but could be improved
   with fuzzy search.

6. **Canvas features** — Copy/paste nodes, duplicate, align, group/ungroup.

7. **Tensor preview improvements** — Show actual data values for small
   tensors, image preview for image_tensor data kind.

8. **More datasets** — FashionMNIST, CIFAR100, custom dataset from folder
   structure, web datasets.

9. **Model zoo** — Pre-built .riko files for common architectures (ResNet,
   VGG, BERT, etc.).

10. **Distributed execution** — Multi-GPU, batch processing queue.

---

## 17. File Count Summary

```
Python source:    ~41 files (src/ + tests/)
TypeScript source: 15 files (src/ui/)
Config:            7 files (package.json, tsconfig, vite, vitest, requirements, .gitignore, ci.yml)
Docs:             17 files (README + docs/ + examples)
Examples:          11 files (.riko + .py)
Total:            ~92 project files (excluding .venv, node_modules, dist)
```

---

*End of handoff document. The project is in a working state with 194 nodes,
89 Python tests, 8 frontend tests, all passing. The main remaining gaps are
native in-executor training and full multi-file workflow persistence.*
