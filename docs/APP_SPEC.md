# App Specification

## Purpose

Entropia Riko is a standalone, node-graph deep-learning editor. Users assemble
machine-learning workflows visually — connecting nodes on a graph canvas — and
the app validates, executes, exports, and trains those graphs.

It is a clean technical workspace that composes PyTorch functionality through a
node graph, with an optional TensorFlow/Keras export path. It must **not**
depend on Houdini.

## Product Direction

The editor should feel like:

- **ComfyUI-style** graph workflow — connect ports to build data flow.
- **Houdini-like** procedural node logic — reusable, parameterized node graphs.
- **Material Design-inspired** visual simplicity — clean, legible, focused UI.
- **Torch-first** runtime — PyTorch is the primary execution and export backend.

## What It Is

| Aspect | Implementation |
| --- | --- |
| Frontend | React + Vite SPA (`src/ui`), React Flow (`@xyflow/react`) graph canvas, Zustand state. Optional Electron desktop shell. |
| Backend | FastAPI (`src/server/app.py`) exposing the runtime over HTTP. |
| Execution | PyTorch; TensorFlow/Keras is an optional alternate export/execution backend. |
| Node library | 200+ nodes across built-ins and loadable plugins. |
| Persistence | Graphs saved as data (`.riko` JSON and `.ric` binary). |

## Features

### Graph editing
- Node library (left) for searching and adding nodes by name/category.
- Graph canvas (center) for placing, connecting, and arranging nodes.
- Side inspector (right) for editing the selected node's parameters.
- Toolbar for run / save / load / export / settings.
- Bottom status area for the execution queue, logs, and streaming training loss.

### Graph runtime
- Graph validation: registered node types, existing ports, required inputs connected, no cycles.
- Kahn topological ordering and execution.
- Per-node output caching and node-level error reporting with context.
- Multi-modal data kinds (`scalar`, `tensor`, `image_tensor`, `text`, `json`, `model`) flowing through ports.

### Code generation
- `export_python`: a self-contained, runnable `torch.nn.Module` script
  (learnable layers in `__init__`, ops in `forward`, `graph_input` → `forward` params,
  `graph_output` → `return`).
- `export_python_project`: a multi-file GitHub-style PyTorch project
  (`README.md`, `requirements.txt`, `src/<name>.py`).
- `export_keras`: a `tf.keras.Model` script with torch → TensorFlow node mapping.

### Training
- In-editor training of self-contained graphs (data loader + loss node).
- AdamW optimizer; per-step losses streamed as SSE/NDJSON for a live loss curve.

### Subgraphs / reuse
- `graph_input` / `graph_output` / `graph_reference` / `import` nodes.
- Referenced `.riko` graphs are inlined recursively at export and executed at runtime.

### Plugins
- User plugins as `plugins/<name>/plugin.json` + entry module (typically `nodes.py`).
- Enable/disable persisted in `plugins/state.json`; plugins installable by uploading a `.py` file.

### File management
- Disk-backed `.riko` / `.ric` files under `workflows/` and `examples/`.
- Working-folder tree, file explorer, import/export of files and folders.

## Architecture Overview

The app is a thin React shell over a Python runtime. The React UI holds the graph
as a `GraphDocument` and calls FastAPI endpoints (`/api/nodes`, `/api/execute`,
`/api/export_*`, `/api/train`, `/api/files`, …). The Python side validates and
executes the graph through the node registry and returns serialized `TensorValue`
results.

See `APP_ARCHITECTURE.md` for the layer breakdown, and `NODE_SYSTEM.md` /
`DATA_FORMAT.md` for the node contract and data model.
