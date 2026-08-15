# App Architecture

## System Shape

```text
React + Vite UI  (src/ui)
        │  HTTP  (FastAPI)
        ▼
API Server  (src/server/app.py)
        │
        ▼
Runtime  (src/runtime)   registry · executor · codegen · subgraph · trainer
        │
        ▼
Core  (src/core)         GraphDocument · TensorValue
        │
        ▼
Backend  (src/backend)   device detection · tensor conversion (torch optional)
```

The UI never touches torch or the executor directly. It edits a `GraphDocument`
(plain JSON-able data) and calls the server. The server runs the graph through
the registry/executor and returns serialized `TensorValue` results.

## Source Layout

```text
entropia-riko/
├── src/
│   ├── ui/            # React + Vite frontend (app shell, canvas, inspector, library)
│   ├── core/          # GraphDocument, NodeModel/EdgeModel/PortModel, TensorValue
│   ├── runtime/       # Registry, executor, codegen (torch + TF), subgraph, trainer
│   ├── backend/       # Device detection + torch<->TensorValue conversion
│   ├── nodes/         # Built-in node definitions (math, torch_ops, subgraph, tf_ops)
│   ├── plugins/       # Plugin loader (manifests, state, upload)
│   └── server/        # FastAPI app
├── plugins/           # User plugins (plugin.json + nodes.py), plugins/state.json
├── workflows/         # Saved .riko / .ric graphs
├── examples/          # Example graphs (also searched for imports)
├── electron/          # Optional desktop shell
├── index.html         # Vite entry → /src/ui/main.tsx
├── package.json       # React / Vite / React Flow / Zustand
└── vite.config.ts     # Dev server :5173, proxies /api → :8000
```

## Layer Responsibilities

### `src/ui` — frontend
The React + Vite SPA. Owns the app shell, toolbar, node library, graph canvas
(React Flow / `@xyflow/react`), side inspector, previews, and the status/log
panels. State is held with Zustand; all execution, node-list, file, plugin, and
training calls go through the `/api` HTTP layer. The Electron shell wraps the
same UI for desktop use.

### `src/core` — data model (framework-free)
- `document.py`: `GraphDocument`, `NodeModel`, `EdgeModel`, `PortModel` —
  the saveable graph. Serializes to dict/JSON and to the binary `.ric` format.
- `tensor.py`: `TensorValue`, the portable tensor IR (data/shape/dtype/device/
  metadata/kind) plus shape inference and broadcasting helpers. Pure Python —
  does **not** import torch.

### `src/runtime` — execution & export
- `registry.py`: node type registry (`@register`, `register`, `get`, `list`, `unregister`).
- `executor.py`: `validate` (types/ports/required inputs/cycles) →
  `execution_order` (Kahn topo sort) → `execute`.
- `codegen.py`: `export_python` (torch `nn.Module`) and `export_python_project`
  (multi-file GitHub layout). Recursively inlines `import`/`graph_reference`.
- `codegen_tf.py`: `export_keras` (`tf.keras.Model`) with torch → TF mapping.
- `subgraph.py`: `graph_input`/`graph_output`/`graph_reference`/`import` support
  and `.riko` module resolution.
- `trainer.py`: AdamW training, `iter_losses` streaming, `train_graph`.

### `src/backend` — torch bridge
Device detection (`resolve_device`, `is_torch_available`) and tensor conversion
(`to_torch`, `from_torch`) between `TensorValue` and `torch.Tensor`. torch is
**optional**: when unavailable, only CPU is supported and conversion raises clear
errors. `core`/`runtime` never import this layer directly.

### `src/nodes` — built-in nodes
Reusable node definitions. Importing the package registers built-ins into the
default registry. Modules: `math`, `torch_ops`, `subgraph` (graph input/output/
reference), and `tf_ops` (TensorFlow/Keras nodes — optional backend).

### `src/plugins` — plugin system
`loader.py` scans `plugins/*/plugin.json`, imports each entry module (which
registers nodes via `@register`), and manages enable/disable state persisted in
`plugins/state.json`. Supports uploading plugins from raw `.py` source.

### `src/server` — API layer
A FastAPI app (`src/server/app.py`) that exposes the runtime over HTTP so the
React UI can list nodes and execute/export/train graphs. Key endpoints:

- `/api/health`, `/api/nodes`, `/api/execute`
- `/api/export_python`, `/api/export_keras`, `/api/export_binary`, `/api/export_project`
- `/api/train`, `/api/train/stream` (SSE/NDJSON)
- `/api/plugins`, `/api/plugins/toggle`, `/api/plugins/upload`
- `/api/files*`, `/api/project/*`, `/api/fs/*` (file management)

Run with: `uvicorn src.server.app:app --reload --port 8000`.

## Data Flow (execute)

```text
UI edits GraphDocument (JSON)
  → POST /api/execute
  → GraphDocument.from_dict
  → runtime.execute (validate → execution_order → per-node execute)
  → {node_id: {port: TensorValue}}
  → server serializes TensorValue (shape/dtype/device/summary/data_kind/data, image as base64)
  → UI renders summaries / previews
```

## Deprecated / Out of Scope

Do not create or implement any Houdini integration:

- `src/houdini`
- HDA generator, Houdini adapter, Houdini node wrapper, Houdini parameter bridge
