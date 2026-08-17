# Architecture

Entropia Riko is a **node-graph deep-learning editor** with a React frontend, a
FastAPI backend, and a pure-Python execution runtime on top of PyTorch. This
document describes the current structure so a new maintainer can navigate the
code in one pass.

## High-level data flow

```text
React UI (entropia-template-ui)      ← node graph, inspector, panels, zustand store (npm package)
        │  HTTP (JSON)
        ▼
FastAPI server (entropia_riko/server)
        │
        ▼
Runtime (entropia_riko/runtime)      ← registry, executor, codegen, trainer, subgraphs
        │
        ▼
Nodes (entropia_riko/nodes)          ← 190+ registered node types
        │
        ▼
Core IR (entropia_riko/core)         ← TensorValue + GraphDocument (.riko/.ric)
        │
        ▼
Backend (entropia_riko/backend)      ← torch / tensorflow conversion, device detection
```

The runtime is **frontend-agnostic**: the UI talks to it only over HTTP, and
everything below `server/` is a plain Python library (`pip install
entropia-riko`) with no React dependency.

## Repository layout

```text
entropia-riko/
├── entropia_riko/            # the Python package (import name: entropia_riko)
│   ├── core/                 # portable IR: TensorValue + GraphDocument
│   ├── nodes/                # node definitions (registered into the runtime registry)
│   │   ├── base.py           # BaseNode / NodeInput / NodeOutput / Parameter contract
│   │   ├── math/             # scalar/tensor arithmetic primitives
│   │   ├── torch_ops/        # layers, activations, losses, loaders, inference, model io
│   │   ├── tf_ops/           # optional TensorFlow/Keras equivalents
│   │   └── subgraph/         # graph_input / graph_output / graph_reference / import
│   ├── runtime/              # the engine (see below)
│   ├── backend/              # torch↔IR conversion, device resolve, TF conversion
│   ├── plugins/              # user-plugin loader (plugins/*/plugin.json)
│   ├── server/               # FastAPI app (thin HTTP layer over the runtime)
│   └── utils/                # config + logging helpers
├── frontend/                 # thin React entry: mounts the entropia-template-ui npm package
├── tests/                    # Python unittest suite (runs against entropia_riko)
├── examples/                 # .riko example graphs (train/infer pairs)
├── templates/project/        # preset tree for "New Project"
├── electron/                 # optional desktop shell
├── scripts/                  # release + brand asset helpers
├── public/                   # brand assets (logo, hero, favicons)
├── site/                     # GitHub Pages landing page
└── docs/                     # user + developer documentation
```

The React UI is **fully decoupled**: it ships as the `entropia-template-ui`
npm package (GitHub: `SakuraEntropia/Entropia-Template-UI`). This repo keeps
only `frontend/main.tsx`, a thin entry that mounts that editor against this
repo's `/api`. The Python package has no React dependency.

## The runtime engine (`entropia_riko/runtime`)

This is the heart of the project. Each module has one job:

| Module | Responsibility |
|---|---|
| `registry.py` | `Registry` maps node type names → classes; `@register` decorator. |
| `executor.py` | `execute()` validates the graph, topologically sorts it, then runs each node. |
| `codegen.py` | `export_python()` compiles a graph into a runnable `torch.nn.Module` script. |
| `codegen_tf.py` | `export_keras()` compiles a graph into a `tf.keras.Model`. |
| `trainer.py` | Trains a self-contained graph (data loader + loss) and can save the fitted state_dict. |
| `subgraph.py` | Resolves/loads referenced `.riko` graphs; runs subgraphs (`run_subgraph`). |

**Two execution modes exist and are intentionally separate:**

- **Live execution** (`executor.py`) runs nodes eagerly; `model` values flow
  through the graph as real objects (`inference`, `save_model`, `model_loader`).
- **Compile mode** (`codegen.py`) turns the graph into one `nn.Module` class for
  export or training. This is what `trainer.py` uses.

## The node contract (`entropia_riko/nodes/base.py`)

Every node is a subclass of `BaseNode` declaring:

- `type_name` / `label` / `category` (registry + UI metadata),
- `inputs: list[NodeInput]` / `outputs: list[NodeOutput]` (ports),
- `parameters: list[Parameter]` (serialized user settings),
- `execute(inputs, params, context) -> dict` (the actual computation).

Nodes register themselves with `@register("type_name")`. `import
entropia_riko.nodes` imports every built-in module and populates the default
registry; plugins register the same way.

## The IR (`entropia_riko/core`)

- **`TensorValue`** (`tensor.py`) is the portable value flowing through a live
  graph: `data` + `shape` + `dtype` + `device` + `metadata`, with kinds
  `scalar | tensor | image_tensor | text | json | model`.
- **`GraphDocument`** (`document.py`) is the serialized graph: `version`,
  `metadata`, `nodes`, `edges`, `settings`. `.riko` is JSON; `.ric` is the same
  document zlib-compressed behind an `ERIK` magic header.

## Dependency direction

Dependencies point **downward only** — no circular imports:

```text
server → runtime → nodes → core
server → plugins → runtime → core
runtime → backend → core
```

`server` may import `runtime` and `nodes`, but `runtime`/`nodes` never import
`server`. The frontend (`frontend/` + the `entropia-template-ui` package) is
independent and talks to `server` over HTTP only.

## Adding a node

1. Create a `BaseNode` subclass (in `nodes/math/`, `nodes/torch_ops/`, …).
2. Decorate it with `@register("type_name")`.
3. Import its module from `nodes/__init__.py` so it registers on startup.
4. Add a unit test under `tests/`.

## Adding an API endpoint

1. Add the route in the relevant router under `server/routers/` (or create one).
2. Include the router in `server/app.py`.
3. Add a test under `tests/` (see `tests/test_api.py`).

## Where things live (quick reference)

| I want to… | Look at |
|---|---|
| change how a node computes | `nodes/…/xxx.py` |
| add a new node type | `nodes/…/` + `@register` + `nodes/__init__.py` |
| change graph execution | `runtime/executor.py` |
| change code export | `runtime/codegen.py` |
| change training | `runtime/trainer.py` |
| change the API | `server/routers/` + `server/app.py` |
| change the UI | `frontend/` (thin entry) + `Entropia-Template-UI` repo (React + zustand) |
| change the file format | `core/document.py` + `core/tensor.py` |
