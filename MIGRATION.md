# Migration Plan — 0.1.x → 0.2.0 (AI workflow framework)

This is the working migration plan for turning Entropia Riko from a node-editor
prototype into a reproducible AI workflow framework. **The UI is frozen** — all
changes land in the backend (core/runtime/nodes/server), examples, and docs.

## 1. Current state (0.1.x)

Already present and reusable:

- **Node contract** — `BaseNode` + `NodeInput`/`NodeOutput`/`Parameter` + `@register`; ~200 nodes.
- **IR** — `TensorValue` with kinds `scalar | tensor | image_tensor | model | text | json`; `GraphDocument` (`.riko`/`.ric`).
- **Runtime** — `registry`, `executor` (validate → topo sort → execute), `codegen` (PyTorch/Keras export), `trainer` (train + save state_dict), `subgraph` (run/import).
- **Model I/O** — `save_model` / `model_loader` (safetensors + torch), `/api/train?save_path=`.
- **Server** — modular routers (`health/plugins/graph/files/train/project/fs`), project mini file-manager with a working-directory tree.
- **UI** — decoupled into `entropia-template-ui` (npm) + thin `frontend/` entry.

## 2. Gaps to close

| Area | Gap |
|---|---|
| Port types | No strong typing: only 6 kinds, **no connection compatibility check** in `validate`. |
| Project system | No `project.riko` manifest, no templates, no project browser APIs (`create/open/scan/validate/migrate`). |
| Data/model mgmt | No `DATASET`/`CHECKPOINT` nodes, no asset registry, paths are absolute. |
| Reproducibility | No experiment records, no unified bake/cache system. |
| Workflow mgmt | Single-file graphs only; no multi-workflow workspace, versioning, or dependency graph. |
| Examples | Mostly toy; need real AI pipelines (CV / diffusion / audio / video). |

## 3. Phases (priority order)

1. **Port type system** — extend `TensorValue` kinds + typed connection validation (foundation).
2. **Project system** — `project.riko` manifest, `requirements.riko`, project templates, project browser APIs.
3. **Professional nodes** — `file_input`, `dataset`, `checkpoint_save/load` (model/training/inference exist).
4. **Bake/Cache + Experiment** — artifact ledger + experiment records for reproducibility.
5. **Workflow management** — multi-workflow workspace + dependency/version metadata.
6. **Examples** — replace toy graphs with real multimodal pipelines.
7. **Docs + release** — `README`, `docs/`, changelog; publish PyPI/npm/GitHub `0.2.0`.

## 4. Key file map

| Change | Files |
|---|---|
| Port kinds + compatibility | `core/tensor.py`, `runtime/executor.py` (`validate`), new `core/types.py` (compat table) |
| Project manifest + templates | new `entropia_riko/project/` (`manifest.py`, `templates.py`, `workspace.py`), `server/routers/project.py` |
| Professional nodes | `nodes/torch_ops/` (`dataset.py`, `checkpoint.py`, `file_io.py`) |
| Bake/cache + experiment | new `entropia_riko/project/cache.py`, `entropia_riko/project/experiment.py` |
| Examples | `examples/**` (rewrite), `templates/project/**` |
| Docs | `README.md`, `docs/*.md`, `CHANGELOG.md` |

## 5. Acceptance criteria (0.2.0)

A new user can: create a project → pick an AI template → get a full folder
structure + default workflows → load dataset → load model → train → save
checkpoint → run inference → export results → reopen the project and continue,
without manually managing files.
