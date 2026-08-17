# Project System

Entropia Riko treats a folder as a **project**, not "a folder of JSON files".
A project is any directory containing a `project.riko` manifest; the project
system provides templates, a manifest, dependency tracking, asset management,
bake/cache, experiments, and multi-workflow organization.

## Manifest (`project.riko`)

```json
{
  "version": "1.0",
  "project": {
    "name": "MyProject",
    "version": "0.1.0",
    "engine_version": "0.2.0",
    "template": "computer_vision",
    "created": "2026-01-01T00:00:00+00:00"
  },
  "runtime": { "gpu": false, "cuda": null, "python": "3.12" },
  "dependencies": { "python": ["torch"], "models": [], "plugins": [] },
  "workflows": {
    "default": "workflows/01_train.riko",
    "all": ["workflows/00_dataset.riko", "workflows/01_train.riko", "..."]
  }
}
```

- **project** — identity (`name`, `version`, `engine_version`, `template`, `created`).
- **runtime** — GPU/CPU and Python environment.
- **dependencies** — Python packages, models, plugins the project needs.
- **workflows** — the default workflow and the full ordered list.

## Requirements (`requirements.riko`)

A companion manifest listing runtime requirements (Python packages, models,
plugins, GPU) so the system can check for missing dependencies.

## Templates

`POST /api/project/new` with `{dir, name, template, gpu}` materializes a full
structure. Templates:

| id | label | seeded workflows |
|---|---|---|
| `empty` | Empty AI Project | `workflows/main.riko` |
| `computer_vision` | Computer Vision | dataset / train / inference / evaluation |
| `diffusion` | Diffusion / Generative AI | prepare / training / generation / batch render |
| `audio` | Audio AI | prepare / training / generation |
| `video` | Video AI | frame extract / processing / export |

Each template also creates `datasets/`, `models/`, `checkpoints/`, `configs/`,
`outputs/`, `logs/`, `experiments/`, etc.

## Project browser APIs

| Endpoint | Purpose |
|---|---|
| `GET /api/project/templates` | list templates |
| `POST /api/project/new` | create project from template |
| `GET /api/project/scan` | manifest + tree + workflows |
| `POST /api/project/validate` | validate the project |
| `POST /api/project/migrate` | migrate/upgrade an old project |
| `POST /api/project/experiment` | record an experiment |
| `GET /api/project/experiments` | list experiments |

## Asset management

Assets are addressed by **project-relative paths** (`datasets/raw`,
`checkpoints/model.safetensors`) rather than absolute paths, so projects are
portable. The `file_input` / `dataset` / `checkpoint_*` nodes resolve relative
paths against the working root (the `/api/execute` context).

## Bake / cache

Generated artifacts are stored under `bakes/<name>/` with a `metadata.json`
sidecar (source node, parameters, workflow version, dependencies, timestamp).
`is_cache_valid(name, params)` enables cache reuse: recompute only when
parameters change.

## Experiments

Each run can be recorded under `experiments/experiment_NNN/` capturing
`workflow.json`, `parameters.json`, `metrics.json`, `metadata.json` (seed +
hardware), plus `outputs/` and `logs/` — the goal is reproducibility.

## Workflows

A project holds multiple workflow documents. Each carries metadata
(`name`, `version`, `category`, `dependencies`, `input_types`, `output_types`),
and `dependency_order()` topologically sorts them so the IDE understands the
pipeline (dataset → training → inference → evaluation).
