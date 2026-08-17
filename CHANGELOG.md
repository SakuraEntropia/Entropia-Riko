# Changelog

All notable changes to Entropia Riko are documented here.

## [0.2.0] — AI workflow framework (unreleased)

The transition from a node-editor prototype to a reproducible AI workflow
framework. The UI is unchanged; everything lands in the backend, examples, and
docs.

### Added

- **Strong-typed port system** (`core/types.py`): 18 data kinds with a
  subtype hierarchy; `executor.validate` now rejects incompatible connections
  (e.g. `text → audio`).
- **IDE-style project system** (`entropia_riko/project/`):
  - `project.riko` manifest (identity, runtime, dependencies, workflows).
  - `requirements.riko` dependency manifest.
  - 5 project templates (Empty / Computer Vision / Diffusion / Audio / Video)
    with full folder structure + seed workflows.
  - Project browser APIs: templates / scan / validate / migrate / create.
- **Professional nodes** (`file_input`, `dataset`, `checkpoint_save`,
  `checkpoint_load`) with Houdini-style file pickers and project-relative paths.
- **Bake/cache system** (`project/cache.py`): artifact provenance + cache reuse.
- **Experiment records** (`project/experiment.py`): workflow/params/metrics/
  hardware captured under `experiments/experiment_NNN/`.
- **Workflow metadata + dependency graph** (`project/workflow.py`): name,
  version, category, I/O types, dependencies, topological ordering.
- **Real CV training pipeline example** (`examples/pipelines/image_training/`):
  dataset → train → checkpoint → inference → evaluation.

### Changed

- `TensorValue` now carries non-tensor kinds (file/folder/dataset/checkpoint/…).
- `MIGRATION.md` added documenting the 0.1.x → 0.2.0 plan.

## [0.1.x] — node editor prototype

- Node graph editor, ~200 nodes, code export, training, subgraphs, plugins.
- PyPI `entropia-riko` + npm `entropia-template-ui` published.
