# Roadmap

## Stage 0: Product Refactor

Status: current

Refactor the project into a standalone Torch node UI app.

Deliverables:

- Remove old integration-specific language.
- Add standalone app spec.
- Add app architecture.
- Add ComfyUI-style workflow reference.
- Preserve cross-platform and UI standards.

## Stage 1: Standalone UI Prototype

Build the first usable app screen.

Deliverables:

- App shell.
- Graph canvas.
- Node library.
- Side inspector.
- Status/queue/log area.
- Example node cards.
- Basic graph state.

## Stage 2: Core Runtime

Build executable graph behavior.

Deliverables:

- Tensor IR.
- Base node model.
- Registry.
- Graph validation.
- Dependency ordering.
- Add node.
- Multiply node.

## Stage 3: Torch Backend

Connect runtime to Torch-compatible computation.

Deliverables:

- CPU baseline.
- Device detection.
- Tensor conversion.
- Basic operation execution.
- Error reporting.

## Stage 4: Model Nodes

Add model-oriented workflows.

Deliverables:

- Model loader node.
- Inference node.
- Tensor preview.
- Output preview.

## Stage 5: Save/Load Workflows

Make graphs reproducible.

Deliverables:

- JSON graph format.
- Save workflow.
- Load workflow.
- Version metadata.
