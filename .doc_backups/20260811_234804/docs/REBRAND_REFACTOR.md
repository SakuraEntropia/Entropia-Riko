# Rebrand and Refactor Plan

## Decision

The project is no longer a Houdini integration project.

The new product direction is:

Build a standalone Torch node UI application with Houdini-like procedural node logic and ComfyUI-style graph workflow.

The software should provide its own interface, node graph, side inspector panel, execution engine, and Torch backend integration.

## Old Direction To Remove

Remove or rewrite all language that implies:

- Houdini plugin.
- HDA generation.
- Houdini adapter.
- Houdini Python API dependency.
- Houdini as the UI host.
- Houdini as a required runtime.

The project may still use "procedural node workflow" as a design reference, but not as a product dependency.

## New Direction

The application should be a standalone node-based Torch workspace.

Core product:

- Visual node graph editor.
- Side inspector panel.
- Torch tensor and model execution backend.
- Node library.
- Graph execution queue.
- Result preview panel.
- Project save/load.
- Cross-platform desktop or local web UI.

## Product Name

Use a neutral product name.

Recommended temporary name:

`Torch Node Studio`

Avoid names that imply Houdini dependency.

## Architecture Change

Old architecture:

```text
Houdini UI
  -> Houdini adapter
  -> node graph
  -> tensor IR
  -> Torch backend
```

New architecture:

```text
Standalone UI
  -> graph editor
  -> node runtime
  -> tensor IR
  -> Torch backend
```

## New Layer Responsibilities

### UI Layer

Owns the visible application:

- Graph canvas.
- Node cards.
- Side inspector.
- Node search menu.
- Toolbar.
- Preview panel.
- Execution status.
- Settings panel.

### Graph Editor Layer

Owns visual graph interaction:

- Node placement.
- Node selection.
- Port connection.
- Canvas pan and zoom.
- Grouping.
- Visual validation state.

### Runtime Layer

Owns graph execution:

- Graph validation.
- Dependency ordering.
- Execution queue.
- Node state.
- Error reporting.
- Cache rules.

### Tensor IR Layer

Owns data passed between nodes:

- Tensor values.
- Shape.
- Dtype.
- Device.
- Metadata.

### Torch Backend Layer

Owns Torch integration:

- Tensor conversion.
- Model loading.
- Inference.
- Training utilities later.
- Device detection.
- CPU/CUDA/MPS fallback.

## ComfyUI-Inspired Concepts

Borrow the workflow ideas, not the exact implementation.

Useful ideas:

- Node graph as the main screen.
- Node library/search for adding nodes.
- Side panel for selected node settings.
- Queue-based execution.
- Preview outputs.
- Save/load workflows.
- Graph as a reproducible computation document.

Do not copy ComfyUI branding or UI directly.

## Refactor Tasks

1. Rename product docs from Houdini-specific language to standalone Torch node language.
2. Replace `torch_houdini_node` concept with `torch_node_studio` concept in docs.
3. Replace Houdini adapter layer with standalone UI and graph editor layers.
4. Keep procedural node logic as an inspiration only.
5. Keep cross-platform rules.
6. Keep Material Design-inspired UI standard.
7. Add ComfyUI-style workflow document.
8. Rebuild roadmap around app UI first, not Houdini plugin first.

## Coding Agent Rule

Any coding agent must treat previous Houdini integration language as deprecated.

If an existing file mentions Houdini integration, rewrite it as:

- "procedural node workflow"
- "standalone node editor"
- "graph canvas"
- "side inspector"
- "Torch backend"

Do not implement Houdini integration unless the user explicitly asks for it in a future phase.
