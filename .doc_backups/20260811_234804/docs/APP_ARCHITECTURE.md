# Application Architecture

## Architecture Summary

Torch Node Studio is a standalone node editor with a Torch execution backend.

The architecture separates the visual editor from the runtime and backend.

```text
Standalone App UI
        ↓
Graph Editor State
        ↓
Node Runtime
        ↓
Tensor IR
        ↓
Torch Backend
```

## Main Layers

### App UI

Owns the visible interface.

Responsibilities:

- App shell.
- Toolbar.
- Graph canvas.
- Node library.
- Side inspector.
- Preview panel.
- Status bar.
- Settings view.

### Graph Editor State

Owns the visual and editable graph state.

Responsibilities:

- Node positions.
- Selected node.
- Edges.
- Port connections.
- Canvas viewport.
- UI-only metadata.

This layer is allowed to care about layout and selection.

### Node Runtime

Owns executable graph behavior.

Responsibilities:

- Graph validation.
- Dependency ordering.
- Node execution.
- Execution queue.
- Runtime errors.
- Node output cache.

This layer should not depend on UI components.

### Tensor IR

Owns the portable tensor value format.

Responsibilities:

- Shape.
- Dtype.
- Device.
- Payload reference.
- Metadata.

### Torch Backend

Owns PyTorch-specific execution.

Responsibilities:

- Convert Tensor IR to Torch tensors.
- Convert Torch tensors to Tensor IR.
- Detect devices.
- Run operations.
- Load models.
- Run inference.

## Frontend Direction

Recommended stack for the first app:

- React or equivalent component framework.
- Node graph library such as React Flow if using React.
- Local Python backend for Torch runtime if needed.
- JSON graph format for save/load.

If using a web app shell, the first screen must be the graph editor.

No marketing landing page.

## Backend Direction

The Torch runtime can start as a Python package.

The UI can communicate with it through:

- Local API.
- IPC.
- Direct Python calls if using a Python-native UI.
- JSON graph execution requests.

The exact transport can be chosen later.

## Graph Document Format

A saved workflow should be represented as data.

Recommended shape:

```json
{
  "version": "0.1",
  "nodes": [],
  "edges": [],
  "settings": {}
}
```

The UI should not be the only source of truth.

## Platform Rule

The app must remain compatible with:

- macOS
- Windows
- Linux

Current test environment:

- macOS

Do not hardcode platform-specific paths or shell commands.

## Deprecated Layer

The previous Houdini adapter layer is removed.

Do not create:

- `src/houdini`
- `hda_generator.py`
- `node_wrapper.py` for Houdini
- Houdini parameter adapter

Use generic names instead:

- `src/ui`
- `src/runtime`
- `src/backend`
- `src/core`
