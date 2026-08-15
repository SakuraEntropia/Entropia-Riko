# Architecture

## Summary

Entropia Riko is a standalone node UI application with a Torch execution backend.

The architecture is:

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

## Layers

### Standalone App UI

Owns the visible application:

- Toolbar.
- Node library.
- Graph canvas.
- Side inspector.
- Preview panel.
- Queue/status/log area.

### Graph Editor State

Owns UI graph state:

- Node positions.
- Edges.
- Selected node.
- Canvas viewport.
- UI metadata.

### Node Runtime

Owns execution:

- Validation.
- Dependency ordering.
- Execution queue.
- Node state.
- Error reporting.

### Tensor IR

Owns portable tensor data:

- Shape.
- Dtype.
- Device.
- Payload.
- Metadata.

### Torch Backend

Owns PyTorch behavior:

- Tensor conversion.
- Device detection.
- CPU/CUDA/MPS fallback.
- Model loading later.
- Inference later.

## Removed Direction

The old integration layer is deprecated.

Do not build:

- Houdini adapter.
- HDA generator.
- Houdini node wrapper.
- Houdini parameter adapter.
