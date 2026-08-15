# Torch Backend

## Purpose

The Torch backend is responsible for PyTorch-specific computation.

The rest of the system should treat PyTorch as an execution backend, not as the entire architecture.

## Responsibilities

The backend should own:

- Conversion between Tensor IR and PyTorch tensors.
- Device selection.
- Model loading.
- Inference.
- Training utilities.
- Backend-specific error handling.

## Non-Responsibilities

The backend should not own:

- Houdini UI.
- Node graph traversal.
- HDA generation.
- General node registration.
- Product-level workflow decisions.

## Device Handling

The backend should support CPU first.

GPU support should be optional and detected safely.

Device selection should allow:

- `cpu`
- `cuda` when available
- `mps` on supported Apple systems
- `auto`

## Tensor Conversion

Tensor IR values should be convertible to PyTorch tensors without losing:

- Shape.
- Dtype.
- Device preference.
- Metadata when possible.

PyTorch tensors should be convertible back into Tensor IR values.

## Model Interface

Models should be wrapped behind a small interface.

Expected operations:

- Load model.
- Prepare inputs.
- Run inference.
- Return Tensor IR outputs.

## Training Interface

Training is not part of the first implementation, but the architecture should leave room for it.

Future training utilities may include:

- Dataset adapters.
- Loss functions.
- Optimizer configuration.
- Checkpoint saving.
- Progress logging.

## First Backend Scope

The first Torch stage should implement:

- Basic tensor conversion.
- CPU execution.
- Simple inference wrapper.
- Linear node support.

## Agent Guidance

When implementing this layer:

- Keep Torch imports inside backend or Torch-specific nodes.
- Make CPU behavior reliable before adding acceleration.
- Do not assume Houdini is available.
- Keep model configuration external and serializable.
