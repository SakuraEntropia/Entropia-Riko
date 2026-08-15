# Torch Backend

The **torch backend** (`src/backend/`) is the layer that owns everything
PyTorch-specific. It is the *only* place that imports `torch`, and `torch` is an
**optional** dependency: when it is missing, the app still runs on CPU and the
backend raises clear, actionable errors instead of crashing.

## Layering

```
src/core/      TensorValue + GraphDocument — pure Python, no torch imports.
src/runtime/   Executor, registry, codegen, trainer, subgraph.
src/backend/   Device detection + TensorValue <-> torch.Tensor conversion.
```

- `core` and the runtime `executor` never import torch (see their module
  docstrings).
- `src/backend/__init__.py` exports the public surface:

```python
from .device import resolve_device, TORCH_AVAILABLE, is_torch_available
if TORCH_AVAILABLE:
    from .converter import to_torch, from_torch
```

  When torch is unavailable, `to_torch` / `from_torch` are `None` rather than
  raising at import time.

---

## Device detection (`src/backend/device.py`)

The device rule is simple and safe:

> **CPU is the guaranteed baseline. CUDA and MPS may only be used after
> checking availability.**

### `resolve_device(device="cpu")`

Resolves a device request to a `torch.device` (or the string `"cpu"` when torch
is absent):

| Request  | Result                                                            |
|----------|-------------------------------------------------------------------|
| `"cpu"`  | `torch.device("cpu")` — always available.                         |
| `"cuda"` | `torch.device("cuda")`; raises `RuntimeError` if CUDA is unavailable. |
| `"mps"`  | `torch.device("mps")`; raises `RuntimeError` if MPS is unsupported. |
| `"auto"` | Best available, in order: **CUDA → MPS → CPU**. Never raises.     |
| a `torch.device` | Passed through unchanged.                               |
| anything else | `ValueError` (unknown device).                          |

Without torch, `"cpu"` and `"auto"` both return `"cpu"`; any explicit CUDA/MPS
request raises `RuntimeError` telling the user to install torch.

MPS availability is guarded carefully because `torch.backends.mps` may not exist
on all builds:

```python
def _mps_available() -> bool:
    return bool(TORCH_AVAILABLE
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available())
```

### Availability helpers

- `TORCH_AVAILABLE` — module constant set by a `try: import torch` guard.
- `is_torch_available() -> bool` — returns that constant.

---

## TensorValue ↔ torch conversion (`src/backend/converter.py`)

`TensorValue` (`src/core/tensor.py`) is the portable tensor IR that flows through
the graph. It carries `data`, `shape`, `dtype`, `device`, `metadata`, and an
optional `kind` — all pure-Python so it never depends on torch. The backend
converts between it and `torch.Tensor` without losing shape / dtype / device /
metadata.

### dtype mapping

| TensorValue dtype | torch dtype  |
|-------------------|--------------|
| `float32`         | `torch.float32` |
| `float64`         | `torch.float64` |
| `float16`         | `torch.float16` |
| `int32`           | `torch.int32`   |
| `int64`           | `torch.int64`   |
| `int16`           | `torch.int16`   |
| `int8`            | `torch.int8`    |
| `uint8`           | `torch.uint8`   |
| `bool`            | `torch.bool`    |

Unknown dtype strings fall back to `float32` in both directions, so a graph
never crashes on an unrecognized dtype string.

### `to_torch(value, device=None)`

```python
def to_torch(value: TensorValue, device: Any = None):
    dtype = _DTYPE_TO_TORCH.get(value.dtype, torch.float32)
    tensor = torch.tensor(value.data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
```

Builds a `torch.Tensor` from the TensorValue payload; pass a `device` (typically
the result of `resolve_device`) to move it to CUDA/MPS/CPU.

### `from_torch(tensor, metadata=None)`

```python
def from_torch(tensor, metadata=None) -> TensorValue:
    dev = str(tensor.device)
    cpu = tensor.detach().cpu()
    dtype = _TORCH_TO_DTYPE.get(cpu.dtype, "float32")
    shape = tuple(int(s) for s in cpu.shape)
    return TensorValue(cpu.tolist(), shape=shape, dtype=dtype,
                       device=dev, metadata=metadata)
```

Detaches and moves to CPU, preserves the original device as a string (e.g.
`"cuda:0"`, `"mps:0"`, `"cpu"`), and stores the payload as a nested Python list
so it can be JSON-serialized by the API.

### When torch is missing

Both functions raise a `RuntimeError` with the project's standard error shape
(`what` / `where` / `how_to_fix`), e.g.:

```text
what: torch 不可用，无法转换为 torch.Tensor。
where: backend.converter.to_torch
how_to_fix: 安装 torch。
```

---

## Responsibilities

- Tensor conversion (`TensorValue` ↔ `torch.Tensor`).
- Device detection and CPU/CUDA/MPS fallback.
- Raising clear, localized errors when torch or a requested device is
  unavailable.
- Serving as the single torch import boundary; core/runtime stay torch-free.
