"""Tensor conversion (TORCH_BACKEND.md).

Converts between TensorValue (core IR) and torch.Tensor without losing
shape / dtype / device / metadata. Requires torch; raises clear errors
when torch is unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from ..core.tensor import TensorValue

# TensorValue dtype string -> torch dtype (and the reverse map). Missing kinds
# fall back to float32 so a graph never crashes on an unknown dtype string.
_DTYPE_TO_TORCH = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
} if TORCH_AVAILABLE else {}

_TORCH_TO_DTYPE = {v: k for k, v in _DTYPE_TO_TORCH.items()}


def to_torch(value: TensorValue, device: Any = None):
    """Convert a TensorValue to a torch.Tensor."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "what: torch 不可用，无法转换为 torch.Tensor。\n"
            "where: backend.converter.to_torch\n"
            "how_to_fix: 安装 torch。"
        )
    dtype = _DTYPE_TO_TORCH.get(value.dtype, torch.float32)
    tensor = torch.tensor(value.data, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def from_torch(tensor, metadata: Optional[Dict[str, Any]] = None) -> TensorValue:
    """Convert a torch.Tensor back to a TensorValue (pure-Python payload)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "what: torch 不可用，无法从 torch.Tensor 转换。\n"
            "where: backend.converter.from_torch\n"
            "how_to_fix: 安装 torch。"
        )
    dev = str(tensor.device)
    cpu = tensor.detach().cpu()
    dtype = _TORCH_TO_DTYPE.get(cpu.dtype, "float32")
    shape = tuple(int(s) for s in cpu.shape)
    return TensorValue(
        cpu.tolist(),
        shape=shape,
        dtype=dtype,
        device=dev,
        metadata=metadata,
    )
