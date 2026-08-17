"""Torch backend layer (TORCH_BACKEND.md).

Owns device detection and tensor conversion. torch is optional:
when unavailable, only CPU is supported and conversion raises clear
errors. Core/runtime never import this layer directly.
"""
from .device import TORCH_AVAILABLE, is_torch_available, resolve_device

if TORCH_AVAILABLE:  # pragma: no cover - branch by environment
    from .converter import from_torch, to_torch
else:
    to_torch = None  # type: ignore
    from_torch = None  # type: ignore

__all__ = [
    "resolve_device",
    "TORCH_AVAILABLE",
    "is_torch_available",
    "to_torch",
    "from_torch",
]
