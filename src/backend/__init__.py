"""Torch backend layer (TORCH_BACKEND.md).

Owns device detection and tensor conversion. torch is optional:
when unavailable, only CPU is supported and conversion raises clear
errors. Core/runtime never import this layer directly.
"""
from .device import resolve_device, TORCH_AVAILABLE, is_torch_available

if TORCH_AVAILABLE:  # pragma: no cover - branch by environment
    from .converter import to_torch, from_torch
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
