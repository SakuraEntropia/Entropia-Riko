"""TensorFlow conversion (optional backend).

Converts between TensorValue (core IR) and ``tf.Tensor``. TensorFlow is an
optional dependency: conversion raises a clear error when TF is not installed.
The rest of the editor (torch backend) is unaffected.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    tf = None  # type: ignore
    TF_AVAILABLE = False

from ..core.tensor import TensorValue

_DTYPE_TO_TF = {
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "int32": "int32",
    "int64": "int64",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}


def is_tf_available() -> bool:
    return TF_AVAILABLE


def to_tf(value: TensorValue, dtype: Optional[str] = None):
    """Convert a TensorValue to a tf.Tensor."""
    if not TF_AVAILABLE:
        raise RuntimeError(
            "what: TensorFlow 不可用，无法转换为 tf.Tensor。\n"
            "where: backend.tf_converter.to_tf\n"
            "how_to_fix: 安装 tensorflow（pip install tensorflow）。"
        )
    dt = _DTYPE_TO_TF.get(dtype or value.dtype, "float32")
    return tf.convert_to_tensor(value.data, dtype=dt)


def from_tf(tensor, metadata: Optional[Dict[str, Any]] = None) -> TensorValue:
    """Convert a tf.Tensor back to a TensorValue (pure-Python payload)."""
    if not TF_AVAILABLE:
        raise RuntimeError(
            "what: TensorFlow 不可用，无法从 tf.Tensor 转换。\n"
            "where: backend.tf_converter.from_tf\n"
            "how_to_fix: 安装 tensorflow。"
        )
    arr = tensor.numpy()
    return TensorValue(
        arr.tolist(),
        shape=tuple(int(s) for s in arr.shape),
        dtype=str(tensor.dtype.name),
        device="cpu",
        metadata=metadata,
    )
