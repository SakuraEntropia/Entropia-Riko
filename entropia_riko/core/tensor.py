"""Tensor IR (DATA_FORMAT.md).

Portable tensor data carrying shape / dtype / device / payload / metadata.
Pure Python; does not import torch. The Torch backend (entropia_riko/backend) is
responsible for converting between TensorValue and torch.Tensor.
"""
from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Dict, Optional, Tuple, Union

Number = Union[int, float]

# DATA_FORMAT.md — first data kinds.
DATA_KINDS: Tuple[str, ...] = (
    "scalar",
    "tensor",
    "image_tensor",
    "model",
    "text",
    "unknown",
)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def infer_shape(data: Any) -> Tuple[int, ...]:
    """Infer shape from a number / nested list."""
    if _is_number(data):
        return ()
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        return (len(data),) + infer_shape(data[0])
    raise TypeError(
        f"what: 无法推断 shape，不支持的 data 类型 {type(data).__name__}。\n"
        f"where: core.tensor.infer_shape\n"
        f"how_to_fix: 传入 number 或 nested list/tuple。"
    )


def broadcast_shapes(
    shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Numpy-style right-aligned broadcast of two shapes."""
    if shape_a == shape_b:
        return shape_a
    if len(shape_a) == 0:
        return shape_b
    if len(shape_b) == 0:
        return shape_a
    ra = shape_a[::-1]
    rb = shape_b[::-1]
    out: list[int] = []
    for da, db in zip(ra, rb):
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError(
                f"what: 无法广播 shape {shape_a} 与 {shape_b}（维度 {da} vs {db}）。\n"
                f"where: core.tensor.broadcast_shapes"
            )
    longer = ra if len(ra) >= len(rb) else rb
    out.extend(longer[len(out):])
    return tuple(out[::-1])


def _index_nested(data: Any, idx: Tuple[int, ...]) -> Any:
    for i in idx:
        data = data[i]
    return data


def _aligned_index(
    idx: Tuple[int, ...],
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    offset = len(out_shape) - len(in_shape)
    in_idx: list[int] = []
    for d, i in enumerate(idx):
        in_d = d - offset
        if in_d < 0:
            continue
        in_idx.append(0 if in_shape[in_d] == 1 else i)
    return tuple(in_idx)


def _build_nested(shape: Tuple[int, ...], fill: Any = None) -> Any:
    if shape == ():
        return fill
    return [_build_nested(shape[1:], fill) for _ in range(shape[0])]


def _set_nested(container: Any, idx: Tuple[int, ...], value: Any) -> None:
    for i in idx[:-1]:
        container = container[i]
    container[idx[-1]] = value


def broadcast_op(
    a_data: Any,
    a_shape: Tuple[int, ...],
    b_data: Any,
    b_shape: Tuple[int, ...],
    op: Callable[[Number, Number], Number],
) -> Any:
    """Element-wise broadcast op over number / nested list payloads."""
    out_shape = broadcast_shapes(a_shape, b_shape)
    if out_shape == ():
        return op(a_data, b_data)
    result = _build_nested(out_shape)
    for idx in itertools.product(*(range(s) for s in out_shape)):
        av = _index_nested(a_data, _aligned_index(idx, a_shape, out_shape))
        bv = _index_nested(b_data, _aligned_index(idx, b_shape, out_shape))
        _set_nested(result, idx, op(av, bv))
    return result


class TensorValue:
    """Portable tensor value flowing through the graph.

    Fields (DATA_FORMAT.md): data / shape / dtype / device / metadata.
    """

    def __init__(
        self,
        data: Any,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
    ) -> None:
        self.data = data
        self._kind = kind
        if kind in ("text", "json"):
            # Non-tensor payloads (string / parsed JSON) have no numeric shape.
            self.shape: Tuple[int, ...] = tuple(shape) if shape is not None else ()
        else:
            self.shape = tuple(shape) if shape is not None else infer_shape(data)
        self.dtype = dtype
        self.device = device
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

    @classmethod
    def from_value(
        cls,
        value: Any,
        dtype: str = "float32",
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
    ) -> "TensorValue":
        return cls(value, dtype=dtype, device=device, metadata=metadata, kind=kind)

    @property
    def data_kind(self) -> str:
        if self._kind:
            return self._kind
        return "scalar" if self.shape == () else "tensor"

    def item(self) -> Number:
        if self.shape != ():
            raise ValueError(
                f"what: 只能对标量取 item，当前 shape={self.shape}。\n"
                f"where: core.tensor.TensorValue.item"
            )
        return self.data  # type: ignore[return-value]

    def to_list(self) -> Any:
        return copy.deepcopy(self.data)

    def summary(self) -> str:
        """Short human-readable summary for UI previews."""
        if self._kind == "text":
            s = str(self.data)
            return f"text: {s[:120]}{'…' if len(s) > 120 else ''}"
        if self._kind == "json":
            import json
            s = json.dumps(self.data, ensure_ascii=False)
            return f"json: {s[:120]}{'…' if len(s) > 120 else ''}"
        if self.shape == ():
            return f"scalar {self.dtype} = {self.data}"
        if len(self.shape) == 1 and self.shape[0] <= 8:
            return f"{self.shape} {self.dtype} = {self.data}"
        return f"shape={self.shape} dtype={self.dtype} device={self.device}"

    def __repr__(self) -> str:
        return (
            f"TensorValue(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, data={self.data!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorValue):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.device == other.device
            and self.data == other.data
        )
