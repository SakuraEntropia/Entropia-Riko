"""Extra torch nodes: tensor creation, conv variants, device control, common ops.

Requires torch at import time (skipped if unavailable).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register
from ...backend.converter import to_torch, from_torch
from ...backend.device import resolve_device

_DTYPES = {
    "float32": torch.float32, "float64": torch.float64, "int32": torch.int32,
    "int64": torch.int64, "float16": torch.float16, "bool": torch.bool,
}


def _dtype(params: Dict[str, Any]):
    return _DTYPES.get(params.get("dtype", "float32"), torch.float32)


# ---------------------------------------------------------- tensor creation
def _creator(_t, _l, _fn, _c="Creation"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _c
        inputs = []
        outputs = [NodeOutput("result")]
        parameters = [
            Parameter("shape", kind="any", required=True),
            Parameter("dtype", default="float32"),
            Parameter("device", default="cpu"),
        ]

        def execute(self, inputs, params, context):
            dev = resolve_device(params["device"])
            shape = tuple(params["shape"])
            dt = _dtype(params)
            return {"result": from_torch(_fn(shape, dtype=dt, device=dev), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


_creator("zeros", "Zeros", lambda s, **kw: torch.zeros(s, **kw))
_creator("ones", "Ones", lambda s, **kw: torch.ones(s, **kw))
_creator("rand", "Rand", lambda s, **kw: torch.rand(s, **kw))
_creator("randn", "Randn", lambda s, **kw: torch.randn(s, **kw))


@register("eye")
class EyeNode(BaseNode):
    type_name = "eye"
    label = "Eye"
    category = "Creation"
    inputs = []
    outputs = [NodeOutput("result")]
    parameters = [Parameter("n", required=True, dtype="int"), Parameter("dtype", default="float32"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        return {"result": from_torch(torch.eye(int(params["n"]), dtype=_dtype(params), device=dev), metadata={"backend": "torch"})}


@register("arange")
class ArangeNode(BaseNode):
    type_name = "arange"
    label = "Arange"
    category = "Creation"
    inputs = []
    outputs = [NodeOutput("result")]
    parameters = [Parameter("start", default=0, dtype="int"), Parameter("end", required=True, dtype="int"), Parameter("step", default=1, dtype="int"), Parameter("dtype", default="int64"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        y = torch.arange(int(params["start"]), int(params["end"]), int(params["step"]), dtype=_dtype(params), device=dev)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("linspace")
class LinspaceNode(BaseNode):
    type_name = "linspace"
    label = "Linspace"
    category = "Creation"
    inputs = []
    outputs = [NodeOutput("result")]
    parameters = [Parameter("start", required=True, dtype="float"), Parameter("end", required=True, dtype="float"), Parameter("steps", required=True, dtype="int"), Parameter("dtype", default="float32"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        y = torch.linspace(float(params["start"]), float(params["end"]), int(params["steps"]), dtype=_dtype(params), device=dev)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("full")
class FullNode(BaseNode):
    type_name = "full"
    label = "Full"
    category = "Creation"
    inputs = []
    outputs = [NodeOutput("result")]
    parameters = [Parameter("shape", kind="any", required=True), Parameter("fill_value", required=True, dtype="float"), Parameter("dtype", default="float32"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        y = torch.full(tuple(params["shape"]), float(params["fill_value"]), dtype=_dtype(params), device=dev)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


# ---------------------------------------------------------- conv variants
@register("conv1d")
class Conv1dNode(BaseNode):
    type_name = "conv1d"
    label = "Conv1D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [
        Parameter("in_channels", required=True, dtype="int"),
        Parameter("out_channels", required=True, dtype="int"),
        Parameter("kernel_size", required=True, dtype="int"),
        Parameter("stride", default=1, dtype="int"),
        Parameter("padding", default=0, dtype="int"),
        Parameter("device", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        d = resolve_device(params["device"])
        layer = nn.Conv1d(int(params["in_channels"]), int(params["out_channels"]), int(params["kernel_size"]), int(params["stride"]), int(params["padding"])).to(d).eval()
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("conv_transpose2d")
class ConvTranspose2dNode(BaseNode):
    type_name = "conv_transpose2d"
    label = "ConvTranspose2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [
        Parameter("in_channels", required=True, dtype="int"),
        Parameter("out_channels", required=True, dtype="int"),
        Parameter("kernel_size", required=True, dtype="int"),
        Parameter("stride", default=1, dtype="int"),
        Parameter("padding", default=0, dtype="int"),
        Parameter("device", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        d = resolve_device(params["device"])
        layer = nn.ConvTranspose2d(int(params["in_channels"]), int(params["out_channels"]), int(params["kernel_size"]), int(params["stride"]), int(params["padding"])).to(d).eval()
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


# ---------------------------------------------------------- device control
@register("to_device")
class ToDeviceNode(BaseNode):
    """Explicitly move a tensor to a device (GPU/CPU control)."""

    type_name = "to_device"
    label = "To Device"
    category = "Device"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("device", required=True)]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        x = to_torch(inputs["x"], dev)
        return {"result": from_torch(x, metadata={"backend": "torch", "device": str(dev)})}


# ---------------------------------------------------------- common ops
@register("clamp")
class ClampNode(BaseNode):
    type_name = "clamp"
    label = "Clamp"
    category = "Tensor"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("min", kind="any", default=None), Parameter("max", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        kw = {}
        if params["min"] is not None:
            kw["min"] = float(params["min"])
        if params["max"] is not None:
            kw["max"] = float(params["max"])
        return {"result": from_torch(x.clamp(**kw), metadata={"backend": "torch"})}


@register("topk")
class TopkNode(BaseNode):
    type_name = "topk"
    label = "TopK"
    category = "Reduce"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("k", required=True, dtype="int"), Parameter("dim", default=-1, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        y = torch.topk(x, int(params["k"]), dim=int(params["dim"])).values
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("sort")
class SortNode(BaseNode):
    type_name = "sort"
    label = "Sort"
    category = "Reduce"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", default=-1, dtype="int"), Parameter("descending", default=False), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        y = torch.sort(x, dim=int(params["dim"]), descending=bool(params["descending"])).values
        return {"result": from_torch(y, metadata={"backend": "torch"})}


def _dev(params):
    return resolve_device(params.get("device", "cpu"))


def _simple_unary(_t, _l, _fn, _c="Tensor"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _c
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            x = to_torch(inputs["x"], _dev(params))
            return {"result": from_torch(_fn(x), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


_simple_unary("contiguous", "Contiguous", lambda t: t.contiguous())
_simple_unary("clone", "Clone", lambda t: t.clone())
_simple_unary("detach", "Detach", lambda t: t.detach())


@register("masked_fill")
class MaskedFillNode(BaseNode):
    type_name = "masked_fill"
    label = "Masked Fill"
    category = "Tensor"
    inputs = [NodeInput("x", required=True), NodeInput("mask", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("value", required=True, dtype="float"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        x = to_torch(inputs["x"], d)
        mask = to_torch(inputs["mask"], d).bool()
        return {"result": from_torch(x.masked_fill(mask, float(params["value"])), metadata={"backend": "torch"})}


@register("where")
class WhereNode(BaseNode):
    type_name = "where"
    label = "Where"
    category = "Tensor"
    inputs = [NodeInput("condition", required=True), NodeInput("x", required=True), NodeInput("y", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        cond = to_torch(inputs["condition"], d).bool()
        x = to_torch(inputs["x"], d)
        y = to_torch(inputs["y"], d)
        return {"result": from_torch(torch.where(cond, x, y), metadata={"backend": "torch"})}


@register("flip")
class FlipNode(BaseNode):
    type_name = "flip"
    label = "Flip"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dims", kind="any", required=True), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        dims = params["dims"] if isinstance(params["dims"], list) else [int(params["dims"])]
        return {"result": from_torch(x.flip(dims), metadata={"backend": "torch"})}


@register("expand_as")
class ExpandAsNode(BaseNode):
    type_name = "expand_as"
    label = "Expand As"
    category = "Shape"
    inputs = [NodeInput("x", required=True), NodeInput("other", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        x = to_torch(inputs["x"], d)
        other = to_torch(inputs["other"], d)
        return {"result": from_torch(x.expand_as(other), metadata={"backend": "torch"})}


@register("repeat_interleave")
class RepeatInterleaveNode(BaseNode):
    type_name = "repeat_interleave"
    label = "Repeat Interleave"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("repeats", required=True, dtype="int"), Parameter("dim", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        dim = int(params["dim"]) if params["dim"] is not None else None
        return {"result": from_torch(x.repeat_interleave(int(params["repeats"]), dim=dim), metadata={"backend": "torch"})}
