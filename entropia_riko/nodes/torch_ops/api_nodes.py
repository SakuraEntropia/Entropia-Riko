"""Broad PyTorch API node coverage (Stage 6).

Registers ~45 common torch operations as nodes via factories. Requires
torch at import time; if torch is unavailable, this module is skipped.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register
from ...backend.converter import to_torch, from_torch
from ...backend.device import resolve_device


def _dev(params: Dict[str, Any]):
    return resolve_device(params.get("device", "cpu"))


# ------------------------------------------------------------------ unary
def _unary(_t, _l, _fn, _c="Tensor"):
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


for _name, _lbl, _fn in [
    ("abs", "Abs", torch.abs),
    ("exp", "Exp", torch.exp),
    ("log", "Log", torch.log),
    ("sqrt", "Sqrt", torch.sqrt),
    ("neg", "Negate", torch.neg),
    ("sign", "Sign", torch.sign),
    ("reciprocal", "Reciprocal", torch.reciprocal),
    ("floor", "Floor", torch.floor),
    ("ceil", "Ceil", torch.ceil),
    ("round", "Round", torch.round),
    ("square", "Square", lambda t: t * t),
    ("softplus", "Softplus", F.softplus),
    ("relu6", "ReLU6", F.relu6),
    ("silu", "SiLU", F.silu),
    ("selu", "SELU", F.selu),
    ("elu", "ELU", F.elu),
    ("gelu", "GELU", F.gelu),
    ("sigmoid", "Sigmoid", torch.sigmoid),
    ("tanh", "Tanh", torch.tanh),
    ("mish", "Mish", F.mish),
    ("hardswish", "Hardswish", F.hardswish),
    ("cos", "Cos", torch.cos),
    ("sin", "Sin", torch.sin),
]:
    _unary(_name, _lbl, _fn)


# ----------------------------------------------------------------- binary
def _binary(_t, _l, _fn, _c="Math"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _c
        inputs = [NodeInput("left", required=True), NodeInput("right", required=True)]
        outputs = [NodeOutput("result")]
        parameters = [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            d = _dev(params)
            a = to_torch(inputs["left"], d)
            b = to_torch(inputs["right"], d)
            return {"result": from_torch(_fn(a, b), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


for _name, _lbl, _fn in [
    ("sub", "Subtract", torch.sub),
    ("div", "Divide", torch.div),
    ("pow", "Power", torch.pow),
    ("maximum", "Maximum", torch.maximum),
    ("minimum", "Minimum", torch.minimum),
    ("fmod", "Fmod", torch.fmod),
    ("remainder", "Remainder", torch.remainder),
    ("matmul", "MatMul", torch.matmul),
    ("mm", "MM", torch.mm),
    ("atan2", "Atan2", torch.atan2),
]:
    _binary(_name, _lbl, _fn)


# ------------------------------------------------------------------ reduce
def _reduce(_t, _l, _fn, _c="Reduce"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _c
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = [
            Parameter("dim", kind="any", default=None),
            Parameter("keepdim", default=False),
            Parameter("device", default="cpu"),
        ]

        def execute(self, inputs, params, context):
            x = to_torch(inputs["x"], _dev(params))
            dim = params["dim"]
            if dim is not None:
                kw: Dict[str, Any] = {"keepdim": bool(params["keepdim"])}
                kw["dim"] = dim if isinstance(dim, list) else int(dim)
                y = _fn(x, **kw)
            else:
                y = _fn(x)
            return {"result": from_torch(y, metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


for _name, _lbl, _fn in [
    ("sum", "Sum", torch.sum),
    ("mean", "Mean", torch.mean),
    ("prod", "Prod", torch.prod),
    ("std", "Std", torch.std),
    ("var", "Var", torch.var),
    ("norm", "Norm", lambda x, **kw: torch.norm(x.float(), **kw)),
    ("max_reduce", "Max Reduce", lambda x, **kw: torch.max(x, **kw).values if "dim" in kw else torch.max(x)),
    ("min_reduce", "Min Reduce", lambda x, **kw: torch.min(x, **kw).values if "dim" in kw else torch.min(x)),
]:
    _reduce(_name, _lbl, _fn)


@register("argmax")
class ArgMaxNode(BaseNode):
    type_name = "argmax"
    label = "ArgMax"
    category = "Reduce"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        dim = params["dim"]
        y = torch.argmax(x, dim=int(dim)) if dim is not None else torch.argmax(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("argmin")
class ArgMinNode(BaseNode):
    type_name = "argmin"
    label = "ArgMin"
    category = "Reduce"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        dim = params["dim"]
        y = torch.argmin(x, dim=int(dim)) if dim is not None else torch.argmin(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("cumsum")
class CumSumNode(BaseNode):
    type_name = "cumsum"
    label = "CumSum"
    category = "Reduce"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        return {"result": from_torch(torch.cumsum(x, dim=int(params["dim"])), metadata={"backend": "torch"})}


# ------------------------------------------------------------- shape ops
@register("reshape")
class ReshapeNode(BaseNode):
    type_name = "reshape"
    label = "Reshape"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("shape", kind="any", required=True), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        shape = tuple(params["shape"])
        return {"result": from_torch(x.reshape(*shape), metadata={"backend": "torch"})}


@register("transpose")
class TransposeNode(BaseNode):
    type_name = "transpose"
    label = "Transpose"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim0", required=True, dtype="int"), Parameter("dim1", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        return {"result": from_torch(x.transpose(int(params["dim0"]), int(params["dim1"])), metadata={"backend": "torch"})}


@register("permute")
class PermuteNode(BaseNode):
    type_name = "permute"
    label = "Permute"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dims", kind="any", required=True), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        dims = tuple(params["dims"])
        return {"result": from_torch(x.permute(*dims), metadata={"backend": "torch"})}


@register("flatten")
class FlattenNode(BaseNode):
    type_name = "flatten"
    label = "Flatten"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("start_dim", default=0, dtype="int"), Parameter("end_dim", default=-1, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        return {"result": from_torch(x.flatten(int(params["start_dim"]), int(params["end_dim"])), metadata={"backend": "torch"})}


@register("squeeze")
class SqueezeNode(BaseNode):
    type_name = "squeeze"
    label = "Squeeze"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        dim = params["dim"]
        y = x.squeeze(int(dim)) if dim is not None else x.squeeze()
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("unsqueeze")
class UnsqueezeNode(BaseNode):
    type_name = "unsqueeze"
    label = "Unsqueeze"
    category = "Shape"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        return {"result": from_torch(x.unsqueeze(int(params["dim"])), metadata={"backend": "torch"})}


@register("concat")
class ConcatNode(BaseNode):
    type_name = "concat"
    label = "Concat"
    category = "Shape"
    inputs = [NodeInput("left", required=True), NodeInput("right", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        a = to_torch(inputs["left"], d)
        b = to_torch(inputs["right"], d)
        return {"result": from_torch(torch.cat([a, b], dim=int(params["dim"])), metadata={"backend": "torch"})}


@register("stack")
class StackNode(BaseNode):
    type_name = "stack"
    label = "Stack"
    category = "Shape"
    inputs = [NodeInput("left", required=True), NodeInput("right", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        a = to_torch(inputs["left"], d)
        b = to_torch(inputs["right"], d)
        return {"result": from_torch(torch.stack([a, b], dim=int(params["dim"])), metadata={"backend": "torch"})}


# --------------------------------------------------------------- softmax
@register("softmax")
class SoftmaxNode(BaseNode):
    type_name = "softmax"
    label = "Softmax"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", default=-1, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        return {"result": from_torch(F.softmax(x, dim=int(params["dim"])), metadata={"backend": "torch"})}


@register("log_softmax")
class LogSoftmaxNode(BaseNode):
    type_name = "log_softmax"
    label = "LogSoftmax"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", default=-1, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        x = to_torch(inputs["x"], _dev(params))
        return {"result": from_torch(F.log_softmax(x, dim=int(params["dim"])), metadata={"backend": "torch"})}


# --------------------------------------------------------------- NN ops
@register("conv2d")
class Conv2dNode(BaseNode):
    type_name = "conv2d"
    label = "Conv2D"
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
        Parameter("weight", kind="any", default=None),
        Parameter("bias", kind="any", default=None),
    ]

    def execute(self, inputs, params, context):
        d = _dev(params)
        layer = nn.Conv2d(
            int(params["in_channels"]), int(params["out_channels"]),
            int(params["kernel_size"]), int(params["stride"]), int(params["padding"]),
        )
        if params["weight"] is not None:
            layer.weight.data = torch.tensor(params["weight"], dtype=torch.float32)
        if params["bias"] is not None:
            layer.bias.data = torch.tensor(params["bias"], dtype=torch.float32)
        layer = layer.to(d).eval()
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("maxpool2d")
class MaxPool2dNode(BaseNode):
    type_name = "maxpool2d"
    label = "MaxPool2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("kernel_size", required=True, dtype="int"), Parameter("stride", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        ks = int(params["kernel_size"])
        stride = int(params["stride"]) if params["stride"] is not None else ks
        layer = nn.MaxPool2d(ks, stride=stride)
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("avgpool2d")
class AvgPool2dNode(BaseNode):
    type_name = "avgpool2d"
    label = "AvgPool2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("kernel_size", required=True, dtype="int"), Parameter("stride", kind="any", default=None), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        ks = int(params["kernel_size"])
        stride = int(params["stride"]) if params["stride"] is not None else ks
        layer = nn.AvgPool2d(ks, stride=stride)
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("embedding")
class EmbeddingNode(BaseNode):
    type_name = "embedding"
    label = "Embedding"
    category = "Neural"
    inputs = [NodeInput("indices", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("num_embeddings", required=True, dtype="int"), Parameter("embedding_dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        layer = nn.Embedding(int(params["num_embeddings"]), int(params["embedding_dim"])).to(d).eval()
        idx = to_torch(inputs["indices"], d).long()
        with torch.no_grad():
            y = layer(idx)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("dropout")
class DropoutNode(BaseNode):
    type_name = "dropout"
    label = "Dropout"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("p", default=0.5, dtype="float"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        layer = nn.Dropout(float(params["p"])).to(d).eval()  # eval -> identity
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("batchnorm1d")
class BatchNorm1dNode(BaseNode):
    type_name = "batchnorm1d"
    label = "BatchNorm1D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("num_features", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        layer = nn.BatchNorm1d(int(params["num_features"])).to(d).eval()
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("layernorm")
class LayerNormNode(BaseNode):
    type_name = "layernorm"
    label = "LayerNorm"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("normalized_shape", kind="any", required=True), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        shape = tuple(params["normalized_shape"]) if isinstance(params["normalized_shape"], list) else int(params["normalized_shape"])
        layer = nn.LayerNorm(shape).to(d).eval()
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            y = layer(x)
        return {"result": from_torch(y, metadata={"backend": "torch"})}
