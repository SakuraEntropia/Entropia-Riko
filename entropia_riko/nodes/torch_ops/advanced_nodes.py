"""Advanced torch coverage: attention, normalization variants, extra ops/losses.

Broadens the node library so the editor can express industrial models —
ViT/AR predictors (multi-head attention, SDPA, AdaLN), normalization
(BatchNorm2d / GroupNorm / RMSNorm), interpolation, einsum, positional
encodings, and a wide set of math / shape / loss ops.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backend.converter import from_torch, to_torch
from ...backend.device import resolve_device
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


def _dev(params: Dict[str, Any]):
    return resolve_device(params.get("device", "cpu"))


# ------------------------------------------------------------------ factories
def _unary(_t, _l, _fn, _cat="Tensor"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            x = to_torch(inputs["x"], _dev(params))
            return {"result": from_torch(_fn(x), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


def _unary_p(_t, _l, _fn, _params, _cat="Tensor"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = _params + [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            x = to_torch(inputs["x"], _dev(params))
            return {"result": from_torch(_fn(x, params), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


def _binary(_t, _l, _fn, _cat="Tensor", _in=("left", "right")):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = [NodeInput(_in[0], required=True), NodeInput(_in[1], required=True)]
        outputs = [NodeOutput("result")]
        parameters = [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            d = _dev(params)
            a = to_torch(inputs[_in[0]], d)
            b = to_torch(inputs[_in[1]], d)
            return {"result": from_torch(_fn(a, b), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


def _binary_p(_t, _l, _fn, _params, _cat="Tensor"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = [NodeInput("left", required=True), NodeInput("right", required=True)]
        outputs = [NodeOutput("result")]
        parameters = _params + [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            d = _dev(params)
            a = to_torch(inputs["left"], d)
            b = to_torch(inputs["right"], d)
            return {"result": from_torch(_fn(a, b, params), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


def _reduce(_t, _l, _fn, _cat="Reduce"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
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
            if dim is None:
                y = _fn(x)
            else:
                kw: Dict[str, Any] = {"dim": dim if isinstance(dim, list) else int(dim), "keepdim": bool(params["keepdim"])}
                y = _fn(x, **kw)
            return {"result": from_torch(y, metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


def _creator(_t, _l, _fn, _params, _cat="Creation"):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = _cat
        inputs = []
        outputs = [NodeOutput("result")]
        parameters = _params + [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            dev = resolve_device(params["device"])
            return {"result": from_torch(_fn(params, dev), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


# ------------------------------------------------------- unary math / act
_unary("log2", "Log2", torch.log2)
_unary("log10", "Log10", torch.log10)
_unary("log1p", "Log1p", torch.log1p)
_unary("expm1", "Expm1", torch.expm1)
_unary("erf", "Erf", torch.erf)
_unary("erfc", "Erfc", torch.erfc)
_unary("softsign", "Softsign", F.softsign, "Neural")
_unary("tanhshrink", "Tanhshrink", F.tanhshrink, "Neural")
_unary("hardsigmoid", "Hardsigmoid", F.hardsigmoid, "Neural")
_unary("log_sigmoid", "LogSigmoid", F.logsigmoid, "Neural")

_unary_p("leaky_relu", "LeakyReLU", lambda x, p: F.leaky_relu(x, negative_slope=float(p.get("negative_slope", 0.01))),
         [Parameter("negative_slope", default=0.01, dtype="float")], "Neural")
_unary_p("hardtanh", "HardTanh", lambda x, p: F.hardtanh(x, min_val=float(p.get("min_val", -1.0)), max_val=float(p.get("max_val", 1.0))),
         [Parameter("min_val", default=-1.0, dtype="float"), Parameter("max_val", default=1.0, dtype="float")], "Neural")
_unary_p("glu", "GLU", lambda x, p: F.glu(x, dim=int(p.get("dim", -1))),
         [Parameter("dim", default=-1, dtype="int")], "Neural")


# ------------------------------------------------------- reduce extras
_reduce("logsumexp", "LogSumExp", torch.logsumexp)
_reduce("amax", "AMax", torch.amax)
_reduce("amin", "AMin", torch.amin)
_reduce("median", "Median", lambda x, **kw: torch.median(x, **kw).values if "dim" in kw else torch.median(x))


# ------------------------------------------------------- binary extras
_binary("bmm", "BMM", torch.bmm)
_binary("dot", "Dot", torch.dot)
_binary("outer", "Outer", torch.outer)
_binary("xlogy", "XLogY", torch.xlogy)
_binary_p("cross", "Cross", lambda a, b, p: torch.cross(a, b, dim=int(p.get("dim", -1))),
          [Parameter("dim", default=-1, dtype="int")])


@register("addmm")
class AddMmNode(BaseNode):
    type_name = "addmm"
    label = "AddMM"
    category = "Tensor"
    inputs = [NodeInput("input", required=True), NodeInput("mat1", required=True), NodeInput("mat2", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("beta", default=1.0, dtype="float"), Parameter("alpha", default=1.0, dtype="float"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        return {"result": from_torch(
            torch.addmm(to_torch(inputs["input"], d), to_torch(inputs["mat1"], d), to_torch(inputs["mat2"], d),
                        beta=float(params["beta"]), alpha=float(params["alpha"])),
            metadata={"backend": "torch"})}


# ------------------------------------------------------- creation extras
def _randint(params, dev):
    return torch.randint(int(params["low"]), int(params["high"]), tuple(params["size"]), device=dev)

def _randperm(params, dev):
    return torch.randperm(int(params["n"]), device=dev)

def _empty(params, dev):
    return torch.empty(tuple(params["shape"]), device=dev)

_creator("randint", "RandInt", _randint,
         [Parameter("low", default=0, dtype="int"), Parameter("high", required=True, dtype="int"), Parameter("size", kind="any", required=True)])
_creator("randperm", "RandPerm", _randperm, [Parameter("n", required=True, dtype="int")])
_creator("empty", "Empty", _empty, [Parameter("shape", kind="any", required=True)])


# like-creation (takes an input tensor)
def _like(_t, _l, _fn):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = "Creation"
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            x = to_torch(inputs["x"], _dev(params))
            return {"result": from_torch(_fn(x), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


_like("zeros_like", "ZerosLike", torch.zeros_like)
_like("ones_like", "OnesLike", torch.ones_like)
_like("randn_like", "RandnLike", torch.randn_like)
_like("rand_like", "RandLike", torch.rand_like)


# ------------------------------------------------------- shape / layout extras
def _shape_op(_t, _l, _fn, _params=None):
    @register(_t)
    class N(BaseNode):
        type_name = _t
        label = _l
        category = "Shape"
        inputs = [NodeInput("x", required=True)]
        outputs = [NodeOutput("result")]
        parameters = (_params or []) + [Parameter("device", default="cpu")]

        def execute(self, inputs, params, context):
            x = to_torch(inputs["x"], _dev(params))
            return {"result": from_torch(_fn(x, params), metadata={"backend": "torch"})}

    N.__name__ = _l.replace(" ", "") + "Node"
    return N


_shape_op("view", "View", lambda x, p: x.view(*tuple(p["shape"])),
          [Parameter("shape", kind="any", required=True)])
_shape_op("swapaxes", "SwapAxes", lambda x, p: x.swapaxes(int(p["dim0"]), int(p["dim1"])),
          [Parameter("dim0", required=True, dtype="int"), Parameter("dim1", required=True, dtype="int")])
_shape_op("movedim", "MoveDim", lambda x, p: x.movedim(int(p["source"]), int(p["destination"])),
          [Parameter("source", required=True, dtype="int"), Parameter("destination", required=True, dtype="int")])
_shape_op("expand", "Expand", lambda x, p: x.expand(*tuple(p["shape"])),
          [Parameter("shape", kind="any", required=True)])
_shape_op("broadcast_to", "BroadcastTo", lambda x, p: x.broadcast_to(tuple(p["shape"])),
          [Parameter("shape", kind="any", required=True)])
_shape_op("tile", "Tile", lambda x, p: x.tile(*tuple(p["dims"])),
          [Parameter("dims", kind="any", required=True)])
_shape_op("repeat", "Repeat", lambda x, p: x.repeat(*tuple(p["dims"])),
          [Parameter("dims", kind="any", required=True)])
_shape_op("tril", "Tril", lambda x, p: x.tril(int(p.get("diagonal", 0))),
          [Parameter("diagonal", default=0, dtype="int")])
_shape_op("triu", "Triu", lambda x, p: x.triu(int(p.get("diagonal", 0))),
          [Parameter("diagonal", default=0, dtype="int")])
_shape_op("diagonal", "Diagonal", lambda x, p: x.diagonal(int(p.get("offset", 0)), int(p.get("dim1", 0)), int(p.get("dim2", 1))),
          [Parameter("offset", default=0, dtype="int"), Parameter("dim1", default=0, dtype="int"), Parameter("dim2", default=1, dtype="int")])
_shape_op("narrow", "Narrow", lambda x, p: x.narrow(int(p["dim"]), int(p["start"]), int(p["length"])),
          [Parameter("dim", required=True, dtype="int"), Parameter("start", required=True, dtype="int"), Parameter("length", required=True, dtype="int")])
_shape_op("roll", "Roll", lambda x, p: x.roll(tuple(p["shifts"]) if isinstance(p["shifts"], list) else int(p["shifts"]), p["dims"] if p["dims"] is not None else None),
          [Parameter("shifts", kind="any", required=True), Parameter("dims", kind="any", default=None)])
_shape_op("interpolate", "Interpolate",
          lambda x, p: F.interpolate(x, size=tuple(p["size"]) if p.get("size") is not None else None,
                                     scale_factor=p.get("scale_factor"), mode=p.get("mode", "nearest")),
          [Parameter("size", kind="any", default=None), Parameter("scale_factor", kind="any", default=None),
           Parameter("mode", default="nearest")])


@register("index_select")
class IndexSelectNode(BaseNode):
    type_name = "index_select"
    label = "IndexSelect"
    category = "Shape"
    inputs = [NodeInput("x", required=True), NodeInput("index", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        return {"result": from_torch(
            to_torch(inputs["x"], d).index_select(int(params["dim"]), to_torch(inputs["index"], d).long()),
            metadata={"backend": "torch"})}


@register("gather")
class GatherNode(BaseNode):
    type_name = "gather"
    label = "Gather"
    category = "Shape"
    inputs = [NodeInput("x", required=True), NodeInput("index", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        return {"result": from_torch(
            to_torch(inputs["x"], d).gather(int(params["dim"]), to_torch(inputs["index"], d).long()),
            metadata={"backend": "torch"})}


@register("einsum")
class EinsumNode(BaseNode):
    type_name = "einsum"
    label = "Einsum"
    category = "Tensor"
    inputs = [NodeInput("a", required=True), NodeInput("b", required=True)]
    outputs = [NodeOutput("result")]
    parameters = [Parameter("equation", kind="scalar", required=True), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        return {"result": from_torch(
            torch.einsum(params["equation"], to_torch(inputs["a"], d), to_torch(inputs["b"], d)),
            metadata={"backend": "torch"})}


# ------------------------------------------------------- extra losses
_binary("l1_loss", "L1 Loss", F.l1_loss, "Loss", _in=("pred", "target"))
_binary("smooth_l1_loss", "SmoothL1 Loss", F.smooth_l1_loss, "Loss", _in=("pred", "target"))
_binary("binary_cross_entropy", "BinaryCE Loss", F.binary_cross_entropy, "Loss", _in=("pred", "target"))
_binary("kl_div", "KLDiv Loss", F.kl_div, "Loss", _in=("pred", "target"))
_binary("nll_loss", "NLL Loss", lambda p, t: F.nll_loss(p, t.long()), "Loss", _in=("pred", "target"))
_binary("hinge_embedding_loss", "HingeEmbedding Loss", F.hinge_embedding_loss, "Loss", _in=("pred", "target"))
_binary("cosine_embedding_loss", "CosineEmbedding Loss", F.cosine_embedding_loss, "Loss", _in=("pred", "target"))


# ------------------------------------------------------- normalization layers
@register("batchnorm2d")
class BatchNorm2dNode(BaseNode):
    type_name = "batchnorm2d"
    label = "BatchNorm2D"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("num_features", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        layer = nn.BatchNorm2d(int(params["num_features"])).to(d).eval()
        with torch.no_grad():
            return {"output": from_torch(layer(to_torch(inputs["x"], d)), metadata={"backend": "torch"})}


@register("groupnorm")
class GroupNormNode(BaseNode):
    type_name = "groupnorm"
    label = "GroupNorm"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("num_groups", required=True, dtype="int"), Parameter("num_channels", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        layer = nn.GroupNorm(int(params["num_groups"]), int(params["num_channels"])).to(d).eval()
        with torch.no_grad():
            return {"output": from_torch(layer(to_torch(inputs["x"], d)), metadata={"backend": "torch"})}


@register("rmsnorm")
class RMSNormNode(BaseNode):
    type_name = "rmsnorm"
    label = "RMSNorm"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("dim", required=True, dtype="int"), Parameter("eps", default=1e-6, dtype="float"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        x = to_torch(inputs["x"], d)
        dim = int(params["dim"])
        eps = float(params["eps"])
        w = torch.ones(x.size(dim), device=d)
        rms = x.pow(2).mean(dim=dim, keepdim=True).add(eps).sqrt()
        y = x / rms * w
        return {"output": from_torch(y, metadata={"backend": "torch"})}


# ------------------------------------------------------- attention
@register("multihead_attention")
class MultiheadAttentionNode(BaseNode):
    type_name = "multihead_attention"
    label = "Multihead Attention"
    category = "Neural"
    inputs = [NodeInput("x", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [
        Parameter("embed_dim", required=True, dtype="int"),
        Parameter("num_heads", required=True, dtype="int"),
        Parameter("device", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        d = _dev(params)
        emb = int(params["embed_dim"])
        heads = int(params["num_heads"])
        attn = nn.MultiheadAttention(emb, heads, batch_first=True).to(d).eval()
        x = to_torch(inputs["x"], d)
        with torch.no_grad():
            out, _ = attn(x, x, x, need_weights=False)
        return {"output": from_torch(out, metadata={"backend": "torch"})}


@register("sdpa")
class SdpaNode(BaseNode):
    type_name = "sdpa"
    label = "Scaled Dot-Product Attention"
    category = "Neural"
    inputs = [NodeInput("q", required=True), NodeInput("k", required=True), NodeInput("v", required=True)]
    outputs = [NodeOutput("output")]
    parameters = [Parameter("is_causal", default=False), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        d = _dev(params)
        q = to_torch(inputs["q"], d)
        k = to_torch(inputs["k"], d)
        v = to_torch(inputs["v"], d)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=bool(params["is_causal"]))
        return {"output": from_torch(out, metadata={"backend": "torch"})}


# ------------------------------------------------------- positional encoding
def _sincos_1d(n, dim):
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


@register("positional_encoding")
class PositionalEncodingNode(BaseNode):
    type_name = "positional_encoding"
    label = "Positional Encoding"
    category = "Creation"
    inputs = []
    outputs = [NodeOutput("result")]
    parameters = [Parameter("n", required=True, dtype="int"), Parameter("dim", required=True, dtype="int"), Parameter("device", default="cpu")]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        return {"result": from_torch(_sincos_1d(int(params["n"]), int(params["dim"])).to(dev), metadata={"backend": "torch"})}
