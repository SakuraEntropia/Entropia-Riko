"""Linear node: torch.nn.Linear inference (Stage 4 model node)."""

from __future__ import annotations

from typing import Any, Dict

from ...backend.converter import from_torch, to_torch
from ...backend.device import resolve_device
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


@register("linear")
class LinearNode(BaseNode):
    """A torch-backed fully-connected layer (nn.Linear)."""

    type_name = "linear"
    label = "Linear"
    category = "Neural"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("output", data_kind="tensor")]
    parameters = [
        Parameter("in_features", kind="scalar", required=True, dtype="int"),
        Parameter("out_features", kind="scalar", required=True, dtype="int"),
        Parameter("use_bias", kind="scalar", default=True),
        Parameter("device", kind="scalar", default="cpu"),
        Parameter("weight", kind="any", default=None),
        Parameter("bias", kind="any", default=None),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch
        import torch.nn as nn

        dev = resolve_device(params["device"])
        in_f = int(params["in_features"])
        out_f = int(params["out_features"])
        use_bias = bool(params["use_bias"])

        layer = nn.Linear(in_f, out_f, bias=use_bias)
        if params["weight"] is not None:
            w = torch.tensor(params["weight"], dtype=torch.float32)
            if tuple(w.shape) != (out_f, in_f):
                raise ValueError(
                    f"what: linear 权重 shape 应为 ({out_f}, {in_f})，实际 {tuple(w.shape)}。\n"
                    f"where: nodes.torch_ops.linear.execute"
                )
            layer.weight.data = w
        if use_bias and params["bias"] is not None:
            b = torch.tensor(params["bias"], dtype=torch.float32)
            if tuple(b.shape) != (out_f,):
                raise ValueError(
                    f"what: linear 偏置 shape 应为 ({out_f},)，实际 {tuple(b.shape)}。\n"
                    f"where: nodes.torch_ops.linear.execute"
                )
            layer.bias.data = b

        layer = layer.to(dev).eval()
        x = to_torch(inputs["x"], dev)
        with torch.no_grad():
            y = layer(x)
        return {"output": from_torch(y, metadata={"backend": "torch", "node": "linear"})}
