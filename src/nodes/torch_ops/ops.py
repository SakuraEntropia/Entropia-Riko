"""Torch-backed math nodes (Stage 3).

Use the backend converter (to_torch / from_torch) so computation runs on
torch with device support. torch is imported lazily in execute; if torch
is unavailable, execution raises a clear error (CROSS_PLATFORM.md).
"""
from __future__ import annotations

from typing import Any, Dict

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...core.tensor import TensorValue  # noqa: F401  part of contract surface
from ...runtime.registry import register
from ...backend.converter import to_torch, from_torch
from ...backend.device import resolve_device


@register("torch_add")
class TorchAddNode(BaseNode):
    type_name = "torch_add"
    label = "Torch Add"
    category = "Math"
    inputs = [
        NodeInput("left", data_kind="tensor", required=True),
        NodeInput("right", data_kind="tensor", required=True),
    ]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [Parameter("device", kind="scalar", default="cpu")]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        dev = resolve_device(params["device"])
        a = to_torch(inputs["left"], dev)
        b = to_torch(inputs["right"], dev)
        y = torch.add(a, b)
        return {"result": from_torch(y, metadata={"backend": "torch"})}


@register("torch_multiply")
class TorchMultiplyNode(BaseNode):
    type_name = "torch_multiply"
    label = "Torch Multiply"
    category = "Math"
    inputs = [
        NodeInput("left", data_kind="tensor", required=True),
        NodeInput("right", data_kind="tensor", required=True),
    ]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [Parameter("device", kind="scalar", default="cpu")]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        dev = resolve_device(params["device"])
        a = to_torch(inputs["left"], dev)
        b = to_torch(inputs["right"], dev)
        y = torch.mul(a, b)
        return {"result": from_torch(y, metadata={"backend": "torch"})}
