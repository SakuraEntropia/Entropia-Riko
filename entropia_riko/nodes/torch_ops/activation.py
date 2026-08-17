"""Activation nodes (torch-backed)."""

from __future__ import annotations

from typing import Any, Dict

from ...backend.converter import from_torch, to_torch
from ...backend.device import resolve_device
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


@register("relu")
class ReluNode(BaseNode):
    """ReLU activation (torch.relu)."""

    type_name = "relu"
    label = "ReLU"
    category = "Neural"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("output", data_kind="tensor")]
    parameters = [Parameter("device", kind="scalar", default="cpu")]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        dev = resolve_device(params["device"])
        x = to_torch(inputs["x"], dev)
        y = torch.relu(x)
        return {"output": from_torch(y, metadata={"backend": "torch"})}
