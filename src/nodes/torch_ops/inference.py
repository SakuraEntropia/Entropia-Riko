"""Inference node: run a loaded model on an input tensor (Stage 6)."""

from __future__ import annotations

from typing import Any, Dict

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register
from ...backend.converter import to_torch, from_torch
from ...backend.device import resolve_device


@register("inference")
class InferenceNode(BaseNode):
    """Run forward() on a model loaded by ``model_loader``."""

    type_name = "inference"
    label = "Inference"
    category = "Model"
    inputs = [
        NodeInput("model", data_kind="model", required=True),
        NodeInput("x", data_kind="tensor", required=True),
    ]
    outputs = [NodeOutput("output", data_kind="tensor")]
    parameters = [Parameter("device", kind="scalar", default="cpu")]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        model = inputs["model"]
        dev = resolve_device(params["device"])
        model = model.to(dev).eval()
        x = to_torch(inputs["x"], dev)
        with torch.no_grad():
            y = model(x)
        return {"output": from_torch(y, metadata={"backend": "torch", "node": "inference"})}
