"""Add node: element-wise addition with broadcasting."""

from __future__ import annotations

from typing import Any, Dict

from ..base import BaseNode, NodeInput, NodeOutput
from ...core.tensor import TensorValue, broadcast_op, broadcast_shapes
from ...runtime.registry import register


@register("add")
class AddNode(BaseNode):
    type_name = "add"
    label = "Add"
    category = "Math"
    inputs = [
        NodeInput("left", data_kind="tensor", required=True),
        NodeInput("right", data_kind="tensor", required=True),
    ]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = []

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        a: TensorValue = inputs["left"]
        b: TensorValue = inputs["right"]
        data = broadcast_op(a.data, a.shape, b.data, b.shape, lambda x, y: x + y)
        shape = broadcast_shapes(a.shape, b.shape)
        dtype = a.dtype if a.dtype == b.dtype else "float32"
        return {"result": TensorValue(data, shape=shape, dtype=dtype, device=a.device)}
