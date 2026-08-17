"""Constant node: data source producing a TensorValue."""

from __future__ import annotations

from typing import Any, Dict

from ...core.tensor import TensorValue
from ...runtime.registry import register
from ..base import BaseNode, NodeOutput, Parameter


@register("constant")
class ConstantNode(BaseNode):
    type_name = "constant"
    label = "Constant"
    category = "Inputs"
    inputs = []
    outputs = [NodeOutput("value", data_kind="tensor")]
    parameters = [Parameter("value", kind="any", required=True)]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"value": TensorValue.from_value(params["value"])}
