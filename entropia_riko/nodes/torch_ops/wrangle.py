"""Wrangle node: inline Python code to solve whatever built-in nodes can't.

Like Houdini's wrangle nodes, the ``code`` parameter is arbitrary Python
executed against the connected inputs. The code sees the input tensors as
``x`` / ``a`` / ``b`` (torch tensors, or ``None`` when unconnected) plus
``torch`` / ``F`` / ``nn`` in scope, and must assign ``result`` (a torch
tensor) — or ``y`` as a shorthand.
"""
from __future__ import annotations

from typing import Any, Dict

from ...backend.converter import from_torch, to_torch
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


@register("wrangle")
class WrangleNode(BaseNode):
    type_name = "wrangle"
    label = "Wrangle"
    category = "Utility"
    inputs = [
        NodeInput("x", data_kind="tensor", required=False, default=None),
        NodeInput("a", data_kind="tensor", required=False, default=None),
        NodeInput("b", data_kind="tensor", required=False, default=None),
    ]
    outputs = [NodeOutput("result", data_kind="tensor")]
    parameters = [
        Parameter("code", kind="scalar", required=True,
                  default="result = x  # write arbitrary PyTorch here"),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        ns: Dict[str, Any] = {
            "torch": torch,
            "F": torch.nn.functional,
            "nn": torch.nn,
        }
        for name in ("x", "a", "b"):
            ns[name] = to_torch(inputs[name]) if inputs.get(name) is not None else None

        code = str(params.get("code", ""))
        exec(code, ns)  # noqa: S102 - user-authored wrangle code (intended)

        result = ns.get("result", ns.get("y"))
        if result is None:
            raise ValueError(
                "what: wrangle 代码未产出 result。\n"
                "where: nodes.torch_ops.wrangle.execute\n"
                "how_to_fix: 在 code 中赋值 result = ...（torch tensor）。"
            )
        return {"result": from_torch(result, metadata={"backend": "torch", "node": "wrangle"})}
