"""Base node contract (NODE_SYSTEM.md, API.md).

Each node provides: type_name, label, category, inputs, outputs,
parameters, and execute(inputs, params, context).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class NodeInput:
    """A node input port."""

    def __init__(
        self,
        name: str,
        data_kind: str = "tensor",
        required: bool = True,
        default: Any = None,
    ) -> None:
        self.name = name
        self.data_kind = data_kind
        self.required = required
        self.default = default


class NodeOutput:
    """A node output port."""

    def __init__(self, name: str, data_kind: str = "tensor") -> None:
        self.name = name
        self.data_kind = data_kind


class Parameter:
    """A user-controlled parameter (serializable, DATA_FORMAT.md)."""

    def __init__(
        self,
        name: str,
        kind: str = "scalar",
        default: Any = None,
        required: bool = False,
        choices: Optional[List[Any]] = None,
        dtype: Optional[str] = None,
        browse: Optional[str] = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.default = default
        self.required = required
        self.choices = choices
        self.dtype = dtype
        self.browse = browse


class BaseNode:
    """Base class for all nodes (NODE_SYSTEM.md)."""

    type_name: str = "base"
    label: str = "Base"
    category: str = "Utility"
    inputs: List[NodeInput] = []
    outputs: List[NodeOutput] = []
    parameters: List[Parameter] = []

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = self._merge_params(params or {})

    def _merge_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        known = {p.name for p in self.parameters}
        merged: Dict[str, Any] = {p.name: p.default for p in self.parameters}
        for key, value in params.items():
            if key not in known:
                raise ValueError(
                    f"what: 节点 '{self.type_name}' 收到未知参数 '{key}'。\n"
                    f"where: nodes.base.BaseNode._merge_params\n"
                    f"how_to_fix: 已知参数 {sorted(known) or '(无)'}。"
                )
            merged[key] = value
        for p in self.parameters:
            if p.required and merged.get(p.name) is None:
                raise ValueError(
                    f"what: 节点 '{self.type_name}' 缺少必需参数 '{p.name}'。\n"
                    f"where: nodes.base.BaseNode._merge_params"
                )
        return merged

    @property
    def input_names(self) -> List[str]:
        return [i.name for i in self.inputs]

    @property
    def output_names(self) -> List[str]:
        return [o.name for o in self.outputs]

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        for inp in self.inputs:
            if inp.required and inp.name not in inputs:
                raise ValueError(
                    f"what: 节点 '{self.type_name}' 缺少必需输入 '{inp.name}'。\n"
                    f"where: nodes.base.BaseNode.validate_inputs"
                )

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"what: 节点 '{self.type_name}' 未实现 execute。\n"
            f"where: nodes.base.BaseNode.execute"
        )
