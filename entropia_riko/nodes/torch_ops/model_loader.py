"""Model loader node: load a torch model / state_dict from file (Stage 6)."""

from __future__ import annotations

from typing import Any, Dict

from ..base import BaseNode, NodeOutput, Parameter
from ...runtime.registry import register


@register("model_loader")
class ModelLoaderNode(BaseNode):
    """Load a torch model or state_dict from a file path.

    The loaded object is passed through the graph as a ``model`` value
    (not a TensorValue). Use an ``inference`` node to run it.
    """

    type_name = "model_loader"
    label = "Model Loader"
    category = "Model"
    inputs = []
    outputs = [NodeOutput("model", data_kind="model")]
    parameters = [
        Parameter("path", kind="scalar", required=True),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch
        from pathlib import Path

        path = Path(params["path"])
        if not path.exists():
            raise ValueError(
                f"what: 模型文件不存在: {path}\n"
                f"where: nodes.torch_ops.model_loader.execute\n"
                f"how_to_fix: 检查 path 参数是否正确。"
            )
        try:
            obj = torch.load(path, map_location=params["device"], weights_only=False)
        except Exception as exc:
            raise ValueError(
                f"what: 加载模型失败: {exc}\n"
                f"where: nodes.torch_ops.model_loader.execute\n"
                f"how_to_fix: 确认文件是 torch.save 保存的对象。"
            ) from exc
        return {"model": obj}
