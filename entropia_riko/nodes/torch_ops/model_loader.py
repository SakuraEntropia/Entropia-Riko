"""Model save/load nodes — "train to model" and "load from model".

`save_model` serializes a model flowing through the graph to a file
(safetensors / torch). `model_loader` loads a model back (safetensors or
torch) and returns it for `inference`. Both expose a `path` parameter with a
file picker (`browse="open" | "save"`), like a Houdini file node.

Weights are always persisted as a *state_dict* (safetensors is state-dict only,
and a state_dict is a plain tensor dict that pickles cleanly — codegen'd
``GraphModel`` classes are dynamic and cannot be `torch.save`d whole). To make a
state_dict callable again, `model_loader` accepts a `module` parameter naming the
model-block ``.riko`` whose structure it rebuilds and loads the weights into.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register


def _state_dict(model: Any) -> Dict[str, Any]:
    """Return a state_dict for a model (nn.Module) or a bare dict as-is."""
    if hasattr(model, "state_dict"):
        return model.state_dict()
    if isinstance(model, dict):
        return model
    raise ValueError(
        "what: 无法序列化的模型对象。\n"
        "where: nodes.torch_ops.model_loader._state_dict\n"
        "how_to_fix: 传入 nn.Module 或 state_dict (dict)。"
    )


def _save_model(model: Any, path: str, fmt: str) -> str:
    """Save a model's state_dict to `path`; `fmt` "auto" chooses by extension."""
    import torch

    p = _resolve_model_path(path)
    state = _state_dict(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".safetensors" and fmt in ("auto", "safetensors"):
        from safetensors.torch import save_file

        save_file(state, str(p))
    else:
        torch.save(state, str(p))  # state_dict (plain dict, pickles cleanly)
    return str(p)


def _resolve_model_path(path: str) -> Path:
    """Resolve a model path: absolute/existing as-is; else search module paths."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    from ...runtime.subgraph import MODULE_SEARCH_PATHS, PROJECT_ROOT

    for sp in MODULE_SEARCH_PATHS:
        if (sp / p).exists():
            return sp / p
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    return p


def _load_model(path: str, device: str) -> Any:
    """Load a model file. `.safetensors` → state_dict (dict); `.pt`/`.pth` → torch.load."""
    import torch

    p = _resolve_model_path(path)
    if not p.exists():
        raise ValueError(
            f"what: 模型文件不存在: {path}\n"
            f"where: nodes.torch_ops.model_loader._load_model\n"
            f"how_to_fix: 检查 path 参数，或先用 save_model 节点保存。"
        )
    if p.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(p), device=device)
    return torch.load(str(p), map_location=device, weights_only=False)


def _build_structure(module: str) -> Any:
    """Rebuild an nn.Module structure from a model-block `.riko` (module name)."""
    from ...core.document import GraphDocument
    from ...runtime.codegen import export_python
    from ...runtime.subgraph import resolve_graph_file

    path = resolve_graph_file(module)
    if path is None:
        raise ValueError(
            f"what: 模型块 '{module}' 未找到。\n"
            f"where: nodes.torch_ops.model_loader._build_structure\n"
            f"how_to_fix: 确保 examples/models/{module}.riko 存在，或 module 参数填完整路径。"
        )
    doc = GraphDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))
    code = export_python(doc)
    namespace: Dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 - generated module under our control
    if "GraphModel" not in namespace:
        raise ValueError(f"模型块 '{module}' 无法构建结构（无 GraphModel）。")
    return namespace["GraphModel"]()


@register("model_loader")
class ModelLoaderNode(BaseNode):
    """Load a model from a file (safetensors / torch state_dict).

    The optional `template` input supplies a ready-made model structure; the
    optional `module` parameter names a model-block `.riko` whose structure is
    rebuilt from codegen when the file only holds a state_dict."""

    type_name = "model_loader"
    label = "Load Model"
    category = "Model"
    inputs = [NodeInput("template", data_kind="model", required=False, default=None)]
    outputs = [NodeOutput("model", data_kind="model")]
    parameters = [
        Parameter("path", kind="path", browse="open", required=True),
        Parameter("module", kind="scalar", default=None),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        model = _load_model(params["path"], params["device"])
        template = inputs.get("template")
        module = params.get("module")
        if isinstance(model, dict):
            # A bare state_dict needs a structure to become a callable model.
            if template is not None and hasattr(template, "load_state_dict"):
                template.load_state_dict(model)
                return {"model": template}
            if module:
                structure = _build_structure(module)
                structure.load_state_dict(model)
                return {"model": structure}
        return {"model": model}


@register("save_model")
class SaveModelNode(BaseNode):
    """Serialize a model's state_dict to disk ("train to model"). Feed a trained
    or loaded model into `model`; the file path is picked with a file dialog."""

    type_name = "save_model"
    label = "Save Model"
    category = "Model"
    inputs = [NodeInput("model", data_kind="model", required=True)]
    outputs = [NodeOutput("path", data_kind="text")]
    parameters = [
        Parameter("path", kind="path", browse="save", default="~/entropia_model.safetensors"),
        Parameter(
            "format",
            kind="scalar",
            default="auto",
            choices=["auto", "safetensors", "pt", "pth"],
        ),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"path": _save_model(inputs["model"], params["path"], params["format"])}
