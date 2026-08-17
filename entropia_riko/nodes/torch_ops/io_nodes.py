"""Professional I/O nodes: file input, dataset, checkpoint save/load.

These make a workflow portable: assets are addressed by relative/project paths
instead of random absolute paths, and datasets/checkpoints are first-class
typed values (`dataset`, `checkpoint`) rather than bare tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ...core.tensor import TensorValue
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from .model_loader import _build_structure, _load_model, _save_model


def _scan_dataset(folder: Path, split: str) -> Dict[str, Any]:
    files = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    return {
        "path": str(folder),
        "split": split,
        "count": len(files),
        "files": files[:200],
    }


def _resolve_project_path(raw: str, context: Dict[str, Any]) -> Path:
    """Resolve a path; relative paths anchor to the working/project root.

    Prefer project-relative paths (``datasets/raw``) over absolute ones so a
    project stays portable across machines.
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        base = context.get("working_root") or context.get("project_root")
        if base:
            p = Path(base) / p
    return p.resolve()


@register("file_input")
class FileInputNode(BaseNode):
    """Load an external file or folder and expose it as a typed FILE / FOLDER /
    DATASET value (Houdini-style file node)."""

    type_name = "file_input"
    label = "File Input"
    category = "IO"
    inputs = []
    outputs = [
        NodeOutput("file", data_kind="file"),
        NodeOutput("folder", data_kind="folder"),
        NodeOutput("dataset", data_kind="dataset"),
    ]
    parameters = [Parameter("path", kind="path", browse="open", required=True)]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        p = _resolve_project_path(params["path"], context)
        if not p.exists():
            raise ValueError(
                f"what: 路径不存在: {p}\n"
                f"where: nodes.torch_ops.io_nodes.FileInputNode.execute\n"
                f"how_to_fix: 检查 path 参数。"
            )
        if p.is_dir():
            return {
                "folder": TensorValue.from_value(str(p), kind="folder"),
                "dataset": TensorValue.from_value(_scan_dataset(p, "all"), kind="dataset"),
            }
        return {"file": TensorValue.from_value(str(p), kind="file")}


@register("dataset")
class DatasetNode(BaseNode):
    """Build a DATASET from a folder: scan files, record metadata + split."""

    type_name = "dataset"
    label = "Dataset"
    category = "Data"
    inputs = [NodeInput("folder", data_kind="folder", required=False)]
    outputs = [NodeOutput("dataset", data_kind="dataset")]
    parameters = [
        Parameter("path", kind="path", browse="open", default=""),
        Parameter("split", kind="scalar", default="train", choices=["train", "val", "test", "all"]),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        folder_value = inputs.get("folder")
        path = (
            folder_value.data
            if isinstance(folder_value, TensorValue) and folder_value.data_kind == "folder"
            else str(params.get("path", "")).strip()
        )
        if not path:
            raise ValueError(
                "what: 缺少数据集路径。\n"
                "where: nodes.torch_ops.io_nodes.DatasetNode.execute\n"
                "how_to_fix: 连接 folder 输入，或填写 path 参数。"
            )
        folder = _resolve_project_path(path, context)
        if not folder.is_dir():
            raise ValueError(f"what: 不是目录: {folder}")
        meta = _scan_dataset(folder, str(params.get("split", "all")))
        return {"dataset": TensorValue.from_value(meta, kind="dataset")}


@register("checkpoint_save")
class CheckpointSaveNode(BaseNode):
    """Persist a model to a checkpoint file (safetensors / torch state_dict)."""

    type_name = "checkpoint_save"
    label = "Save Checkpoint"
    category = "Model"
    inputs = [NodeInput("model", data_kind="model", required=True)]
    outputs = [NodeOutput("checkpoint", data_kind="checkpoint")]
    parameters = [
        Parameter("path", kind="path", browse="save", default="~/checkpoints/model.safetensors"),
        Parameter("format", kind="scalar", default="auto", choices=["auto", "safetensors", "pt", "pth"]),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        p = _resolve_project_path(params["path"], context)
        path = _save_model(inputs["model"], str(p), params["format"])
        return {"checkpoint": TensorValue.from_value(path, kind="checkpoint")}


@register("checkpoint_load")
class CheckpointLoadNode(BaseNode):
    """Load a checkpoint file and rebuild the model (via `module` or `template`)."""

    type_name = "checkpoint_load"
    label = "Load Checkpoint"
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
        p = _resolve_project_path(params["path"], context)
        model = _load_model(str(p), params["device"])
        template = inputs.get("template")
        module = params.get("module")
        if isinstance(model, dict):
            if template is not None and hasattr(template, "load_state_dict"):
                template.load_state_dict(model)
                return {"model": template}
            if module:
                structure = _build_structure(module)
                structure.load_state_dict(model)
                return {"model": structure}
        return {"model": model}
