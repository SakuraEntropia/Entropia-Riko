"""Graph endpoints: node registry, execution, and code export."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from ...core.document import GraphDocument
from ...core.tensor import TensorValue
from ...runtime.codegen import export_python, export_python_project
from ...runtime.codegen_tf import export_keras
from ...runtime.executor import RuntimeExecutionError, execute
from ...runtime.registry import default_registry

router = APIRouter()


@router.get("/api/nodes")
def get_nodes() -> Dict[str, Any]:
    """Return all registered node definitions for the UI library."""
    reg = default_registry()
    nodes: List[Dict[str, Any]] = []
    for type_name in reg.list():
        cls = reg.get(type_name)
        nodes.append(
            {
                "type": type_name,
                "label": getattr(cls, "label", type_name),
                "category": getattr(cls, "category", ""),
                "inputs": [
                    {"name": i.name, "data_kind": i.data_kind, "required": i.required}
                    for i in cls.inputs
                ],
                "outputs": [
                    {"name": o.name, "data_kind": o.data_kind} for o in cls.outputs
                ],
                "parameters": [
                    {
                        "name": p.name,
                        "kind": p.kind,
                        "default": p.default,
                        "required": p.required,
                        "dtype": p.dtype,
                        "browse": getattr(p, "browse", None),
                    }
                    for p in cls.parameters
                ],
            }
        )
    return {"nodes": nodes}


def _serialize_value(tv: TensorValue) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "shape": list(tv.shape),
        "dtype": tv.dtype,
        "device": tv.device,
        "summary": tv.summary(),
        "data_kind": tv.data_kind,
    }
    if tv.data_kind == "image_tensor":
        out["data"] = None  # images are too large to ship as JSON lists
        img = tv.metadata.get("preview", {}).get("image")
        if img:
            out["image"] = img  # base64 PNG data-URL
    else:
        out["data"] = tv.to_list()
    return out


@router.post("/api/execute")
def execute_graph(body: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a graph document (JSON) and return serialized outputs."""
    try:
        doc = GraphDocument.from_dict(body)
        outputs = execute(doc)
        result: Dict[str, Dict[str, Any]] = {}
        for nid, ports in outputs.items():
            result[nid] = {}
            for port, tv in ports.items():
                if isinstance(tv, TensorValue):
                    result[nid][port] = _serialize_value(tv)
                else:
                    result[nid][port] = {"summary": str(tv), "data": tv}
        return {"status": "success", "outputs": result, "errors": []}
    except RuntimeExecutionError as exc:
        return {"status": "error", "outputs": {}, "errors": [str(exc)]}
    except Exception as exc:
        return {"status": "error", "outputs": {}, "errors": [f"{type(exc).__name__}: {exc}"]}


@router.post("/api/export_python")
def export_python_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executable Python code from a graph document."""
    try:
        doc = GraphDocument.from_dict(body)
        code = export_python(doc)
        return {"status": "success", "code": code}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/export_keras")
def export_keras_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a tf.keras.Model script from a graph of TensorFlow/Keras nodes."""
    try:
        doc = GraphDocument.from_dict(body)
        code = export_keras(doc)
        return {"status": "success", "code": code}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/export_binary")
def export_binary_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Encode a graph document to the binary .ric format (base64 for download)."""
    try:
        doc = GraphDocument.from_dict(body)
        return {
            "status": "success",
            "base64": base64.b64encode(doc.to_binary()).decode("ascii"),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/export_project")
def export_project_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Write a multi-file PyTorch project into a directory (GitHub layout)."""
    doc = body.get("doc")
    target_dir = str(body.get("dir", "")).strip()
    if doc is None or not target_dir:
        return {"status": "error", "error": "缺少 doc 或 dir"}
    root = Path(target_dir).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        files = export_python_project(GraphDocument.from_dict(doc))
        written: List[str] = []
        for f in files:
            p = root / f["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f["content"], encoding="utf-8")
            written.append(f["path"])
        return {"status": "success", "root": str(root), "files": written}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
