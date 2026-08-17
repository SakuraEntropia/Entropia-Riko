"""Workflow file endpoints: list/read/save .riko/.ric files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Request

from ...core.document import GraphDocument
from ...runtime.subgraph import MODULE_SEARCH_PATHS, PROJECT_ROOT, resolve_graph_file

router = APIRouter()

_WORKFLOWS_DIR = PROJECT_ROOT / "workflows"


def _safe_project_file(path: Path) -> bool:
    """Only allow .riko / .ric files inside the project root (path guard)."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return (
        resolved.suffix in (".riko", ".ric")
        and str(resolved).startswith(str(PROJECT_ROOT.resolve()) + "/")
        and resolved.is_file()
    )


def _read_doc_dict(p: Path) -> Dict[str, Any]:
    """Read a graph document from either format: .riko (ASCII JSON) or .ric (binary)."""
    if p.suffix == ".ric":
        return GraphDocument.from_binary(p.read_bytes()).to_dict()
    return json.loads(p.read_text(encoding="utf-8"))


def _scan_riko_files() -> List[Dict[str, Any]]:
    """Scan workflows/ and examples/ for .riko/.ric files with their import targets."""
    roots = [PROJECT_ROOT / "workflows", PROJECT_ROOT / "examples"]
    seen: set = set()
    files: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.riko")) + sorted(root.rglob("*.ric")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            info: Dict[str, Any] = {
                "name": p.stem,
                "path": rel,
                "format": "binary" if p.suffix == ".ric" else "ascii",
                "imports": [],
            }
            try:
                data = _read_doc_dict(p)
                for n in data.get("nodes", []):
                    if n.get("type_name") in ("graph_reference", "import"):
                        spec = (
                            n.get("parameters", {}).get("file")
                            or n.get("parameters", {}).get("module")
                        )
                        if not spec:
                            continue
                        resolved = resolve_graph_file(spec, base_dir=p.parent)
                        info["imports"].append({
                            "spec": spec,
                            "path": resolved.relative_to(PROJECT_ROOT).as_posix() if resolved else None,
                            "resolved": resolved is not None,
                        })
            except Exception:
                # Malformed file: still list it, just without import metadata.
                pass
            files.append(info)
    return files


@router.get("/api/files")
def list_files() -> Dict[str, Any]:
    """List all .riko files with their import (graph_reference/import) targets."""
    return {
        "files": _scan_riko_files(),
        "search_paths": [str(p.relative_to(PROJECT_ROOT)) for p in MODULE_SEARCH_PATHS],
    }


@router.get("/api/files/content")
def get_file_content(path: str) -> Dict[str, Any]:
    """Return the JSON content of a .riko/.ric file (relative to project root)."""
    target = PROJECT_ROOT / path
    if not _safe_project_file(target):
        return {"status": "error", "error": f"文件不存在或非法: {path}"}
    try:
        doc = _read_doc_dict(target)
        return {"status": "success", "doc": doc}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/files/decode")
async def decode_file(request: Request) -> Dict[str, Any]:
    """Decode an uploaded .ric binary file body into a graph document JSON."""
    try:
        raw = await request.body()
        doc = GraphDocument.from_binary(raw)
        return {"status": "success", "doc": doc.to_dict()}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/files/save")
def save_file(body: Dict[str, Any]) -> Dict[str, Any]:
    """Save a graph document to workflows/<name>.riko (ASCII) or .ric (binary)."""
    name = str(body.get("name", "")).strip()
    doc = body.get("doc")
    if not name or doc is None:
        return {"status": "error", "error": "缺少 name 或 doc"}
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_.") or "workflow"

    binary = body.get("format") == "binary" or safe_name.lower().endswith(".ric")
    if not safe_name.lower().endswith((".riko", ".ric")):
        safe_name += ".ric" if binary else ".riko"

    path = _WORKFLOWS_DIR / safe_name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        gd = GraphDocument.from_dict(doc)
        if binary:
            path.write_bytes(gd.to_binary())
        else:
            path.write_text(gd.to_json(indent=2), encoding="utf-8")
        return {
            "status": "success",
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "format": "binary" if binary else "ascii",
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
