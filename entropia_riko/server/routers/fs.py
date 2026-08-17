"""Built-in file explorer endpoints (browse + import/export file or folder)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from ...core.document import GraphDocument
from ..state import get_working_root, resolve_working

router = APIRouter()


@router.get("/api/fs/list")
def fs_list(path: str = "") -> Dict[str, Any]:
    """List a directory (Windows-explorer style) for the built-in file explorer."""
    p = Path(path or "~").expanduser().resolve()
    if not p.is_dir():
        return {"status": "error", "error": f"不是目录: {path}"}
    try:
        entries: List[Dict[str, Any]] = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else 0,
            })
        return {
            "status": "success",
            "path": str(p),
            "parent": str(p.parent),
            "entries": entries,
        }
    except PermissionError as exc:
        return {"status": "error", "error": f"PermissionError: {exc}"}


@router.post("/api/fs/import")
def fs_import(body: Dict[str, Any]) -> Dict[str, Any]:
    """Copy an external file/folder into the working folder (import)."""
    src = str(body.get("src", "")).strip()
    if not src:
        return {"status": "error", "error": "缺少 src"}
    s = Path(src).expanduser().resolve()
    if not s.exists():
        return {"status": "error", "error": f"不存在: {src}"}
    dest = get_working_root() / s.name
    if dest.exists():
        return {"status": "error", "error": f"已存在: {dest.name}"}
    try:
        if s.is_dir():
            shutil.copytree(s, dest)
        else:
            shutil.copy2(s, dest)
        return {"status": "success", "path": str(dest)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/fs/export")
def fs_export(body: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a working-folder file/folder to a chosen destination directory (export)."""
    src = str(body.get("src", "")).strip()
    dest_dir = str(body.get("dest", "")).strip()
    if not src or not dest_dir:
        return {"status": "error", "error": "缺少 src 或 dest"}
    s = resolve_working(src)
    if s is None or not s.exists():
        return {"status": "error", "error": f"工作目录中不存在: {src}"}
    d = Path(dest_dir).expanduser().resolve()
    if not d.is_dir():
        return {"status": "error", "error": f"目标目录无效: {dest_dir}"}
    dest = d / s.name
    try:
        if s.is_dir():
            shutil.copytree(s, dest)
        else:
            shutil.copy2(s, dest)
        return {"status": "success", "path": str(dest)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/fs/save")
def fs_save(body: Dict[str, Any]) -> Dict[str, Any]:
    """Save the graph document to an absolute path (file-manager style save)."""
    path = str(body.get("path", "")).strip()
    doc = body.get("doc")
    if not path or doc is None:
        return {"status": "error", "error": "缺少 path 或 doc"}
    binary = body.get("format") == "binary"
    p = Path(path).expanduser().resolve()
    if binary:
        if p.suffix.lower() != ".ric":
            p = p.with_suffix(".ric")
    elif p.suffix.lower() != ".riko":
        p = p.with_suffix(".riko")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        gd = GraphDocument.from_dict(doc)
        if binary or p.suffix.lower() == ".ric":
            p.write_bytes(gd.to_binary())
        else:
            p.write_text(gd.to_json(indent=2), encoding="utf-8")
        return {"status": "success", "path": str(p), "name": p.stem}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
