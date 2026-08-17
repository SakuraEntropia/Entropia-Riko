"""Project (working-directory) mini file manager endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from ...core.document import GraphDocument
from ...project import (
    PROJECT_TEMPLATES,
    create_project,
    list_experiments,
    migrate_project,
    record_experiment,
    scan_project,
    validate_project,
)
from ...runtime.codegen import export_python
from ..state import (
    get_working_root,
    read_project_doc,
    resolve_working,
    set_working_root,
    walk_tree,
)

router = APIRouter()


@router.get("/api/project/templates")
def project_templates() -> Dict[str, Any]:
    """List available AI project templates."""
    return {"templates": [t.to_dict() for t in PROJECT_TEMPLATES]}


@router.get("/api/project/scan")
def project_scan() -> Dict[str, Any]:
    """Scan the working project: manifest + tree + workflows."""
    try:
        data = scan_project(get_working_root())
        return {"status": "success", **data}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/validate")
def project_validate() -> Dict[str, Any]:
    """Validate the working project folder."""
    errors = validate_project(get_working_root())
    return {"status": "success" if not errors else "error", "errors": errors}


@router.post("/api/project/migrate")
def project_migrate() -> Dict[str, Any]:
    """Migrate the working project in place (add manifest / stamp version)."""
    try:
        result = migrate_project(get_working_root())
        return {"status": "success", **result}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/experiment")
def project_record_experiment(body: Dict[str, Any]) -> Dict[str, Any]:
    """Record an experiment under the working project (workflow + params + metrics)."""
    try:
        exp = record_experiment(
            get_working_root(),
            workflow=body.get("workflow", {}),
            parameters=body.get("parameters", {}),
            metrics=body.get("metrics", {}),
            seed=body.get("seed"),
        )
        return {"status": "success", "experiment": exp.name}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.get("/api/project/experiments")
def project_list_experiments() -> Dict[str, Any]:
    """List recorded experiments under the working project."""
    return {"experiments": list_experiments(get_working_root())}


@router.get("/api/project/tree")
def project_tree() -> Dict[str, Any]:
    """Return the working directory as a recursive tree."""
    root = get_working_root()
    return {"root": str(root), "tree": walk_tree(root)}


@router.post("/api/project/set_root")
def project_set_root(body: Dict[str, Any]) -> Dict[str, Any]:
    """Import a working folder: validate it, create its `.riko` cache, persist."""
    raw = str(body.get("path", "")).strip()
    if not raw:
        return {"status": "error", "error": "缺少 path"}
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        return {"status": "error", "error": f"不是有效目录: {raw}"}
    root = set_working_root(p)
    return {"status": "success", "root": str(root), "tree": walk_tree(root)}


@router.post("/api/project/new")
def project_new(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new project folder from an AI template and open it."""
    target = str(body.get("dir", "")).strip()
    name = str(body.get("name", "")).strip() or Path(target).name
    template = str(body.get("template", "empty")).strip()
    gpu = bool(body.get("gpu", False))
    if not target:
        return {"status": "error", "error": "缺少 dir"}
    root = Path(target).expanduser().resolve()
    try:
        create_project(root, name=name, template_id=template, gpu=gpu)
        set_working_root(root)
        return {"status": "success", "root": str(root), "tree": walk_tree(root)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/create")
def project_create(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new empty workflow file under the working folder."""
    name = str(body.get("name", "")).strip()
    subdir = str(body.get("dir", "")).strip().replace("\\", "/").strip("/")
    if not name:
        return {"status": "error", "error": "缺少 name"}
    safe = "".join(c for c in name if c.isalnum() or c in "-_.") or "workflow"
    if not safe.lower().endswith((".riko", ".ric")):
        safe += ".riko"
    base = get_working_root()
    if subdir:
        target_dir = resolve_working(subdir)
        if target_dir is None or not target_dir.is_dir():
            return {"status": "error", "error": f"目录无效: {subdir}"}
        base = target_dir
    path = base / safe
    if path.exists():
        return {"status": "error", "error": f"文件已存在: {safe}"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(GraphDocument().to_json(indent=2), encoding="utf-8")
        rel = f"{subdir}/{safe}" if subdir else safe
        return {"status": "success", "path": rel}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/mkdir")
def project_mkdir(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a folder under the working folder."""
    name = str(body.get("name", "")).strip()
    subdir = str(body.get("dir", "")).strip().replace("\\", "/").strip("/")
    if not name:
        return {"status": "error", "error": "缺少 name"}
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "folder"
    base = get_working_root()
    if subdir:
        target_dir = resolve_working(subdir)
        if target_dir is None or not target_dir.is_dir():
            return {"status": "error", "error": f"目录无效: {subdir}"}
        base = target_dir
    path = base / safe
    try:
        path.mkdir(parents=True, exist_ok=False)
        rel = f"{subdir}/{safe}" if subdir else safe
        return {"status": "success", "path": rel}
    except FileExistsError:
        return {"status": "error", "error": f"已存在: {safe}"}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/delete")
def project_delete(body: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a file or empty folder inside the working folder (path-guarded)."""
    rel = str(body.get("path", "")).strip()
    target = resolve_working(rel)
    if target is None or not target.exists():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    try:
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()
        return {"status": "success", "path": rel}
    except OSError as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/rename")
def project_rename(body: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a file or folder inside the working folder."""
    rel = str(body.get("path", "")).strip()
    new_name = str(body.get("newName", "")).strip()
    target = resolve_working(rel)
    if target is None or not target.exists():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    if not new_name or "/" in new_name or "\\" in new_name:
        return {"status": "error", "error": "非法名称"}
    try:
        target.rename(target.with_name(new_name))
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        new_rel = f"{parent}/{new_name}" if parent else new_name
        return {"status": "success", "path": new_rel}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/move")
def project_move(body: Dict[str, Any]) -> Dict[str, Any]:
    """Move a file/folder into another folder (drag & drop in the file tree)."""
    rel = str(body.get("path", "")).strip()
    target_dir = str(body.get("targetDir", "")).strip().replace("\\", "/").strip("/")
    src = resolve_working(rel)
    if src is None or not src.exists():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    base = get_working_root() if not target_dir else resolve_working(target_dir)
    if base is None or not base.is_dir():
        return {"status": "error", "error": f"目标目录无效: {target_dir}"}
    dest = base / src.name
    if dest.exists():
        return {"status": "error", "error": f"已存在: {src.name}"}
    try:
        src.rename(dest)
        new_rel = f"{target_dir}/{src.name}" if target_dir else src.name
        return {"status": "success", "path": new_rel}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/project/code")
def project_code(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return the exported PyTorch code for a .riko/.ric file in the working folder."""
    rel = str(body.get("path", "")).strip()
    target = resolve_working(rel)
    if target is None or not target.is_file():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    try:
        code = export_python(read_project_doc(target))
        return {"status": "success", "code": code}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.get("/api/project/open")
def project_open(path: str) -> Dict[str, Any]:
    """Return the JSON content of a .riko/.ric file in the working folder."""
    rel = str(path).strip()
    target = resolve_working(rel)
    if target is None or not target.is_file():
        return {"status": "error", "error": f"不存在或非法: {rel}"}
    try:
        return {"status": "success", "doc": read_project_doc(target).to_dict()}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
