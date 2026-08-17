"""Project workspace operations: create / open / scan / validate.

The IDE-facing entry point. A project is a folder containing ``project.riko``;
these helpers treat it as such (not as "a folder of JSON files").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .manifest import (
    MANIFEST_FILENAME,
    ProjectManifest,
    is_project,
    validate_manifest,
)
from .templates import generate_project, get_template


def create_project(
    root: Path,
    name: str,
    template_id: str = "empty",
    gpu: bool = False,
    engine_version: str = "0.2.0",
) -> ProjectManifest:
    """Create a new project folder from a template and write its manifest."""
    get_template(template_id)  # validate the template exists (raises if unknown)
    manifest = ProjectManifest(
        name=name, template=template_id, gpu=gpu, engine_version=engine_version
    )
    generate_project(root, manifest)
    return manifest


def open_project(root: Path) -> ProjectManifest:
    """Load the manifest for an existing project."""
    return ProjectManifest.load(root)


def scan_project(root: Path) -> Dict[str, Any]:
    """Describe a project: manifest + directory tree + workflow files."""
    manifest = ProjectManifest.load(root)
    tree = _walk(root)
    workflows = _list_workflows(root)
    return {
        "manifest": manifest.to_dict(),
        "tree": tree,
        "workflows": workflows,
    }


def validate_project(root: Path) -> List[str]:
    """Validate a project folder; empty list means valid."""
    errors: List[str] = []
    if not is_project(root):
        errors.append(f"缺少 {MANIFEST_FILENAME}（不是 Entropia-Riko 项目）")
        return errors
    try:
        manifest = ProjectManifest.load(root)
    except (FileNotFoundError, ValueError) as exc:
        return [str(exc)]
    errors.extend(validate_manifest(manifest))
    return errors


def migrate_project(root: Path) -> Dict[str, Any]:
    """Migrate an older project in place (add missing manifest/fields)."""
    if not is_project(root):
        manifest = ProjectManifest(name=root.name, template="empty")
        generate_project(root, manifest)
        return {"status": "migrated", "created_manifest": True}
    manifest = ProjectManifest.load(root)
    # Ensure the modern engine_version is stamped.
    manifest.engine_version = "0.2.0"
    manifest.save(root)
    return {"status": "migrated", "created_manifest": False}


def _walk(root: Path, rel: str = "") -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue
        child_rel = f"{rel}/{child.name}" if rel else child.name
        if child.is_dir():
            items.append({"name": child.name, "path": child_rel, "type": "dir",
                          "children": _walk(child, child_rel)})
        else:
            items.append({"name": child.name, "path": child_rel, "type": "file"})
    return items


def _list_workflows(root: Path) -> List[str]:
    wf_dir = root / "workflows"
    if not wf_dir.is_dir():
        return []
    return sorted(
        p.relative_to(root).as_posix()
        for p in wf_dir.rglob("*.riko")
    )
