"""Bake/cache system — unified artifact management.

Generated artifacts (checkpoints, rendered images, processed datasets, …) are
stored under ``bakes/`` with a metadata sidecar recording their provenance, so
results can be reused and reproduced instead of recomputed blindly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bake_dir(project_root: Path, name: str) -> Path:
    """Return the bake directory for `name` (created lazily)."""
    d = project_root / "bakes" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def bake_artifact(
    project_root: Path,
    name: str,
    content: bytes,
    source_node: str,
    parameters: Dict[str, Any],
    workflow_version: str = "1",
    dependencies: Optional[List[str]] = None,
) -> Path:
    """Persist `content` as a baked artifact and record its provenance."""
    d = bake_dir(project_root, name)
    artifact = d / "artifact"
    artifact.write_bytes(content)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "name": name,
                "source_node": source_node,
                "parameters": parameters,
                "workflow_version": workflow_version,
                "dependencies": dependencies or [],
                "created": _now(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact


def get_bake_metadata(project_root: Path, name: str) -> Optional[Dict[str, Any]]:
    meta = project_root / "bakes" / name / "metadata.json"
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def get_bake_artifact(project_root: Path, name: str) -> Optional[Path]:
    artifact = project_root / "bakes" / name / "artifact"
    return artifact if artifact.is_file() else None


def is_cache_valid(project_root: Path, name: str, parameters: Dict[str, Any]) -> bool:
    """Whether a cached artifact exists with matching parameters (cache hit)."""
    meta = get_bake_metadata(project_root, name)
    if meta is None:
        return False
    return meta.get("parameters") == parameters


def list_bakes(project_root: Path) -> List[Dict[str, Any]]:
    base = project_root / "bakes"
    if not base.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        entry: Dict[str, Any] = {"name": d.name}
        meta = get_bake_metadata(project_root, d.name)
        if meta:
            entry.update({
                "source_node": meta.get("source_node"),
                "workflow_version": meta.get("workflow_version"),
                "created": meta.get("created"),
            })
        out.append(entry)
    return out
