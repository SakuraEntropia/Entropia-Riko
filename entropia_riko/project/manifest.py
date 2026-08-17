"""Project manifest (``project.riko``) — the identity of an Entropia-Riko project.

A folder is an Entropia-Riko project iff it contains a ``project.riko`` manifest.
The manifest stores identity, runtime, dependency, and workflow metadata so the
IDE can treat a folder as a project rather than "a folder of JSON files".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_FILENAME = "project.riko"
MANIFEST_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectManifest:
    """Portable project identity, persisted as ``project.riko`` (JSON)."""

    def __init__(
        self,
        name: str,
        template: str = "empty",
        version: str = "0.1.0",
        engine_version: str = "0.2.0",
        gpu: bool = False,
        cuda: Optional[str] = None,
        python: str = "3.12",
        dependencies: Optional[Dict[str, List[str]]] = None,
        workflows: Optional[Dict[str, Any]] = None,
        created: Optional[str] = None,
    ) -> None:
        self.name = name
        self.template = template
        self.version = version
        self.engine_version = engine_version
        self.gpu = gpu
        self.cuda = cuda
        self.python = python
        self.dependencies: Dict[str, List[str]] = dict(dependencies or {})
        self.workflows: Dict[str, Any] = dict(workflows or {})
        self.created = created or _now()

    # ---------------------------------------------------------------- serialize
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "project": {
                "name": self.name,
                "version": self.version,
                "engine_version": self.engine_version,
                "template": self.template,
                "created": self.created,
            },
            "runtime": {
                "gpu": self.gpu,
                "cuda": self.cuda,
                "python": self.python,
            },
            "dependencies": self.dependencies,
            "workflows": self.workflows,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProjectManifest:
        p = d.get("project", {})
        r = d.get("runtime", {})
        return cls(
            name=str(p.get("name", "untitled")),
            template=str(p.get("template", "empty")),
            version=str(p.get("version", "0.1.0")),
            engine_version=str(p.get("engine_version", "0.2.0")),
            gpu=bool(r.get("gpu", False)),
            cuda=r.get("cuda"),
            python=str(r.get("python", "3.12")),
            dependencies=d.get("dependencies", {}),
            workflows=d.get("workflows", {}),
            created=p.get("created"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # --------------------------------------------------------------------- io
    def save(self, root: Path) -> Path:
        """Write the manifest into ``root/project.riko``."""
        path = root / MANIFEST_FILENAME
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, root: Path) -> ProjectManifest:
        """Load a manifest from ``root/project.riko``; raise if missing/invalid."""
        path = root / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"what: 不是 Entropia-Riko 项目（缺少 {MANIFEST_FILENAME}）。\n"
                f"where: project.manifest.ProjectManifest.load\n"
                f"how_to_fix: 在该目录创建项目，或指定包含 {MANIFEST_FILENAME} 的目录。"
            )
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"what: {MANIFEST_FILENAME} 损坏: {exc}\n"
                f"where: project.manifest.ProjectManifest.load"
            ) from exc


def is_project(root: Path) -> bool:
    """Whether `root` looks like an Entropia-Riko project folder."""
    return (root / MANIFEST_FILENAME).is_file()


def validate_manifest(manifest: ProjectManifest) -> List[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: List[str] = []
    if not manifest.name.strip():
        errors.append("project.name 不能为空")
    if not manifest.template.strip():
        errors.append("project.template 不能为空")
    if not manifest.engine_version.strip():
        errors.append("project.engine_version 不能为空")
    return errors
