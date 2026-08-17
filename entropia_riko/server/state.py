"""Server shared state: the active working directory and filesystem helpers.

Routers import these helpers instead of touching globals directly, so the
working-directory concept lives in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.document import GraphDocument
from ..runtime.subgraph import PROJECT_ROOT

_RIKO_CONFIG_DIR = PROJECT_ROOT / ".riko"
_RIKO_CONFIG_FILE = _RIKO_CONFIG_DIR / "config.json"


def _default_working_root() -> Path:
    return (PROJECT_ROOT / "workflows").resolve()


def _load_working_root() -> Path:
    try:
        if _RIKO_CONFIG_FILE.exists():
            data = json.loads(_RIKO_CONFIG_FILE.read_text(encoding="utf-8"))
            p = Path(str(data.get("working_root", ""))).expanduser().resolve()
            if p.is_absolute() and p.is_dir():
                return p
    except Exception:
        pass
    return _default_working_root()


_working_root: Path = _load_working_root()
_working_root.mkdir(parents=True, exist_ok=True)


def get_working_root() -> Path:
    """Return the current working directory (project-as-unit root)."""
    return _working_root


def set_working_root(root: Path) -> Path:
    """Switch the working directory, persist the choice, and prepare its cache."""
    global _working_root
    _working_root = root
    cache = ensure_cache_dir(root)
    (cache / "config.json").write_text(
        json.dumps({"app": "Entropia Riko", "working_root": str(root)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _RIKO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _RIKO_CONFIG_FILE.write_text(
        json.dumps({"working_root": str(root)}, indent=2), encoding="utf-8"
    )
    return root


def ensure_cache_dir(root: Path) -> Path:
    """Create the tool's cache folder (`.riko`) inside a working folder."""
    cache = root / ".riko"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def resolve_working(rel: str) -> Optional[Path]:
    """Resolve a working-root-relative path (or an absolute path) to a Path.

    Returns ``None`` when the path escapes the working root (path-traversal guard).
    """
    rel = rel.replace("\\", "/").strip("/")
    if not rel:
        return None
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = _working_root.resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target


def walk_tree(root: Path, rel: str = "") -> List[Dict[str, Any]]:
    """Recursively map a directory into {name, path, type, children} entries."""
    items: List[Dict[str, Any]] = []
    if not root.exists():
        return items
    for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.name.startswith("."):
            continue  # hide the .riko cache folder and other dotfiles
        child_rel = f"{rel}/{child.name}" if rel else child.name
        if child.is_dir():
            items.append({
                "name": child.name,
                "path": child_rel,
                "type": "dir",
                "children": walk_tree(child, child_rel),
            })
        elif child.suffix in (".riko", ".ric"):
            items.append({
                "name": child.name,
                "path": child_rel,
                "type": "file",
            })
    return items


def read_project_doc(target: Path) -> GraphDocument:
    """Load a .riko (JSON) or .ric (binary) file into a GraphDocument."""
    if target.suffix == ".ric":
        return GraphDocument.from_binary(target.read_bytes())
    return GraphDocument.from_dict(json.loads(target.read_text(encoding="utf-8")))
