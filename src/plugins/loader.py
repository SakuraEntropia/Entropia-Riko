"""Plugin loader.

Scans ``plugins/*/plugin.json`` for manifests and imports their Python entry
modules (which typically register nodes via ``@register``). Plugins are
user-authored and loaded from the filesystem — they do not need to live under
``src/``.

Enable/disable state is persisted in ``plugins/state.json``::

    {"disabled": ["some_plugin", ...]}

Disabled plugins are listed by the API but their entry modules are not
imported, so their nodes stay unregistered (and disappear from the node
library). Loading is idempotent: a plugin's entry module is imported at most
once per process, so re-listing after toggling another plugin never re-runs
``@register``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from ..runtime.registry import default_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_DIR = PROJECT_ROOT / "plugins"
STATE_PATH = PLUGINS_DIR / "state.json"

# Manifest entries (name -> status dict) populated by load_plugins().
loaded_plugins: List[Dict[str, Any]] = []

# Process-level bookkeeping (idempotent imports + reversible toggles).
_disabled: Set[str] = set()
_imported: Set[str] = set()
_plugin_nodes: Dict[str, List[str]] = {}


def _import_file(module_name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 创建模块 spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _load_state() -> Set[str]:
    if not STATE_PATH.exists():
        return set()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        names = data.get("disabled", [])
        return {n for n in names if isinstance(n, str)}
    except Exception:
        return set()


def _save_state() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"disabled": sorted(_disabled)}, indent=2), encoding="utf-8"
    )


def _manifest_paths() -> List[Path]:
    if not PLUGINS_DIR.exists():
        return []
    return sorted(PLUGINS_DIR.glob("*/plugin.json"))


def _import_plugin(name: str, plugin_dir: Path, entry: str) -> List[str]:
    """Import a plugin's entry module, returning the node types it registered."""
    before = set(default_registry().list())
    _import_file(f"rik_plugin_{name}", plugin_dir / entry)
    after = set(default_registry().list())
    return sorted(after - before)


def load_plugins() -> List[Dict[str, Any]]:
    """Load every plugin under ``plugins/`` and return their manifests.

    Idempotent: entry modules are imported once per process; re-calling just
    refreshes the manifest list (used after toggles / uploads).
    """
    global _disabled
    _disabled = _load_state()
    loaded_plugins.clear()  # mutate in place so imported references stay valid

    for manifest_path in _manifest_paths():
        plugin_dir = manifest_path.parent
        name = plugin_dir.name
        info: Dict[str, Any] = {"name": name, "dir": name}

        manifest: Dict[str, Any] = {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            info.update(manifest)
        except Exception as exc:  # malformed manifest
            info["status"] = "error"
            info["error"] = f"{type(exc).__name__}: {exc}"

        enabled = name not in _disabled
        info["enabled"] = enabled

        if "status" not in info:
            entry = manifest.get("entry")
            if not enabled:
                info["status"] = "disabled"
            elif not entry:
                info["status"] = "loaded"  # manifest-only plugin (no nodes)
            else:
                entry_file = plugin_dir / entry
                if not entry_file.exists():
                    info["status"] = "error"
                    info["error"] = f"入口模块不存在: {entry}"
                else:
                    try:
                        if name not in _imported:
                            _plugin_nodes[name] = _import_plugin(name, plugin_dir, entry)
                            _imported.add(name)
                        info["status"] = "loaded"
                    except Exception as exc:
                        info["status"] = "error"
                        info["error"] = f"{type(exc).__name__}: {exc}"

        info["nodes"] = _plugin_nodes.get(name, [])
        loaded_plugins.append(info)

    return loaded_plugins


def _unregister_plugin(name: str) -> None:
    for type_name in _plugin_nodes.get(name, []):
        default_registry().unregister(type_name)
    _plugin_nodes.pop(name, None)
    _imported.discard(name)


def set_plugin_enabled(name: str, enabled: bool) -> List[Dict[str, Any]]:
    """Enable or disable a plugin by its directory name, then reload the list.

    Disabling unregisters the plugin's nodes; enabling (re)imports its entry.
    """
    if not (PLUGINS_DIR / name / "plugin.json").exists():
        raise FileNotFoundError(f"插件不存在: {name}")

    if enabled:
        was_disabled = name in _disabled
        _disabled.discard(name)
        if was_disabled:
            _imported.discard(name)  # nodes were unregistered on disable
    else:
        _disabled.add(name)
        _unregister_plugin(name)

    _save_state()
    return load_plugins()


def upload_plugin(name: str, code: str) -> List[Dict[str, Any]]:
    """Create (or replace) a plugin from raw Python source, then enable it.

    The source is written to ``plugins/<name>/nodes.py`` together with a
    generated ``plugin.json`` manifest, then imported (replacing any previous
    registration for that plugin directory).
    """
    safe = "".join(c for c in name if c.isalnum() or c in "-_").strip("-_") or "plugin"
    plugin_dir = PLUGINS_DIR / safe
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "nodes.py").write_text(code, encoding="utf-8")

    manifest_path = plugin_dir / "plugin.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                {
                    "name": safe,
                    "version": "1.0.0",
                    "description": f"User plugin loaded from file: {name}",
                    "author": "User",
                    "entry": "nodes.py",
                    "requires": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Replace any previous registration for this directory, then import fresh.
    _unregister_plugin(safe)
    _disabled.discard(safe)
    _save_state()
    return load_plugins()
