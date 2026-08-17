"""Plugin endpoints: list / toggle / upload user plugins."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from ...plugins.loader import loaded_plugins, set_plugin_enabled, upload_plugin

router = APIRouter()


@router.get("/api/plugins")
def get_plugins() -> Dict[str, Any]:
    """List loaded user plugins (enabled/disabled status + registered nodes)."""
    return {"plugins": loaded_plugins}


@router.post("/api/plugins/toggle")
def toggle_plugin(body: Dict[str, Any]) -> Dict[str, Any]:
    """Enable or disable a plugin by name (unregisters its nodes when disabled)."""
    name = str(body.get("name", "")).strip()
    enabled = bool(body.get("enabled", True))
    if not name:
        return {"status": "error", "error": "缺少 name"}
    try:
        return {"status": "success", "plugins": set_plugin_enabled(name, enabled)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/plugins/upload")
def upload_plugin_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Install a plugin from raw Python source (the UI reads a .py file client-side)."""
    name = str(body.get("name", "")).strip()
    code = str(body.get("code", ""))
    if not name or not code:
        return {"status": "error", "error": "缺少 name 或 code"}
    try:
        return {"status": "success", "plugins": upload_plugin(name, code)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
