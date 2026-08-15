"""Configuration helpers (CROSS_PLATFORM.md).

Uses pathlib.Path; no hardcoded platform paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def default_config_dir() -> Path:
    """User config directory (cross-platform via pathlib)."""
    return Path.home() / ".entropia_riko"


class AppConfig:
    """Simple serializable app configuration."""

    def __init__(
        self,
        device: str = "cpu",
        log_level: str = "info",
        extra: Dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self.log_level = log_level
        self.extra: Dict[str, Any] = dict(extra or {})

    def to_dict(self) -> Dict[str, Any]:
        return {"device": self.device, "log_level": self.log_level, "extra": dict(self.extra)}
