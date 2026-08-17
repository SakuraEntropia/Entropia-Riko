"""Shared utilities: config and logging (CROSS_PLATFORM.md)."""

from .config import AppConfig, default_config_dir
from .logging import get_logger

__all__ = ["AppConfig", "default_config_dir", "get_logger"]
