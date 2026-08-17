"""Node registry (NODE_SYSTEM.md, API.md).

Maps node type names to node classes. Supports registration, lookup,
listing, and duplicate prevention.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type


class Registry:
    """Node type registry."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Type] = {}

    def register(self, type_name: str, node_cls: Type) -> None:
        if not type_name:
            raise ValueError(
                "what: 节点类型名不能为空。\n"
                "where: runtime.registry.Registry.register"
            )
        if type_name in self._nodes:
            existing = self._nodes[type_name]
            raise ValueError(
                f"what: 节点类型 '{type_name}' 已注册为 {existing.__name__}。\n"
                f"where: runtime.registry.Registry.register\n"
                f"how_to_fix: 使用不同类型名，或先移除旧注册。"
            )
        self._nodes[type_name] = node_cls

    def unregister(self, type_name: str) -> None:
        """Remove a node type (no-op if absent). Used by plugin toggling."""
        self._nodes.pop(type_name, None)

    def get(self, type_name: str) -> Type:
        try:
            return self._nodes[type_name]
        except KeyError:
            available = ", ".join(self.list()) or "(空)"
            raise KeyError(
                f"what: 未注册的节点类型 '{type_name}'。\n"
                f"where: runtime.registry.Registry.get\n"
                f"how_to_fix: 已注册类型: {available}"
            ) from None

    def list(self) -> List[str]:
        return sorted(self._nodes.keys())

    def __contains__(self, type_name: object) -> bool:
        return type_name in self._nodes


_default_registry = Registry()


def default_registry() -> Registry:
    """Process-level default registry."""
    return _default_registry


def register(type_name: str, registry: Optional[Registry] = None):
    """Class decorator: register a node class into a registry.

    Usage::

        @register("add")
        class AddNode(BaseNode): ...
    """
    target = registry if registry is not None else _default_registry

    def decorator(cls: Type) -> Type:
        target.register(type_name, cls)
        return cls

    return decorator
