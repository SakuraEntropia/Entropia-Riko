"""Runtime layer: node registry and graph execution (API.md runtime contract).

Validation, dependency ordering, execution queue, and readable errors.
Does not import Houdini or torch.
"""

from .executor import (
    RuntimeExecutionError,
    execute,
    execution_order,
    validate,
)
from .registry import Registry, default_registry, register

__all__ = [
    "Registry",
    "default_registry",
    "register",
    "RuntimeExecutionError",
    "validate",
    "execution_order",
    "execute",
]
