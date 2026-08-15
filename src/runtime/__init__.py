"""Runtime layer: node registry and graph execution (API.md runtime contract).

Validation, dependency ordering, execution queue, and readable errors.
Does not import Houdini or torch.
"""

from .registry import Registry, default_registry, register
from .executor import (
    RuntimeExecutionError,
    validate,
    execution_order,
    execute,
)

__all__ = [
    "Registry",
    "default_registry",
    "register",
    "RuntimeExecutionError",
    "validate",
    "execution_order",
    "execute",
]
