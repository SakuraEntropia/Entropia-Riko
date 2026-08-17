"""Core layer: tensor IR and graph document model.

Platform-neutral; does not import Houdini or torch (TORCH_BACKEND.md,
CROSS_PLATFORM.md).
"""

from .document import EdgeModel, GraphDocument, NodeModel, PortModel
from .tensor import TensorValue
from .types import DATA_KINDS

__all__ = [
    "TensorValue",
    "DATA_KINDS",
    "GraphDocument",
    "NodeModel",
    "EdgeModel",
    "PortModel",
]
