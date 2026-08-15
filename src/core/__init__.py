"""Core layer: tensor IR and graph document model.

Platform-neutral; does not import Houdini or torch (TORCH_BACKEND.md,
CROSS_PLATFORM.md).
"""

from .tensor import TensorValue, DATA_KINDS
from .document import GraphDocument, NodeModel, EdgeModel, PortModel

__all__ = [
    "TensorValue",
    "DATA_KINDS",
    "GraphDocument",
    "NodeModel",
    "EdgeModel",
    "PortModel",
]
