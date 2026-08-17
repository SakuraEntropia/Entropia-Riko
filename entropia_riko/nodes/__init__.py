"""Reusable node definitions.

Importing this package registers the built-in nodes into the default
runtime registry. torch-backed nodes register too but require torch only
at execution time.
"""

from . import (
    math,  # noqa: F401
    subgraph,  # noqa: F401  graph_input/output/reference
    tf_ops,  # noqa: F401  TensorFlow/Keras nodes (optional backend)
    torch_ops,  # noqa: F401
)
