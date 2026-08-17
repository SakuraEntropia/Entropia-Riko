"""Reusable node definitions.

Importing this package registers the built-in nodes into the default
runtime registry. torch-backed nodes register too but require torch only
at execution time.
"""

from . import math  # noqa: F401
from . import torch_ops  # noqa: F401
from . import subgraph  # noqa: F401  graph_input/output/reference
from . import tf_ops  # noqa: F401  TensorFlow/Keras nodes (optional backend)
