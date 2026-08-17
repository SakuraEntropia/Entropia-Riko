"""Strong-typed port system (DATA_FORMAT.md, NODE_SYSTEM.md).

Defines the set of legal port data kinds and the compatibility rules used to
reject invalid connections (e.g. ``text → audio``). This is the single source
of truth for port typing; nodes declare ports with one of these kinds.
"""

from __future__ import annotations

from typing import Tuple

# Legal port data kinds (snake_case; the UI shows them as-is).
DATA_KINDS: Tuple[str, ...] = (
    "scalar",        # NUMBER
    "tensor",        # TENSOR
    "image_tensor",  # IMAGE / IMAGE_BATCH
    "mask",          # MASK
    "embedding",     # EMBEDDING
    "latent",        # LATENT
    "audio",         # AUDIO
    "video",         # VIDEO
    "model",         # MODEL
    "checkpoint",    # CHECKPOINT
    "dataset",       # DATASET
    "text",          # TEXT
    "json",          # JSON
    "config",        # CONFIG
    "metadata",      # METADATA
    "file",          # FILE
    "folder",        # FOLDER
    "unknown",       # untyped / pass-through
)

# Subtype → parent widening (a subtype may flow into a parent-kind port).
# e.g. image_tensor → tensor is allowed, tensor → image_tensor is not.
_PORT_HIERARCHY = {
    "image_tensor": "tensor",
    "mask": "image_tensor",
    "embedding": "tensor",
    "latent": "tensor",
    "audio": "tensor",
    "video": "tensor",
    "checkpoint": "file",
    "config": "json",
    "metadata": "json",
    "dataset": "folder",
}


def is_compatible(src: str, dst: str) -> bool:
    """Whether a value of kind `src` can flow into a port of kind `dst`.

    Exact match, or a subtype widening to a parent. ``unknown`` ports accept
    anything and may connect anywhere (untyped escape hatch).
    """
    if src == dst:
        return True
    if src == "unknown" or dst == "unknown":
        return True
    seen = set()
    kind = src
    while kind in _PORT_HIERARCHY and kind not in seen:
        seen.add(kind)
        kind = _PORT_HIERARCHY[kind]
        if kind == dst:
            return True
    return False


def normalize(kind: str) -> str:
    """Return `kind` if legal, else ``unknown``."""
    return kind if kind in DATA_KINDS else "unknown"
