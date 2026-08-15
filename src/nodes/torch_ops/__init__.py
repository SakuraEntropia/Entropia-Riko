"""Torch-backed nodes (require torch at execution time)."""

from . import ops  # noqa: F401
from . import linear  # noqa: F401
from . import activation  # noqa: F401
from . import model_loader  # noqa: F401
from . import inference  # noqa: F401
from . import transformer  # noqa: F401

try:
    from . import api_nodes  # noqa: F401  broad torch API coverage
    from . import extra_nodes  # noqa: F401  creation/conv/device/common ops
    from . import data_nodes  # noqa: F401  mnist/dataloader/loss
    from . import advanced_nodes  # noqa: F401  attention/norm/extra ops/losses
    from . import wrangle  # noqa: F401  inline Python code node
    from . import preview_nodes  # noqa: F401  image/text/json loaders + previews
    from . import hf_nodes  # noqa: F401  diffusers/transformers pretrained models
except ImportError:  # pragma: no cover - torch not installed
    pass
