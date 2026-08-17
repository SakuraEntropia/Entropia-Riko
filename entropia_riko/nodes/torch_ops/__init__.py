"""Torch-backed nodes (require torch at execution time)."""

from . import (
    activation,  # noqa: F401
    inference,  # noqa: F401
    linear,  # noqa: F401
    model_loader,  # noqa: F401
    ops,  # noqa: F401
    transformer,  # noqa: F401
)

try:
    from . import (
        advanced_nodes,  # noqa: F401  attention/norm/extra ops/losses
        api_nodes,  # noqa: F401  broad torch API coverage
        data_nodes,  # noqa: F401  mnist/dataloader/loss
        extra_nodes,  # noqa: F401  creation/conv/device/common ops
        hf_nodes,  # noqa: F401  diffusers/transformers pretrained models
        preview_nodes,  # noqa: F401  image/text/json loaders + previews
        wrangle,  # noqa: F401  inline Python code node
    )
except ImportError:  # pragma: no cover - torch not installed
    pass
