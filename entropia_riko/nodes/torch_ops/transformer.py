"""Transformer encoder node (nn.TransformerEncoder, Stage 6 example).

A full multi-layer transformer encoder. Input shape (batch, seq, d_model)
when batch_first=True; output shape is the same.
"""

from __future__ import annotations

from typing import Any, Dict

from ...backend.converter import from_torch, to_torch
from ...backend.device import resolve_device
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


@register("transformer_encoder")
class TransformerEncoderNode(BaseNode):
    """Multi-layer Transformer encoder (nn.TransformerEncoder)."""

    type_name = "transformer_encoder"
    label = "Transformer Encoder"
    category = "Neural"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("output", data_kind="tensor")]
    parameters = [
        Parameter("d_model", kind="scalar", required=True, dtype="int"),
        Parameter("nhead", kind="scalar", required=True, dtype="int"),
        Parameter("num_layers", kind="scalar", default=1, dtype="int"),
        Parameter("dim_feedforward", kind="scalar", default=None, dtype="int"),
        Parameter("batch_first", kind="scalar", default=True),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(
        self,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch
        import torch.nn as nn

        dev = resolve_device(params["device"])
        d_model = int(params["d_model"])
        nhead = int(params["nhead"])
        num_layers = int(params["num_layers"])

        if d_model % nhead != 0:
            raise ValueError(
                f"what: d_model({d_model}) 必须能被 nhead({nhead}) 整除。\n"
                f"where: nodes.torch_ops.transformer.execute\n"
                f"how_to_fix: 调整 d_model / nhead。"
            )

        dim_ff = int(params["dim_feedforward"]) if params["dim_feedforward"] is not None else d_model * 4
        batch_first = bool(params["batch_first"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=batch_first,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        encoder = encoder.to(dev).eval()

        x = to_torch(inputs["x"], dev)
        with torch.no_grad():
            y = encoder(x)
        return {"output": from_torch(y, metadata={"backend": "torch", "node": "transformer_encoder"})}
