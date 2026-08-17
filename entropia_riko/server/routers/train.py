"""Training endpoints: batch train + streaming per-step loss."""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ...core.document import GraphDocument
from ...runtime.trainer import iter_losses, train_and_save, train_graph

router = APIRouter()


@router.post("/api/train")
def train_graph_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """Train a self-contained graph (data loader + loss) and return loss history.

    Pass ``save_path`` to also persist the fitted model to disk ("train to
    model"): `.safetensors` saves a safetensors state_dict, otherwise a full
    `torch.save` object is written.
    """
    try:
        doc = GraphDocument.from_dict(body.get("doc", body))
        steps = int(body.get("steps", 20))
        lr = float(body.get("lr", 1e-3))
        wd = float(body.get("wd", 0.0))
        save_path = str(body.get("save_path", "")).strip()
        if save_path:
            losses = train_and_save(doc, save_path, steps=steps, lr=lr, wd=wd)
            return {"status": "success", "losses": losses, "saved": save_path}
        losses = train_graph(doc, steps=steps, lr=lr)
        return {"status": "success", "losses": losses}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/api/train/stream")
def train_stream_endpoint(body: Dict[str, Any]):
    """Stream per-step losses (NDJSON) for the live loss curve."""
    doc = GraphDocument.from_dict(body.get("doc", body))
    steps = int(body.get("steps", 20))
    lr = float(body.get("lr", 1e-3))
    wd = float(body.get("wd", 0.0))

    def gen():
        try:
            for item in iter_losses(doc, steps=steps, lr=lr, wd=wd):
                yield json.dumps(item) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
