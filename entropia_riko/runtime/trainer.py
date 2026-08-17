"""Training runner for self-contained graphs (data loader + loss).

Reuses codegen to build a real ``nn.Module`` (no ``no_grad``), then runs
optimizer steps. ``iter_losses`` yields loss per step (for streaming / the
live loss curve); ``train_graph`` returns the full history at once.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..core.document import GraphDocument
from .codegen import export_python
from .registry import Registry, default_registry


def _build_model(doc: GraphDocument, registry: Optional[Registry]):
    """Generate + exec the module; raise if the graph is not trainable."""
    reg = registry if registry is not None else default_registry()
    code = export_python(doc, reg)
    namespace: Dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 - generated module under our control

    if "train" not in namespace or "GraphModel" not in namespace:
        raise ValueError(
            "what: 图不可训练。\n"
            "where: runtime.trainer._build_model\n"
            "how_to_fix: 需要 loss 节点（mse_loss / cross_entropy_loss 等）且"
            "数据由数据加载节点提供（不含 graph_input）。"
        )
    return namespace["GraphModel"]()


def iter_losses(
    doc: GraphDocument,
    steps: int = 20,
    lr: float = 1e-3,
    wd: float = 0.0,
    registry: Optional[Registry] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield ``{"step": i, "loss": float}`` per optimizer step (for streaming)."""
    import torch.optim as optim

    model = _build_model(doc, registry)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    for step in range(int(steps)):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        yield {"step": step, "loss": float(loss.item())}


def train_graph(
    doc: GraphDocument,
    steps: int = 20,
    lr: float = 1e-3,
    registry: Optional[Registry] = None,
) -> List[float]:
    """Run `steps` optimizer steps; return the loss history."""
    return [item["loss"] for item in iter_losses(doc, steps=steps, lr=lr, registry=registry)]


def train_and_save(
    doc: GraphDocument,
    save_path: str,
    steps: int = 20,
    lr: float = 1e-3,
    wd: float = 0.0,
    registry: Optional[Registry] = None,
) -> List[float]:
    """Train a self-contained graph, then save the fitted model to `save_path`.

    The format is chosen by the file extension (`.safetensors` → safetensors
    state_dict; otherwise a full `torch.save` object). Returns the loss history.
    """
    import torch.optim as optim

    from ..nodes.torch_ops.model_loader import _save_model

    model = _build_model(doc, registry)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))
    losses: List[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    _save_model(model, save_path, "auto")
    return losses
