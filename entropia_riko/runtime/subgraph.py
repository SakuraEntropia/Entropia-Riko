"""Subgraph loading & execution shared by graph_reference / import nodes.

Implements Python-like module resolution for ``.riko`` graph files::

    import xx          # resolves ``xx.riko`` under the module search path
    graph_reference    # file=path/to/xx.riko (explicit path)

Both ultimately execute the referenced graph with a single tensor input
(``graph_input`` name='input') and read a single output (``graph_output``
name='output').
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.document import GraphDocument
from .executor import execute as exec_graph
from .registry import Registry

# Project root (the directory that contains entropia_riko/, examples/, workflows/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories searched (in order) when a bare module name is imported.
MODULE_SEARCH_PATHS: List[Path] = [
    PROJECT_ROOT / "workflows",
    PROJECT_ROOT / "examples",
    PROJECT_ROOT / "examples" / "models",
]


def resolve_graph_file(spec: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a module name or ``.riko`` path to an existing file.

    Resolution order:

    - Absolute path -> used directly.
    - Relative path (with separators or a ``.riko`` suffix) -> relative to
      ``base_dir``, then the project root.
    - Bare module name -> ``<name>.riko`` under ``MODULE_SEARCH_PATHS``.
    """
    if spec is None or spec == "":
        return None
    s = str(spec).replace("\\", "/")
    p = Path(spec)

    candidates: List[Path] = []
    bases: List[Path] = ([base_dir] if base_dir else []) + [PROJECT_ROOT]

    if p.is_absolute():
        candidates.append(p)
    else:
        for b in bases:
            candidates.append(b / s)
            if p.suffix != ".riko":
                candidates.append(b / f"{s}.riko")
        if "/" not in s and p.suffix != ".riko":
            for sp in MODULE_SEARCH_PATHS:
                candidates.append(sp / f"{s}.riko")

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return None


def load_graph_file(path: Path) -> GraphDocument:
    """Load a ``.riko`` file into a GraphDocument."""
    return GraphDocument.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_subgraph(
    spec: Any,
    inputs: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    registry: Optional[Registry] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute a referenced ``.riko`` graph with named tensor inputs.

    ``inputs`` maps ``graph_input`` names to values (e.g. ``{"input": x,
    "input_2": y}``). Returns every ``graph_output`` value keyed by name
    (e.g. ``{"output": out, "output_2": out2}``).
    """
    path = resolve_graph_file(spec, base_dir=base_dir)
    if path is None:
        raise ValueError(
            f"what: 无法解析引用的图文件 '{spec}'。\n"
            f"where: runtime.subgraph.run_subgraph\n"
            f"how_to_fix: 确认文件存在于 "
            f"{[str(p.relative_to(PROJECT_ROOT)) for p in MODULE_SEARCH_PATHS]}，"
            f"或提供正确的相对/绝对路径。"
        )

    doc = load_graph_file(path)
    ctx = dict(context or {})
    # Pass every provided input through to the graph_input nodes by name.
    ctx["graph_inputs"] = {k: v for k, v in (inputs or {}).items()}
    ctx["graph_outputs"] = {}
    exec_graph(doc, registry=registry, context=ctx)

    # Return all graph_output values keyed by name (multi-output subgraphs).
    return dict(ctx["graph_outputs"])
