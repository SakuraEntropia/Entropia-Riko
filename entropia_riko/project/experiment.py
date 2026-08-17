"""Experiment records for reproducibility.

Every execution can optionally be recorded as an ``experiments/experiment_NNN/``
directory capturing the workflow, parameters, metrics, outputs, and hardware
info — the goal is that a past run can be inspected and reproduced.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def hardware_info() -> Dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        info["device_count"] = str(torch.cuda.device_count())
    except Exception:
        pass
    return info


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_experiment_dir(project_root: Path) -> Path:
    """Return the next ``experiments/experiment_NNN`` directory path."""
    base = project_root / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("experiment_")
    ]
    n = len(existing) + 1
    return base / f"experiment_{n:03d}"


def record_experiment(
    project_root: Path,
    workflow: Dict[str, Any],
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: Optional[List[Path]] = None,
    seed: Optional[int] = None,
) -> Path:
    """Create and populate an experiment directory; return its path."""
    exp = next_experiment_dir(project_root)
    (exp / "outputs").mkdir(parents=True, exist_ok=True)
    (exp / "logs").mkdir(exist_ok=True)

    (exp / "workflow.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (exp / "parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (exp / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (exp / "metadata.json").write_text(
        json.dumps(
            {
                "created": _now(),
                "seed": seed,
                "hardware": hardware_info(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if artifacts:
        for src in artifacts:
            if src.is_file():
                (exp / "outputs" / src.name).write_bytes(src.read_bytes())
    return exp


def list_experiments(project_root: Path) -> List[Dict[str, Any]]:
    """List experiment records with their metrics summaries."""
    base = project_root / "experiments"
    if not base.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("experiment_"):
            continue
        entry: Dict[str, Any] = {"name": d.name, "path": d.name}
        metrics_path = d / "metrics.json"
        if metrics_path.is_file():
            try:
                entry["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                entry["metrics"] = {}
        out.append(entry)
    return out
