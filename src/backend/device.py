"""Device detection (TORCH_BACKEND.md, CROSS_PLATFORM.md).

CPU is the guaranteed baseline. CUDA / MPS may only be used after checking
availability. ``auto`` falls back safely to CPU.
"""
from __future__ import annotations

from typing import Any

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    return TORCH_AVAILABLE


def _mps_available() -> bool:
    return bool(
        TORCH_AVAILABLE
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_device(device: Any = "cpu"):
    """Resolve a device string to a torch.device (or 'cpu' without torch).

    - ``cpu``: always available.
    - ``cuda`` / ``mps``: only when available; explicit request that is
      unavailable raises RuntimeError.
    - ``auto``: cuda -> mps -> cpu fallback.
    """
    if not TORCH_AVAILABLE:
        name = str(device).lower()
        if name in ("cpu", "auto"):
            return "cpu"
        raise RuntimeError(
            f"what: torch 不可用，无法使用设备 '{device}'。\n"
            f"where: backend.device.resolve_device\n"
            f"how_to_fix: 安装 torch，或使用 device='cpu'。"
        )

    if isinstance(device, torch.device):
        return device
    name = str(device).lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "what: 请求 CUDA 但不可用。\n"
                "where: backend.device.resolve_device\n"
                "how_to_fix: 安装 CUDA 版 torch，或用 device='cpu'/'auto'。"
            )
        return torch.device("cuda")

    if name == "mps":
        if not _mps_available():
            raise RuntimeError(
                "what: 请求 MPS 但当前环境不支持。\n"
                "where: backend.device.resolve_device\n"
                "how_to_fix: 改用 device='cpu'/'auto'。"
            )
        return torch.device("mps")

    if name == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"what: 未知设备 '{device}'。\n"
        f"where: backend.device.resolve_device\n"
        f"how_to_fix: 使用 cpu / cuda / mps / auto。"
    )
