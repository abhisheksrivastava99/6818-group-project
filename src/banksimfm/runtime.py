"""Runtime and device helpers."""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Prefer Apple Metal on Apple Silicon, then CUDA, then CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
