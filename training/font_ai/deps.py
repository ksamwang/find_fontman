from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = nn = F = DataLoader = Dataset = None


def available() -> bool:
    return torch is not None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Run training/scripts/train_full_windows_gpu.ps1.")
