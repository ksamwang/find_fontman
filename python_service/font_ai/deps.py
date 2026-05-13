from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover
    torch = nn = F = Dataset = DataLoader = None


def available() -> bool:
    return torch is not None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch/torchvision to use font_ai.")
