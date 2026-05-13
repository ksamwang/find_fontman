from __future__ import annotations

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
except Exception:  # pragma: no cover - used when env is incomplete
    Image = ImageDraw = ImageFilter = ImageFont = ImageOps = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover
    PaddleOCR = None


def require_pillow() -> None:
    if Image is None or ImageFont is None:
        raise RuntimeError("Pillow is not installed. Install python_service/requirements.txt.")


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is not installed. Install python_service/requirements.txt.")
