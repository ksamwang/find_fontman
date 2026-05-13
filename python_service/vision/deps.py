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
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

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


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install python_service/requirements.txt.")
