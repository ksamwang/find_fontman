from __future__ import annotations

from PIL import Image
import numpy as np

from .config import IMAGE_HEIGHT, IMAGE_WIDTH


def fit_canvas(image: Image.Image, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def image_to_tensor(image: Image.Image):
    from .deps import torch

    image = fit_canvas(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)
