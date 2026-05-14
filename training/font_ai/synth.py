from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

from .config import IMAGE_HEIGHT, IMAGE_WIDTH
from .transforms import fit_canvas


def render_training_image(font_path: str, text: str, rng: random.Random | None = None) -> Image.Image:
    rng = rng or random.Random()
    font_size = rng.randint(42, 88)
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    width = max(IMAGE_WIDTH, text_w + rng.randint(48, 120))
    height = max(IMAGE_HEIGHT, text_h + rng.randint(24, 72))
    bg = rng.randint(224, 255)
    fg = rng.randint(0, 48)
    image = Image.new("RGB", (width, height), (bg, bg, bg))
    draw = ImageDraw.Draw(image)
    x = (width - text_w) // 2 - bbox[0] + rng.randint(-12, 12)
    y = (height - text_h) // 2 - bbox[1] + rng.randint(-8, 8)
    draw.text((x, y), text, font=font, fill=(fg, fg, fg))
    angle = rng.uniform(-2.0, 2.0)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(bg, bg, bg))
    if rng.random() < 0.65:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.55)))
    if rng.random() < 0.8:
        arr = np.asarray(image, dtype=np.int16)
        noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, rng.uniform(1.0, 5.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
    return fit_canvas(image)


def can_render(font_path: str, text: str) -> bool:
    try:
        font = ImageFont.truetype(font_path, 48)
        bbox = font.getbbox(text)
        return bbox[2] > bbox[0] and bbox[3] > bbox[1]
    except Exception:
        return False
