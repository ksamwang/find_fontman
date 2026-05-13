from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

from .deps import Image, ImageDraw, ImageFont, cv2, np, require_cv2, require_numpy, require_pillow
from .engine_types import AlignParams, RenderedCandidate
from .preprocess import features_from_mask


def render_text_mask(font_path: str, text: str, canvas_size: tuple[int, int], align: AlignParams) -> RenderedCandidate:
    require_pillow()
    require_numpy()
    require_cv2()
    width, height = canvas_size
    font = load_font(font_path, max(1, align.font_size))
    base = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(base)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2 - bbox[0] + align.offset_x
    y = (height - text_h) // 2 - bbox[1] + align.offset_y
    draw.text((x, y), text, font=font, fill=255)
    arr = np.asarray(base, dtype=np.uint8)
    if align.scale_x != 1.0 or align.scale_y != 1.0:
        scaled_w = max(1, int(width * align.scale_x))
        scaled_h = max(1, int(height * align.scale_y))
        scaled = cv2.resize(arr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        arr = paste_center(scaled, width, height)
    mask = (arr > 32).astype(np.uint8)
    features = features_from_mask(255 - arr, mask)
    image = Image.fromarray(255 - arr).convert("L")
    return RenderedCandidate(image=image, features=features, align=align)


def paste_center(src, width: int, height: int):
    dst = np.zeros((height, width), dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    copy_w = min(width, src_w)
    copy_h = min(height, src_h)
    src_x = max(0, (src_w - copy_w) // 2)
    src_y = max(0, (src_h - copy_h) // 2)
    dst_x = max(0, (width - copy_w) // 2)
    dst_y = max(0, (height - copy_h) // 2)
    dst[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = src[src_y : src_y + copy_h, src_x : src_x + copy_w]
    return dst


def estimate_base_font_size(font_path: str, text: str, target_size: tuple[int, int]) -> int:
    width, height = target_size
    low, high = 4, max(12, height * 4)
    best = low
    while low <= high:
        mid = (low + high) // 2
        try:
            font = load_font(font_path, mid)
            bbox = font.getbbox(text)
        except Exception:
            return 0
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= width * 0.96 and th <= height * 0.92:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


@lru_cache(maxsize=4096)
def load_font(font_path: str, size: int):
    return ImageFont.truetype(font_path, size=size)
