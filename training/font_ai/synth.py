from __future__ import annotations

import random
from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

from .config import IMAGE_HEIGHT, IMAGE_WIDTH
from .transforms import fit_canvas


def render_training_image(
    font_path: str,
    text: str,
    rng: random.Random | None = None,
    style: dict | None = None,
    face_index: int = 0,
) -> Image.Image:
    rng = rng or random.Random()
    style = style or {}
    font_size = int(style.get("font_size") or rng.randint(42, 88))
    font = ImageFont.truetype(font_path, font_size, index=face_index)
    lines = text.split("\n")
    spacing = int(style.get("spacing", rng.randint(-4, 8)))
    line_metrics = [measure_line(font, line, spacing) for line in lines]
    text_w = max(metric["width"] for metric in line_metrics)
    text_h = sum(metric["height"] for metric in line_metrics)
    scale_x = float(style.get("scale_x", 1.0))
    scale_y = float(style.get("scale_y", 1.0))
    width = max(IMAGE_WIDTH, text_w + rng.randint(48, 120))
    height = max(IMAGE_HEIGHT, text_h + rng.randint(24, 72))
    bg = int(style.get("bg") or rng.randint(224, 255))
    fg = int(style.get("fg") or rng.randint(0, 48))
    image = Image.new("RGB", (width, height), (bg, bg, bg))
    draw = ImageDraw.Draw(image)
    align = style.get("align") or rng.choice(["center", "left", "right"])
    line_gap = int(style.get("line_gap") or rng.randint(4, 20))
    base_offset_x = int(style.get("offset_x") or rng.randint(-12, 12))
    base_offset_y = int(style.get("offset_y") or rng.randint(-8, 8))
    shadow_enabled = rng.random() < float(style.get("shadow_prob", 0.25))
    stroke_enabled = rng.random() < float(style.get("stroke_prob", 0.15))
    shadow_offset = (rng.randint(1, 4), rng.randint(1, 4))
    shadow_fill = (max(0, fg - rng.randint(10, 35)),) * 3
    stroke_fill = (min(255, fg + rng.randint(45, 95)),) * 3
    total_text_h = sum(metric["height"] for metric in line_metrics)
    total_text_h += line_gap * max(0, len(lines) - 1)
    cursor_y = (height - total_text_h) // 2 + base_offset_y
    for metric in line_metrics:
        line_w = metric["width"]
        line_h = metric["height"]
        if align == "left":
            x = 24 + base_offset_x
        elif align == "right":
            x = width - 24 - line_w + base_offset_x
        else:
            x = (width - line_w) // 2 + base_offset_x
        if shadow_enabled:
            draw_spaced_text(draw, (x + shadow_offset[0], cursor_y + shadow_offset[1]), metric["chars"], font, fill=shadow_fill, spacing=spacing)
        draw_spaced_text(
            draw,
            (x, cursor_y),
            metric["chars"],
            font,
            fill=(fg, fg, fg),
            spacing=spacing,
            stroke_width=1 if stroke_enabled else 0,
            stroke_fill=stroke_fill,
        )
        cursor_y += line_h + line_gap
    image = stretch_around_center(image, scale_x, scale_y, fill=(bg, bg, bg))
    angle = float(style.get("angle") or rng.uniform(-3.5, 3.5))
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(bg, bg, bg))
    if rng.random() < float(style.get("crop_prob", 0.2)):
        image = edge_crop(image, rng)
    if rng.random() < float(style.get("blur_prob", 0.65)):
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.85)))
    if rng.random() < float(style.get("noise_prob", 0.8)):
        arr = np.asarray(image, dtype=np.int16)
        noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, rng.uniform(1.0, 8.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
    if rng.random() < float(style.get("contrast_prob", 0.45)):
        arr = np.asarray(image, dtype=np.float32)
        arr = np.clip((arr - 128.0) * rng.uniform(0.85, 1.2) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
    if rng.random() < float(style.get("jpeg_prob", 0.35)):
        image = jpeg_roundtrip(image, quality=rng.randint(58, 92))
    return fit_canvas(image)


def measure_line(font, line: str, spacing: int) -> dict:
    chars = list(line) or [" "]
    widths = []
    height = 1
    for ch in chars:
        bbox = font.getbbox(ch)
        widths.append(max(1, bbox[2] - bbox[0]))
        height = max(height, max(1, bbox[3] - bbox[1]))
    width = sum(widths) + spacing * max(0, len(chars) - 1)
    return {"chars": chars, "width": max(1, width), "height": height}


def draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    chars: list[str],
    font,
    fill,
    spacing: int,
    stroke_width: int = 0,
    stroke_fill=None,
) -> None:
    x, y = xy
    for ch in chars:
        bbox = font.getbbox(ch)
        draw.text((x - bbox[0], y - bbox[1]), ch, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        x += max(1, bbox[2] - bbox[0]) + spacing


def stretch_around_center(image: Image.Image, scale_x: float, scale_y: float, fill: tuple[int, int, int]) -> Image.Image:
    if abs(scale_x - 1.0) < 0.015 and abs(scale_y - 1.0) < 0.015:
        return image
    scaled_w = max(1, int(round(image.width * scale_x)))
    scaled_h = max(1, int(round(image.height * scale_y)))
    scaled = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", image.size, fill)
    left = max(0, (scaled_w - image.width) // 2)
    top = max(0, (scaled_h - image.height) // 2)
    right = left + min(image.width, scaled_w)
    bottom = top + min(image.height, scaled_h)
    cropped = scaled.crop((left, top, right, bottom))
    x = max(0, (image.width - cropped.width) // 2)
    y = max(0, (image.height - cropped.height) // 2)
    canvas.paste(cropped, (x, y))
    return canvas


def edge_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    max_x = max(1, int(image.width * 0.025))
    max_y = max(1, int(image.height * 0.06))
    left = rng.randint(0, max_x)
    top = rng.randint(0, max_y)
    right = image.width - rng.randint(0, max_x)
    bottom = image.height - rng.randint(0, max_y)
    if right - left < IMAGE_WIDTH // 2 or bottom - top < IMAGE_HEIGHT // 2:
        return image
    return image.crop((left, top, right, bottom))


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def can_render(font_path: str, text: str, face_index: int = 0) -> bool:
    try:
        font = ImageFont.truetype(font_path, 48, index=face_index)
        bbox = font.getbbox(text)
        return bbox[2] > bbox[0] and bbox[3] > bbox[1]
    except Exception:
        return False
