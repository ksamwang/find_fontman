from __future__ import annotations

from typing import Any

from .deps import Image, cv2, np, require_cv2, require_numpy, require_pillow
from .engine_types import ImageFeatures
from .image_ops import bbox_from_mask


def extract_target_features(image: Any) -> ImageFeatures:
    require_pillow()
    require_numpy()
    require_cv2()
    gray = pil_to_gray(image)
    gray = normalize_gray(gray)
    mask = foreground_mask(gray)
    mask = cleanup_mask(mask)
    return features_from_mask(gray, mask)


def pil_to_gray(image: Any):
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def normalize_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def foreground_mask(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = np.count_nonzero(otsu_dark) / otsu_dark.size
    light_ratio = np.count_nonzero(otsu_light) / otsu_light.size
    chosen = otsu_dark if 0.005 <= dark_ratio <= 0.55 else otsu_light
    if not 0.005 <= (np.count_nonzero(chosen) / chosen.size) <= 0.65:
        chosen = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
    return (chosen > 0).astype(np.uint8)


def cleanup_mask(mask):
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned.astype(np.uint8)


def features_from_mask(gray, mask) -> ImageFeatures:
    mask = mask.astype(np.uint8)
    edge = cv2.Canny((mask * 255).astype(np.uint8), 80, 160)
    edge = (edge > 0).astype(np.uint8)
    inverse = (1 - edge).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        aspect = 0.0
        density = 0.0
        stroke = 0.0
    else:
        _, _, w, h = bbox
        aspect = w / max(1, h)
        density = float(mask.sum() / max(1, w * h))
        stroke = estimate_stroke(mask)
    return ImageFeatures(
        image=gray,
        mask=mask,
        edge=edge,
        distance=distance,
        bbox=bbox,
        aspect=aspect,
        density=density,
        stroke=stroke,
        size=(int(gray.shape[1]), int(gray.shape[0])),
    )


def estimate_stroke(mask) -> float:
    if mask.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = dist[mask > 0]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 75) * 2.0)
