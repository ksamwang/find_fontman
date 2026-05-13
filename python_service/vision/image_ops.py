from __future__ import annotations

import math
from typing import Any

from .deps import ImageOps, np


def prepare_target(crop: Any) -> dict[str, Any]:
    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray)
    mask = image_to_mask(gray)
    return {"image": gray, "mask": mask, "edge": mask_to_edge(mask), "size": gray.size}


def image_to_mask(gray: Any):
    arr = np.asarray(gray, dtype=np.uint8)
    threshold = int(np.percentile(arr, 55))
    dark = arr <= threshold
    light = arr >= threshold
    if dark.mean() > 0.5:
        dark = light
    return dark.astype(np.uint8)


def mask_to_edge(mask):
    padded = np.pad(mask, 1)
    center = padded[1:-1, 1:-1]
    edge = (
        (center != padded[:-2, 1:-1])
        | (center != padded[2:, 1:-1])
        | (center != padded[1:-1, :-2])
        | (center != padded[1:-1, 2:])
    )
    return edge.astype(np.uint8)


def mask_iou(a, b) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / union)


def simple_ssim(a, b) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    ux, uy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - ux) * (y - uy)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    denom = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    if denom == 0:
        return 0.0
    return float(((2 * ux * uy + c1) * (2 * cov + c2)) / denom)


def shape_score(a, b) -> float:
    ab = bbox_from_mask(a)
    bb = bbox_from_mask(b)
    if ab is None or bb is None:
        return 0.0
    ar = ab[2] / max(1, ab[3])
    br = bb[2] / max(1, bb[3])
    ratio_score = 1.0 - min(1.0, abs(math.log((ar + 1e-6) / (br + 1e-6))))
    area_a = ab[2] * ab[3]
    area_b = bb[2] * bb[3]
    area_score = 1.0 - min(1.0, abs(math.log((area_a + 1) / (area_b + 1))))
    return clamp01(0.6 * ratio_score + 0.4 * area_score)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))
