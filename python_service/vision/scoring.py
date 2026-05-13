from __future__ import annotations

import math

from .deps import cv2, np, require_cv2, require_numpy
from .engine_types import AlignParams, ImageFeatures, ScoreResult
from .image_ops import clamp01, mask_iou, shape_score, simple_ssim


def score_features(target: ImageFeatures, candidate: ImageFeatures, align: AlignParams) -> ScoreResult:
    require_numpy()
    require_cv2()
    iou = mask_iou(target.mask, candidate.mask)
    edge = mask_iou(target.edge, candidate.edge)
    ssim = clamp01((simple_ssim(target.mask, candidate.mask) + 1.0) / 2.0)
    shape = shape_score(target.mask, candidate.mask)
    chamfer = chamfer_score(target, candidate)
    density = density_score(target, candidate)
    total = (
        0.30 * chamfer
        + 0.22 * iou
        + 0.18 * edge
        + 0.14 * ssim
        + 0.10 * shape
        + 0.06 * density
    )
    return ScoreResult(
        score_total=clamp01(total),
        score_ssim=ssim,
        score_iou=clamp01(iou),
        score_edge=clamp01(edge),
        score_shape=clamp01(shape),
        score_chamfer=clamp01(chamfer),
        score_density=clamp01(density),
        align=align,
    )


def chamfer_score(target: ImageFeatures, candidate: ImageFeatures) -> float:
    target_edge = target.edge.astype(bool)
    cand_edge = candidate.edge.astype(bool)
    if target_edge.sum() == 0 or cand_edge.sum() == 0:
        return 0.0
    target_to_candidate = float(candidate.distance[target_edge].mean())
    candidate_to_target = float(target.distance[cand_edge].mean())
    norm = max(4.0, min(target.size) * 0.06)
    distance = (target_to_candidate + candidate_to_target) / 2.0
    return math.exp(-distance / norm)


def density_score(target: ImageFeatures, candidate: ImageFeatures) -> float:
    density_delta = abs(target.density - candidate.density)
    stroke_delta = abs(target.stroke - candidate.stroke) / max(1.0, target.stroke, candidate.stroke)
    return clamp01(1.0 - 0.75 * density_delta - 0.25 * stroke_delta)
