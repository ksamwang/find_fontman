from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignParams:
    font_size: int
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int


@dataclass
class ImageFeatures:
    image: Any
    mask: Any
    edge: Any
    distance: Any
    bbox: tuple[int, int, int, int] | None
    aspect: float
    density: float
    stroke: float
    size: tuple[int, int]


@dataclass
class RenderedCandidate:
    image: Any
    features: ImageFeatures
    align: AlignParams


@dataclass
class ScoreResult:
    score_total: float
    score_ssim: float
    score_iou: float
    score_edge: float
    score_shape: float
    score_chamfer: float
    score_density: float
    align: AlignParams
