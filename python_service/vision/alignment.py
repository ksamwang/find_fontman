from __future__ import annotations

from dataclasses import replace

from .engine_types import AlignParams, ImageFeatures, RenderedCandidate, ScoreResult
from .rendering import estimate_base_font_size, render_text_mask
from .scoring import score_features


class AlignmentSearcher:
    def coarse_align(self, font_path: str, text: str, target: ImageFeatures) -> tuple[RenderedCandidate, ScoreResult] | None:
        base = estimate_base_font_size(font_path, text, target.size)
        if base <= 0:
            return None
        sizes = [base]
        offsets = [(0, 0)]
        scales = [(1.0, 1.0)]
        return self.search(font_path, text, target, sizes, offsets, scales)

    def fine_align(self, font_path: str, text: str, target: ImageFeatures, seed: AlignParams | None = None) -> tuple[RenderedCandidate, ScoreResult] | None:
        base = seed.font_size if seed else estimate_base_font_size(font_path, text, target.size)
        if base <= 0:
            return None
        sizes = sorted({max(4, base + delta) for delta in (-4, -2, 0, 2, 4)})
        offsets = self.offset_grid(target, coarse=False, seed=seed)
        scales = [(1.0, 1.0), (0.94, 1.0), (0.98, 1.0), (1.02, 1.0), (1.06, 1.0), (1.0, 0.94), (1.0, 1.06)]
        return self.search(font_path, text, target, sizes, offsets, scales)

    def search(
        self,
        font_path: str,
        text: str,
        target: ImageFeatures,
        sizes: list[int],
        offsets: list[tuple[int, int]],
        scales: list[tuple[float, float]],
    ) -> tuple[RenderedCandidate, ScoreResult] | None:
        best: tuple[RenderedCandidate, ScoreResult] | None = None
        for size in sizes:
            for scale_x, scale_y in scales:
                for offset_x, offset_y in offsets:
                    align = AlignParams(size, scale_x, scale_y, offset_x, offset_y)
                    rendered = render_text_mask(font_path, text, target.size, align)
                    score = score_features(target, rendered.features, rendered.align)
                    if best is None or score.score_total > best[1].score_total:
                        best = (rendered, score)
        return best

    def offset_grid(self, target: ImageFeatures, coarse: bool, seed: AlignParams | None = None) -> list[tuple[int, int]]:
        width, height = target.size
        if coarse:
            step = max(2, min(width, height) // 16)
            values = [-step, 0, step]
            return [(x, y) for x in values for y in values]
        step = max(1, min(width, height) // 45)
        center_x = seed.offset_x if seed else 0
        center_y = seed.offset_y if seed else 0
        values = [-2 * step, -step, 0, step, 2 * step]
        return [(center_x + x, center_y + y) for x in values for y in values]
