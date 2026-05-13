from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

from .deps import ImageFont, require_cv2, require_numpy, require_pillow
from .engine import FontMatchEngine
from .engine_types import AlignParams
from .font_index import FontIndex, FontRecord


class FontMatcher:
    def __init__(self, index: FontIndex, previews_dir: Path) -> None:
        self.index = index
        self.previews_dir = previews_dir
        self.max_candidates = int(os.getenv("FONTMAN_MAX_CANDIDATES", "0"))
        self.fine_candidates = int(os.getenv("FONTMAN_FINE_CANDIDATES", "0"))
        self.max_workers = int(os.getenv("FONTMAN_MATCH_WORKERS", str(min(8, os.cpu_count() or 4))))
        self.engine = FontMatchEngine(max_workers=self.max_workers)
        self._can_render_cache: dict[tuple[str, str], bool] = {}

    def match(
        self,
        crop: Any,
        text: str,
        top_k: int,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        require_pillow()
        require_numpy()
        require_cv2()
        started = time.time()
        target = self.engine.prepare_target(crop)
        candidates = self.filter_candidates(text)
        self.emit(progress, "candidates", 0, len(candidates), f"{len(candidates)} candidates selected")

        coarse = self.engine.coarse_rank(candidates, target, text, progress=progress)
        fine_records = [item["record"] for item in coarse[: self.fine_candidates]] if self.fine_candidates > 0 else [item["record"] for item in coarse]
        fine = self.engine.fine_rank(fine_records, target, text, progress=progress)

        results = [self.to_result(item, text) for item in fine[:top_k]]
        return {
            "results": results,
            "candidate_size": len(candidates),
            "elapsed_ms": int((time.time() - started) * 1000),
            "warning": "" if fine else "No scoreable fonts found. Check text, fonts, and image dependencies.",
        }

    def filter_candidates(self, text: str) -> list[FontRecord]:
        kind = classify_text(text)
        records = self.index.records()
        if kind == "english":
            records = [r for r in records if "\u82f1\u6587" in r.category]
        elif kind == "cjk":
            records = [r for r in records if "\u4e2d\u6587" in r.category]
        filtered: list[FontRecord] = []
        for record in records:
            cache_key = (record.path, text)
            if cache_key in self._can_render_cache:
                can_render = self._can_render_cache[cache_key]
                cached = can_render
            else:
                cached = self.index.cached_can_render(record, text)
                can_render = cached if cached is not None else self.font_can_render(record.path, text)
                self._can_render_cache[cache_key] = can_render
            if cached is None:
                self.index.save_can_render(record, text, can_render)
            if can_render:
                filtered.append(record)
            if self.max_candidates > 0 and len(filtered) >= self.max_candidates:
                break
        return filtered

    def font_can_render(self, font_path: str, text: str) -> bool:
        try:
            font = ImageFont.truetype(font_path, size=32)
            bbox = font.getbbox(text)
            return bbox[2] > bbox[0] and bbox[3] > bbox[1]
        except Exception:
            return False

    def to_result(self, item: dict[str, Any], text: str) -> dict[str, Any]:
        rec = item["record"]
        rendered = item["rendered"]
        score = item["score"]
        preview_path = self.save_preview(rendered.image, rec, text, score.score_total)
        return {
            "font_name": rec.name,
            "font_path": rec.path,
            "score_total": round(score.score_total, 6),
            "score_ssim": round(score.score_ssim, 6),
            "score_iou": round(score.score_iou, 6),
            "score_edge": round(score.score_edge, 6),
            "score_shape": round(score.score_shape, 6),
            "score_chamfer": round(score.score_chamfer, 6),
            "score_density": round(score.score_density, 6),
            "align": align_to_dict(score.align),
            "preview_path": str(preview_path),
            "preview_base64": self.preview_base64(preview_path),
            "preview_mime": "image/png",
        }

    def save_preview(self, image: Any, rec: FontRecord, text: str, score: float) -> Path:
        digest = hashlib.sha1(f"{rec.path}|{text}|{score}".encode("utf-8")).hexdigest()[:16]
        path = self.previews_dir / f"{digest}.png"
        image.convert("RGB").save(path)
        return path

    def preview_base64(self, path: Path) -> str:
        with path.open("rb") as file:
            return base64.b64encode(file.read()).decode("ascii")

    def emit(
        self,
        progress: Callable[[dict[str, Any]], None] | None,
        phase: str,
        done: int,
        total: int,
        message: str,
    ) -> None:
        if progress is None:
            return
        progress({"phase": phase, "done": done, "total": total, "message": message})


def align_to_dict(align: AlignParams) -> dict[str, Any]:
    return {
        "font_size": align.font_size,
        "scale_x": round(align.scale_x, 4),
        "scale_y": round(align.scale_y, 4),
        "offset_x": align.offset_x,
        "offset_y": align.offset_y,
    }


def classify_text(text: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "cjk"
    return "english"
