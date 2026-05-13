from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from .deps import Image, ImageDraw, ImageFilter, ImageFont, require_numpy, require_pillow
from .font_index import FontIndex, FontRecord
from .image_ops import clamp01, image_to_mask, mask_iou, mask_to_edge, prepare_target, shape_score, simple_ssim


class FontMatcher:
    def __init__(self, index: FontIndex, previews_dir: Path) -> None:
        self.index = index
        self.previews_dir = previews_dir
        self.max_candidates = int(os.getenv("FONTMAN_MAX_CANDIDATES", "200"))

    def match(self, crop: Any, text: str, top_k: int) -> dict[str, Any]:
        require_pillow()
        require_numpy()
        started = time.time()
        target = prepare_target(crop)
        candidates = self.filter_candidates(text)

        coarse: list[dict[str, Any]] = []
        for rec in candidates:
            try:
                rendered = self.render_for_target(rec.path, text, target["size"], fast=True)
                coarse.append({"record": rec, **self.score(target, rendered)})
            except Exception:
                continue
        coarse.sort(key=lambda item: item["score_total"], reverse=True)

        fine: list[dict[str, Any]] = []
        for item in coarse[:100]:
            rec = item["record"]
            try:
                rendered = self.render_for_target(rec.path, text, target["size"], fast=False)
                scores = self.score(target, rendered)
                preview_path = self.save_preview(rendered["image"], rec, text, scores["score_total"])
                fine.append({"record": rec, "preview_path": str(preview_path), **scores})
            except Exception:
                continue
        fine.sort(key=lambda item: item["score_total"], reverse=True)

        return {
            "results": [self.to_result(item) for item in fine[:top_k]],
            "candidate_size": len(candidates),
            "elapsed_ms": int((time.time() - started) * 1000),
            "warning": "" if fine else "No scoreable fonts found. Check text, fonts, and image dependencies.",
        }

    def filter_candidates(self, text: str) -> list[FontRecord]:
        kind = classify_text(text)
        records = self.index.records()
        if kind == "english":
            records = [r for r in records if "英文" in r.category]
        elif kind == "cjk":
            records = [r for r in records if "中文" in r.category]
        filtered: list[FontRecord] = []
        for record in records:
            cached = self.index.cached_can_render(record, text)
            can_render = cached if cached is not None else self.font_can_render(record.path, text)
            if cached is None:
                self.index.save_can_render(record, text, can_render)
            if can_render:
                filtered.append(record)
            if len(filtered) >= self.max_candidates:
                break
        return filtered

    def font_can_render(self, font_path: str, text: str) -> bool:
        try:
            font = ImageFont.truetype(font_path, size=32)
            bbox = font.getbbox(text)
            return bbox[2] > bbox[0] and bbox[3] > bbox[1]
        except Exception:
            return False

    def render_for_target(self, font_path: str, text: str, size: tuple[int, int], fast: bool) -> dict[str, Any]:
        width, height = size
        font_size = self.fit_font_size(font_path, text, width, height, fast)
        font = ImageFont.truetype(font_path, size=font_size)
        canvas = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = max(0, (width - text_w) // 2 - bbox[0])
        y = max(0, (height - text_h) // 2 - bbox[1])
        draw.text((x, y), text, font=font, fill=0)
        if not fast:
            canvas = canvas.filter(ImageFilter.SHARPEN)
        mask = image_to_mask(canvas)
        return {"image": canvas, "mask": mask, "edge": mask_to_edge(mask), "size": canvas.size}

    def fit_font_size(self, font_path: str, text: str, width: int, height: int, fast: bool) -> int:
        low, high = 4, max(8, height * 3)
        best = low
        while low <= high:
            mid = (low + high) // 2
            font = ImageFont.truetype(font_path, size=mid)
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= width * 0.96 and th <= height * 0.92:
                best = mid
                low = mid + (2 if fast else 1)
            else:
                high = mid - 1
        return best

    def score(self, target: dict[str, Any], rendered: dict[str, Any]) -> dict[str, float]:
        iou = mask_iou(target["mask"], rendered["mask"])
        edge = mask_iou(target["edge"], rendered["edge"])
        ssim = simple_ssim(target["mask"], rendered["mask"])
        shape = shape_score(target["mask"], rendered["mask"])
        total = 0.35 * ssim + 0.30 * iou + 0.20 * edge + 0.15 * shape
        return {
            "score_total": clamp01(total),
            "score_ssim": clamp01(ssim),
            "score_iou": clamp01(iou),
            "score_edge": clamp01(edge),
            "score_shape": clamp01(shape),
        }

    def save_preview(self, image: Any, rec: FontRecord, text: str, score: float) -> Path:
        digest = hashlib.sha1(f"{rec.path}|{text}|{score}".encode("utf-8")).hexdigest()[:16]
        path = self.previews_dir / f"{digest}.png"
        image.convert("RGB").save(path)
        return path

    def to_result(self, item: dict[str, Any]) -> dict[str, Any]:
        rec = item["record"]
        return {
            "font_name": rec.name,
            "font_path": rec.path,
            "score_total": round(item["score_total"], 6),
            "score_ssim": round(item["score_ssim"], 6),
            "score_iou": round(item["score_iou"], 6),
            "score_edge": round(item["score_edge"], 6),
            "score_shape": round(item["score_shape"], 6),
            "preview_path": item["preview_path"],
        }


def classify_text(text: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "cjk"
    return "english"
