from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import Settings
from .deps import Image, PaddleOCR, require_pillow
from .font_index import FontIndex
from .font_index import FontRecord
from .matcher import FontMatcher
from .ocr import OCR
from python_service.font_ai.infer import FontEmbeddingMatcher


class VisionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.crop_dir = settings.data / "crops"
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        settings.previews.mkdir(parents=True, exist_ok=True)
        self.index = FontIndex(settings.fonts, settings.data / "font_index.sqlite")
        self.index.ensure()
        self.ocr = OCR()
        self.matcher = FontMatcher(self.index, settings.previews)
        self.ai_matcher = FontEmbeddingMatcher(settings.root, settings.fonts, settings.data)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pillow": Image is not None,
            "numpy": self._numpy_available(),
            "paddleocr": PaddleOCR is not None,
            "font_count": self.index.count(),
            "match_mode": "embedding" if self.ai_matcher.available else "renderer",
            "capabilities": {
                "match_tasks": True,
                "preview_base64": True,
                "embedding_match": self.ai_matcher.available,
            },
        }

    def analyze_crop(self, payload: dict[str, Any]) -> dict[str, Any]:
        require_pillow()
        crop = self.crop(Path(payload["image_path"]), payload["box"])
        crop_path = self.crop_dir / f"crop_{int(time.time() * 1000)}.png"
        crop.save(crop_path)

        if not self.ocr.available():
            return {
                "text": "",
                "confidence": 0.0,
                "crop_url": str(crop_path),
                "warning": "PaddleOCR is not installed. Crop was saved; enter text manually or install dependencies.",
            }

        text, confidence = self.ocr.recognize(crop_path)
        return {"text": text, "confidence": confidence, "crop_url": str(crop_path)}

    def match_fonts(self, payload: dict[str, Any], progress=None) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        top_k = int(payload.get("top_k") or 10)
        crop = self.crop(Path(payload["image_path"]), payload["box"])
        rerank = bool(payload.get("rerank")) or str(payload.get("rerank", "")).lower() == "true"
        if self.ai_matcher.available and not rerank:
            started = time.time()
            result = self.ai_matcher.match(crop, text=text, top_k=top_k)
            result["elapsed_ms"] = int((time.time() - started) * 1000)
            return result
        if self.ai_matcher.available and rerank:
            started = time.time()
            ai_result = self.ai_matcher.match(crop, text=text, top_k=top_k, top_n=100)
            allowed_scopes = {"zh_simplified", "zh_traditional"} if any("\u4e00" <= ch <= "\u9fff" for ch in text) else {"english"}
            top_indices = ai_result.get("top_indices", [])
            records = [
                FontRecord(
                    path=self.ai_matcher.records[idx]["path"],
                    name=self.ai_matcher.records[idx]["name"],
                    category=self.ai_matcher.records[idx].get("category", ""),
                    mtime=0,
                    size=0,
                )
                for idx in top_indices
                if self.ai_matcher.records[idx].get("category", "unknown") in allowed_scopes
            ]
            if not records:
                records = [
                    FontRecord(
                        path=self.ai_matcher.records[idx]["path"],
                        name=self.ai_matcher.records[idx]["name"],
                        category=self.ai_matcher.records[idx].get("category", ""),
                        mtime=0,
                        size=0,
                    )
                    for idx in top_indices
                ]
            result = self.matcher.match(crop, text, top_k, progress=progress, records_override=records)
            result["match_mode"] = "embedding_rerank"
            result["elapsed_ms"] = int((time.time() - started) * 1000)
            return result
        return self.matcher.match(crop, text, top_k, progress=progress)

    def crop(self, image_path: Path, raw_box: dict[str, Any]) -> Any:
        require_pillow()
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            x = max(0, int(raw_box.get("x", 0)))
            y = max(0, int(raw_box.get("y", 0)))
            w = max(1, int(raw_box.get("w", 1)))
            h = max(1, int(raw_box.get("h", 1)))
            right = min(img.width, x + w)
            bottom = min(img.height, y + h)
            return img.crop((x, y, right, bottom))

    def _numpy_available(self) -> bool:
        try:
            from .deps import np

            return np is not None
        except Exception:
            return False
