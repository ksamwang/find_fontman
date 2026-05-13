from __future__ import annotations

from pathlib import Path
from typing import Any

from .deps import PaddleOCR


class OCR:
    def __init__(self) -> None:
        self._engine: Any | None = None

    def available(self) -> bool:
        return PaddleOCR is not None

    def recognize(self, image_path: Path) -> tuple[str, float]:
        if PaddleOCR is None:
            return "", 0.0
        if self._engine is None:
            self._engine = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )

        result = self._engine.predict(str(image_path))
        return best_ocr_text(result)


def best_ocr_text(result: Any) -> tuple[str, float]:
        best_text = ""
        best_conf = 0.0
        for page in result or []:
            page_data = dict(page) if not isinstance(page, dict) else page

            texts = page_data.get("rec_texts") or []
            scores = page_data.get("rec_scores") or []
            for text, conf in zip(texts, scores):
                if float(conf) > best_conf:
                    best_text, best_conf = str(text), float(conf)

            # Compatibility with PaddleOCR 2.x style nested output.
            for line in page if isinstance(page, list) else []:
                if len(line) >= 2 and line[1]:
                    text, conf = line[1][0], float(line[1][1])
                    if conf > best_conf:
                        best_text, best_conf = text, conf
        return best_text, best_conf
