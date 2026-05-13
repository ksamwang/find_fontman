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
            self._engine = PaddleOCR(lang="ch", use_textline_orientation=True)

        result = self._engine.ocr(str(image_path), cls=True)
        best_text = ""
        best_conf = 0.0
        for page in result or []:
            for line in page or []:
                if len(line) >= 2 and line[1]:
                    text, conf = line[1][0], float(line[1][1])
                    if conf > best_conf:
                        best_text, best_conf = text, conf
        return best_text, best_conf
