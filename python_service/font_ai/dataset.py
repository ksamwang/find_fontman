from __future__ import annotations

import json
import random
from pathlib import Path

from .config import DEFAULT_TEXTS
from .deps import Dataset, require_torch
from .synth import can_render, render_training_image
from .transforms import image_to_tensor


BaseDataset = Dataset if Dataset is not None else object


class FontRenderDataset(BaseDataset):
    def __init__(
        self,
        font_records: list[dict],
        texts: list[str],
        samples_per_font: int,
        seed: int = 7,
    ) -> None:
        require_torch()
        self.font_records = font_records
        self.texts = texts or DEFAULT_TEXTS
        self.samples_per_font = samples_per_font
        self.seed = seed

    def __len__(self) -> int:
        return len(self.font_records) * self.samples_per_font

    def __getitem__(self, idx: int):
        font_idx = idx // self.samples_per_font
        sample_idx = idx % self.samples_per_font
        record = self.font_records[font_idx]
        rng = random.Random(self.seed + idx * 9973)
        text = self.texts[(sample_idx + rng.randint(0, len(self.texts) - 1)) % len(self.texts)]
        image = render_training_image(record["path"], text, rng)
        return image_to_tensor(image), font_idx


def scan_font_records(fonts_dir: Path, limit: int = 0, texts: list[str] | None = None) -> list[dict]:
    texts = texts or DEFAULT_TEXTS
    records = []
    for path in sorted(fonts_dir.rglob("*")):
        if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        if not any(can_render(str(path), text) for text in texts[:3]):
            continue
        records.append({"path": str(path), "name": path.stem, "category": fonts_dir.name})
        if limit > 0 and len(records) >= limit:
            break
    return records


def read_texts(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return DEFAULT_TEXTS
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts or DEFAULT_TEXTS


def write_metadata(path: Path, records: list[dict], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fonts": records, "texts": texts}, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
