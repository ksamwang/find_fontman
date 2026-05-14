from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from .deps import Dataset, require_torch
from .synth import can_render, render_training_image
from .texts import TextSampler, read_fixed_texts
from .transforms import image_to_tensor


BaseDataset = Dataset if Dataset is not None else object


class FontRenderDataset(BaseDataset):
    def __init__(self, font_records: list[dict], text_sampler: TextSampler, samples_per_font: int, seed: int = 7) -> None:
        require_torch()
        self.font_records = font_records
        self.text_sampler = text_sampler
        self.samples_per_font = samples_per_font
        self.seed = seed

    def __len__(self) -> int:
        return len(self.font_records) * self.samples_per_font

    def __getitem__(self, idx: int):
        font_idx = idx // self.samples_per_font
        sample_idx = idx % self.samples_per_font
        record = self.font_records[font_idx]
        rng = random.Random(self.seed + idx * 9973)
        text = self.text_sampler.sample(rng).text
        image = render_training_image(record["path"], text, rng)
        return image_to_tensor(image), font_idx


def scan_font_records(
    fonts_dir: Path,
    limit: int = 0,
    texts: list[str] | None = None,
    progress: Callable[[dict], None] | None = None,
    log_every: int = 100,
) -> list[dict]:
    texts = texts or TextSampler().probe_texts()
    records = []
    scanned = 0
    skipped = 0
    for path in sorted(fonts_dir.rglob("*")):
        if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        scanned += 1
        if not any(can_render(str(path), text) for text in texts[:3]):
            skipped += 1
            if progress and scanned % log_every == 0:
                progress({"event": "font_scan_progress", "scanned": scanned, "accepted": len(records), "skipped": skipped})
            continue
        records.append({"path": str(path), "name": path.stem, "category": path.parent.name})
        if progress and scanned % log_every == 0:
            progress({"event": "font_scan_progress", "scanned": scanned, "accepted": len(records), "skipped": skipped})
        if limit > 0 and len(records) >= limit:
            break
    if progress:
        progress({"event": "font_scan_done", "scanned": scanned, "accepted": len(records), "skipped": skipped})
    return records


def read_texts(path: Path | None) -> list[str]:
    return read_fixed_texts(path, defaults=True)


def write_metadata(path: Path, records: list[dict], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fonts": records, "texts": texts}, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
