from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .deps import Dataset, require_torch
from .synth import can_render, render_training_image
from .texts import TextSampler, read_fixed_texts
from .transforms import image_to_tensor

try:
    from fontTools.ttLib import TTCollection, TTFont
except Exception:  # pragma: no cover
    TTCollection = None
    TTFont = None


BaseDataset = Dataset if Dataset is not None else object
FONT_EXTS = {".ttf", ".otf", ".ttc"}


@dataclass(frozen=True)
class FontFaceRecord:
    path: str
    file_path: str
    face_index: int
    name: str
    script_scope: str
    style_group: str
    family_name: str
    weight_name: str
    is_italic: bool
    is_ttc: bool
    category: str


class FontRenderDataset(BaseDataset):
    def __init__(
        self,
        font_records: list[dict],
        text_sampler: TextSampler,
        samples_per_font: int,
        seed: int = 7,
        hard_negative_ratio: float = 0.25,
    ) -> None:
        require_torch()
        self.font_records = font_records
        self.text_sampler = text_sampler
        self.samples_per_font = samples_per_font
        self.seed = seed
        self.epoch = 0
        self.order = build_balanced_order(font_records, samples_per_font, seed, hard_negative_ratio=hard_negative_ratio)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int):
        font_idx, sample_idx = self.order[idx]
        record = self.font_records[font_idx]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + idx * 9_973 + sample_idx * 37)
        sample = self.text_sampler.sample(rng, record.get("script_scope", "zh_simplified"))
        style = sample_style(rng, sample.kind, record)
        image = render_training_image(
            record["file_path"],
            sample.text,
            rng,
            style=style,
            face_index=int(record.get("face_index", 0)),
        )
        return image_to_tensor(image), font_idx


def build_balanced_order(records: list[dict], samples_per_font: int, seed: int, hard_negative_ratio: float = 0.25) -> list[tuple[int, int]]:
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        group = family_bucket(record)
        groups.setdefault(group, []).append(idx)
    group_names = sorted(groups)
    rng = random.Random(seed)
    order: list[tuple[int, int]] = []
    for sample_idx in range(samples_per_font):
        shuffled_groups = group_names[:]
        rng.shuffle(shuffled_groups)
        for group in shuffled_groups:
            font_indices = groups[group][:]
            rng.shuffle(font_indices)
            for font_idx in font_indices:
                order.append((font_idx, sample_idx))
    order.extend(build_hard_negative_tail(records, seed, hard_negative_ratio=hard_negative_ratio, sample_offset=samples_per_font))
    return order


def family_bucket(record: dict) -> str:
    return "|".join(
        [
            str(record.get("script_scope", "unknown")),
            str(record.get("style_group", "unknown")),
            str(record.get("family_name", "unknown")),
            str(record.get("weight_name", "unknown")),
            "italic" if record.get("is_italic") else "upright",
        ]
    )


def build_hard_negative_tail(
    records: list[dict],
    seed: int,
    hard_negative_ratio: float = 0.25,
    sample_offset: int = 0,
) -> list[tuple[int, int]]:
    if hard_negative_ratio <= 0:
        return []
    by_family: dict[str, list[int]] = {}
    by_style: dict[str, list[int]] = {}
    by_weight: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        by_family.setdefault(str(record.get("family_name", "unknown")), []).append(idx)
        by_style.setdefault(str(record.get("style_group", "unknown")), []).append(idx)
        by_weight.setdefault(str(record.get("weight_name", "unknown")), []).append(idx)
    rng = random.Random(seed + 999)
    tail: list[tuple[int, int]] = []
    sample_idx = sample_offset

    def append_tail(font_idx: int) -> None:
        nonlocal sample_idx
        tail.append((font_idx, sample_idx))
        sample_idx += 1

    families = [name for name, members in by_family.items() if len(members) > 1]
    styles = [name for name, members in by_style.items() if len(members) > 1]
    if families:
        for family in families:
            members = by_family[family][:]
            rng.shuffle(members)
            for idx in members[: max(1, int(len(members) * hard_negative_ratio))]:
                append_tail(idx)
                if hard_negative_ratio >= 0.5:
                    append_tail(idx)
    if styles:
        for style in styles:
            members = by_style[style][:]
            rng.shuffle(members)
            for idx in members[: max(1, int(len(members) * hard_negative_ratio))]:
                append_tail(idx)
    if by_weight:
        for weight, members in by_weight.items():
            if len(members) > 1:
                rng.shuffle(members)
                for idx in members[: min(max(1, int(len(members) * hard_negative_ratio)), len(members))]:
                    append_tail(idx)
    return tail


def coverage_summary(records: list[dict]) -> dict[str, int]:
    summary: dict[str, int] = {
        "records": len(records),
        "families": len({record.get("family_name", "unknown") for record in records}),
        "styles": len({record.get("style_group", "unknown") for record in records}),
        "weights": len({record.get("weight_name", "unknown") for record in records}),
        "italic": sum(1 for record in records if record.get("is_italic")),
    }
    return summary


def scan_font_records(
    fonts_dir: Path,
    limit: int = 0,
    texts: list[str] | None = None,
    progress: Callable[[dict], None] | None = None,
    log_every: int = 100,
) -> list[dict]:
    texts = texts or TextSampler().probe_texts()
    records: list[dict] = []
    scanned = 0
    skipped = 0
    for path in sorted(fonts_dir.rglob("*")):
        if path.suffix.lower() not in FONT_EXTS:
            continue
        scanned += 1
        faces = extract_font_faces(path)
        if not faces:
            skipped += 1
            continue
        for face in faces:
            if not any(can_render(face["file_path"], text, int(face.get("face_index", 0))) for text in texts[:3]):
                skipped += 1
                continue
            records.append(face)
            if limit > 0 and len(records) >= limit:
                break
        if progress and scanned % log_every == 0:
            progress({"event": "font_scan_progress", "scanned": scanned, "accepted": len(records), "skipped": skipped})
        if limit > 0 and len(records) >= limit:
            break
    if progress:
        progress({"event": "font_scan_done", "scanned": scanned, "accepted": len(records), "skipped": skipped})
    return records


def extract_font_faces(path: Path) -> list[dict]:
    base = base_record(path)
    if path.suffix.lower() == ".ttc" and TTCollection is not None:
        try:
            collection = TTCollection(str(path))
            faces = []
            for face_index, font in enumerate(collection.fonts):
                face = dict(base)
                face.update(face_from_font(path, font, face_index))
                faces.append(face)
            return faces
        except Exception:
            return [dict(base, face_index=0, is_ttc=True)]
    if TTFont is not None:
        try:
            font = TTFont(str(path), lazy=True)
            return [dict(base, **face_from_font(path, font, 0))]
        except Exception:
            pass
    return [dict(base, face_index=0, is_ttc=path.suffix.lower() == ".ttc")]


def base_record(path: Path) -> dict:
    return {
        "path": str(path),
        "file_path": str(path),
        "face_index": 0,
        "name": path.stem,
        "script_scope": script_scope_for(path),
        "style_group": style_group_for(path),
        "family_name": family_name_for(path),
        "weight_name": "regular",
        "is_italic": False,
        "is_ttc": path.suffix.lower() == ".ttc",
        "category": top_category_for(path),
    }


def face_from_font(path: Path, font, face_index: int) -> dict:
    family = ""
    subfamily = ""
    try:
        names = font["name"].names
        family = pick_name(names, 1) or ""
        subfamily = pick_name(names, 2) or ""
    except Exception:
        pass
    joined = " ".join(part for part in [family, subfamily] if part).strip()
    display = joined or path.stem
    return {
        "path": str(path),
        "file_path": str(path),
        "face_index": face_index,
        "name": display,
        "script_scope": script_scope_for(path),
        "style_group": style_group_for(path),
        "family_name": family_name_for(path, display),
        "weight_name": weight_from_name(display),
        "is_italic": is_italic_name(display),
        "is_ttc": path.suffix.lower() == ".ttc",
        "category": top_category_for(path),
    }


def pick_name(names, name_id: int) -> str:
    for item in names:
        if getattr(item, "nameID", None) == name_id:
            try:
                value = item.toUnicode().strip()
                if value:
                    return value
            except Exception:
                continue
    return ""


def top_category_for(path: Path) -> str:
    return "english" if any("\u82f1\u6587" in part for part in path.parts) else "zh_simplified"


def script_scope_for(path: Path) -> str:
    return top_category_for(path)


def style_group_for(path: Path) -> str:
    for part in path.parts:
        if "\u9ed1\u4f53" in part:
            return "black"
        if "\u5b8b\u4f53" in part:
            return "serif"
        if "\u5706\u4f53" in part:
            return "round"
        if "\u6977\u4f53" in part:
            return "kai"
        if "\u6bdb\u7b14" in part or "\u786c\u7b14" in part:
            return "brush"
        if "\u521b\u610f" in part:
            return "creative"
        if "UI" in part or "\u7f16\u7801" in part:
            return "ui"
    return "other"


def family_name_for(path: Path, fallback: str | None = None) -> str:
    for part in reversed(path.parts):
        if part.lower().endswith((".ttf", ".otf", ".ttc")):
            continue
        if any(token in part for token in ("\u7b80", "\u4e2d\u6587", "\u82f1\u6587", "\u9ed1\u4f53", "\u5b8b\u4f53", "\u6977\u4f53", "\u5706\u4f53", "UI")):
            return normalize_family_name(part)
    return normalize_family_name(fallback or path.stem)


def normalize_family_name(text: str) -> str:
    text = re.sub(r"(完整版|简体|繁体|字体|字库|系列|全集|全套)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def weight_from_name(text: str) -> str:
    lowered = text.lower()
    if "extralight" in lowered or "ultralight" in lowered or "thin" in lowered:
        return "extralight"
    if "light" in lowered or "\u7ec6" in text:
        return "light"
    if "semibold" in lowered or "demibold" in lowered or "medium" in lowered:
        return "medium"
    if "bold" in lowered or "\u7c97" in text:
        return "bold"
    if "heavy" in lowered or "black" in lowered:
        return "heavy"
    return "regular"


def is_italic_name(text: str) -> bool:
    lowered = text.lower()
    return "italic" in lowered or "oblique" in lowered or "\u659c" in text


def read_texts(path: Path | None) -> list[str]:
    return read_fixed_texts(path, defaults=True)


def write_metadata(path: Path, records: list[dict], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "2",
        "fonts": records,
        "texts": texts,
        "label_summary": summarize_labels(records),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_labels(records: list[dict]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        scope = record.get("script_scope", "unknown")
        style = record.get("style_group", "unknown")
        weight = record.get("weight_name", "unknown")
        family = record.get("family_name", "unknown")
        summary[f"scope:{scope}"] = summary.get(f"scope:{scope}", 0) + 1
        summary[f"style:{style}"] = summary.get(f"style:{style}", 0) + 1
        summary[f"weight:{weight}"] = summary.get(f"weight:{weight}", 0) + 1
        summary[f"family:{family}"] = summary.get(f"family:{family}", 0) + 1
    return summary


def sample_style(rng: random.Random, kind: str, record: dict | None = None) -> dict:
    record = record or {}
    scope = record.get("script_scope", "zh_simplified")
    style_group = record.get("style_group", "other")
    weight = record.get("weight_name", "regular")
    base = {
        "font_size": rng.randint(42, 88),
        "scale_x": rng.uniform(0.85, 1.18),
        "scale_y": rng.uniform(0.85, 1.12),
        "spacing": rng.randint(-6, 10),
        "line_gap": rng.randint(4, 20),
        "offset_x": rng.randint(-16, 16),
        "offset_y": rng.randint(-12, 12),
        "angle": rng.uniform(-4.0, 4.0),
        "blur_prob": 0.65,
        "noise_prob": 0.8,
        "contrast_prob": 0.45,
        "shadow_prob": 0.25,
        "stroke_prob": 0.15,
        "crop_prob": 0.18,
        "jpeg_prob": 0.35,
    }
    if scope == "english":
        base.update({"font_size": rng.randint(32, 72), "spacing": rng.randint(-2, 6), "angle": rng.uniform(-2.0, 2.0)})
    if kind in {"person", "company", "industry"}:
        base.update({"spacing": rng.randint(-8, 12), "line_gap": rng.randint(2, 14), "scale_x": rng.uniform(0.82, 1.24)})
    if kind in {"promo", "mixed", "number"}:
        base.update({"angle": rng.uniform(-5.5, 5.5), "shadow_prob": 0.35, "stroke_prob": 0.2, "jpeg_prob": 0.5})
    if style_group in {"black", "ui"}:
        base["stroke_prob"] = 0.22
    if weight in {"bold", "heavy"}:
        base["font_size"] = rng.randint(48, 92)
        base["spacing"] = rng.randint(-8, 8)
    if rng.random() < 0.35:
        base["align"] = rng.choice(["left", "center", "right"])
    return base


def balanced_family_order(records: list[dict], seed: int, hard_negative_ratio: float = 0.25) -> list[int]:
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        key = "|".join(
            [
                str(record.get("script_scope", "unknown")),
                str(record.get("style_group", "unknown")),
                str(record.get("family_name", "unknown")),
            ]
        )
        groups.setdefault(key, []).append(idx)
    keys = sorted(groups)
    rng = random.Random(seed)
    order: list[int] = []
    for _ in range(2):
        shuffled_keys = keys[:]
        rng.shuffle(shuffled_keys)
        for key in shuffled_keys:
            members = groups[key][:]
            rng.shuffle(members)
            order.extend(members)
    order.extend(idx for idx, _ in build_hard_negative_tail(records, seed, hard_negative_ratio=hard_negative_ratio))
    return order
