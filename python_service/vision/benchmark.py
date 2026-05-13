from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .config import Settings
from .deps import Image, ImageDraw, ImageFilter, ImageFont, np, require_numpy, require_pillow
from .font_index import FontIndex
from .matcher import FontMatcher


def run_benchmark(settings: Settings, texts_path: Path, sample_size: int, seed: int) -> dict:
    require_pillow()
    require_numpy()
    random.seed(seed)
    out_dir = settings.data / "benchmark"
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    texts = read_texts(texts_path)
    index = FontIndex(settings.fonts, settings.data / "font_index.sqlite")
    index.ensure()
    matcher = FontMatcher(index, settings.previews)
    fonts = []
    for record in index.records():
        if "\u4e2d\u6587" not in record.category:
            continue
        if matcher.font_can_render(record.path, texts[0]):
            fonts.append(record)
        if len(fonts) >= max(sample_size * 3, sample_size):
            break
    random.shuffle(fonts)
    samples = fonts[:sample_size]
    cases = []
    started = time.time()
    for idx, font in enumerate(samples, start=1):
        text = texts[(idx - 1) % len(texts)]
        image = synthesize(font.path, text)
        image_path = image_dir / f"sample_{idx:03d}.png"
        image.save(image_path)
        result = matcher.match(image, text, top_k=10, progress=None)
        paths = [item["font_path"] for item in result["results"]]
        rank = paths.index(font.path) + 1 if font.path in paths else None
        cases.append(
            {
                "font_name": font.name,
                "font_path": font.path,
                "text": text,
                "image_path": str(image_path),
                "rank": rank,
                "elapsed_ms": result["elapsed_ms"],
                "top1": result["results"][0]["font_name"] if result["results"] else "",
            }
        )
    report = summarize(cases, int((time.time() - started) * 1000))
    write_report(out_dir, report)
    return report


def read_texts(path: Path) -> list[str]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("字体识别\n设计标题\n品牌海报\n", encoding="utf-8")
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts or ["字体识别"]


def synthesize(font_path: str, text: str):
    font_size = 72
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    width = max(240, bbox[2] - bbox[0] + 80)
    height = max(120, bbox[3] - bbox[1] + 60)
    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), (246, 246, 242))
    draw = ImageDraw.Draw(image)
    scaled_font = ImageFont.truetype(font_path, font_size * scale)
    draw.text((40 * scale, 28 * scale), text, font=scaled_font, fill=(26, 26, 26))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
    arr = np.asarray(image).astype(np.int16)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def summarize(cases: list[dict], elapsed_ms: int) -> dict:
    total = len(cases)
    top1 = sum(1 for case in cases if case["rank"] == 1)
    top3 = sum(1 for case in cases if case["rank"] is not None and case["rank"] <= 3)
    top10 = sum(1 for case in cases if case["rank"] is not None and case["rank"] <= 10)
    return {
        "total": total,
        "top1": top1 / total if total else 0,
        "top3": top3 / total if total else 0,
        "top10": top10 / total if total else 0,
        "elapsed_ms": elapsed_ms,
        "cases": cases,
    }


def write_report(out_dir: Path, report: dict) -> None:
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "\n".join(
        f"<tr><td>{case['font_name']}</td><td>{case['text']}</td><td>{case['rank']}</td><td>{case['top1']}</td></tr>"
        for case in report["cases"]
    )
    html = f"""
    <html><meta charset="utf-8"><body>
    <h1>Find Fontman Benchmark</h1>
    <p>Top1: {report['top1']:.2%} Top3: {report['top3']:.2%} Top10: {report['top10']:.2%}</p>
    <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Truth</th><th>Text</th><th>Rank</th><th>Top1</th></tr>
    {rows}
    </table>
    </body></html>
    """
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fonts", default="fonts")
    parser.add_argument("--data", default="data")
    parser.add_argument("--previews", default="data/previews")
    parser.add_argument("--texts", default="data/benchmark_texts.txt")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    settings = Settings(
        addr="benchmark",
        root=root,
        fonts=Path(args.fonts).resolve(),
        data=Path(args.data).resolve(),
        previews=Path(args.previews).resolve(),
    )
    report = run_benchmark(settings, Path(args.texts), args.sample_size, args.seed)
    print(json.dumps({k: report[k] for k in ("total", "top1", "top3", "top10", "elapsed_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
