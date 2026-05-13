from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import AIPaths
from .dataset import read_metadata
from .deps import torch, require_torch
from .synth import render_training_image
from .transforms import image_to_tensor
from python_service.vision.deps import np


def build_index(args: argparse.Namespace) -> None:
    require_torch()
    from .model import create_model

    root = Path(args.root).resolve()
    paths = AIPaths(root=root, fonts=Path(args.fonts).resolve(), data=Path(args.data).resolve())
    metadata = read_metadata(paths.metadata)
    records = metadata["fonts"]
    texts = metadata["texts"]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(paths.checkpoint, map_location=device)
    model = create_model(num_classes=int(ckpt["num_classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    embeddings = []
    with torch.no_grad():
        for idx, record in enumerate(records):
            vectors = []
            rng = random.Random(args.seed + idx * 97)
            for sample_idx in range(args.samples_per_font):
                text = texts[(idx + sample_idx) % len(texts)]
                image = render_training_image(record["path"], text, rng)
                tensor = image_to_tensor(image).unsqueeze(0).to(device)
                vector = model(tensor).cpu().numpy()[0]
                vectors.append(vector)
            mean = np.mean(np.stack(vectors), axis=0)
            mean = mean / max(1e-12, float(np.linalg.norm(mean)))
            embeddings.append(mean.astype(np.float32))
            if (idx + 1) % args.log_every == 0:
                print(json.dumps({"indexed": idx + 1, "total": len(records)}, ensure_ascii=False))

    paths.ai_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.index, embeddings=np.stack(embeddings).astype(np.float32))
    paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"index": str(paths.index), "fonts": len(records), "dim": int(embeddings[0].shape[0])}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fonts", default="fonts/1中文简体")
    parser.add_argument("--data", default="data")
    parser.add_argument("--samples-per-font", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
