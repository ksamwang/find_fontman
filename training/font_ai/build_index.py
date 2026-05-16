from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import AIPaths, resolve_path
from .dataset import read_metadata, render_record_with_retries
from .deps import torch, require_torch
from .texts import TextSampler
from .transforms import image_to_tensor


def build_index(args: argparse.Namespace) -> None:
    require_torch()
    from .model import create_model

    root = Path(args.root).resolve()
    paths = AIPaths(root=root, fonts=resolve_path(root, args.fonts), output=resolve_path(root, args.output))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(paths.checkpoint, map_location=device)
    if "records" in ckpt and "texts" in ckpt:
        records = ckpt["records"]
        texts = ckpt["texts"]
        metadata = {"fonts": records, "texts": texts}
    else:
        metadata = read_metadata(paths.metadata)
        records = metadata["fonts"]
        texts = metadata["texts"]
    model = create_model(num_classes=int(ckpt["num_classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    text_sampler = TextSampler(texts)

    embeddings = []
    with torch.no_grad():
        for idx, record in enumerate(records):
            vectors = []
            for sample_idx in range(args.samples_per_font):
                image = render_record_with_retries(record, text_sampler, args.seed + idx * 97 + sample_idx * 10_007)
                tensor = image_to_tensor(image).unsqueeze(0).to(device)
                vector = model(tensor).cpu().numpy()[0]
                vectors.append(vector)
            mean = np.mean(np.stack(vectors), axis=0)
            mean = mean / max(1e-12, float(np.linalg.norm(mean)))
            embeddings.append(mean.astype(np.float32))
            if (idx + 1) % args.log_every == 0:
                print(json.dumps({"event": "index_progress", "indexed": idx + 1, "total": len(records)}, ensure_ascii=False), flush=True)

    paths.ai_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.index, embeddings=np.stack(embeddings).astype(np.float32))
    paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "index_done", "index": str(paths.index), "fonts": len(records), "dim": int(embeddings[0].shape[0])}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fonts", default="fonts")
    parser.add_argument("--output", default="training/output")
    parser.add_argument("--samples-per-font", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
