from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AIPaths, resolve_fonts_dir
from .dataset import FontRenderDataset, read_texts, scan_font_records, write_metadata
from .deps import DataLoader, F, torch, require_torch


def train(args: argparse.Namespace) -> None:
    require_torch()
    from .model import create_model

    root = Path(args.root).resolve()
    fonts_dir = resolve_fonts_dir(root, args.fonts)
    paths = AIPaths(root=root, fonts=fonts_dir, data=Path(args.data).resolve())
    paths.ai_dir.mkdir(parents=True, exist_ok=True)
    texts = read_texts(Path(args.texts) if args.texts else None)
    records = scan_font_records(fonts_dir, limit=args.limit_fonts, texts=texts)
    if not records:
        raise RuntimeError("no trainable fonts found")
    write_metadata(paths.metadata, records, texts)

    dataset = FontRenderDataset(records, texts, samples_per_font=args.samples_per_font, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(num_classes=len(records)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0

    if args.resume and paths.checkpoint.exists():
        ckpt = torch.load(paths.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels) / args.grad_accum
            loss.backward()
            total_loss += float(loss.item()) * args.grad_accum
            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": total_loss / step}, ensure_ascii=False))
        save_checkpoint(paths.checkpoint, model, optimizer, epoch, records, texts)


def save_checkpoint(path: Path, model, optimizer, epoch: int, records: list[dict], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "num_classes": len(records),
            "records": records,
            "texts": texts,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fonts", default="")
    parser.add_argument("--data", default="data")
    parser.add_argument("--texts", default="data/benchmark_texts.txt")
    parser.add_argument("--samples-per-font", type=int, default=100)
    parser.add_argument("--limit-fonts", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
