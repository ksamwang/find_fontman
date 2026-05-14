from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import AIPaths, resolve_path
from .dataset import FontRenderDataset, read_texts, scan_font_records, write_metadata
from .deps import DataLoader, F, torch, require_torch
from .logging import JsonlLogger
from .texts import TextSampler


def train(args: argparse.Namespace) -> None:
    require_torch()
    from .model import create_model

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_interop_threads)

    root = Path(args.root).resolve()
    fonts_dir = resolve_path(root, args.fonts)
    output_dir = resolve_path(root, args.output)
    paths = AIPaths(root=root, fonts=fonts_dir, output=output_dir)
    paths.ai_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(paths.train_log)

    fixed_texts = read_texts(resolve_path(root, args.texts) if args.texts else None)
    text_sampler = TextSampler(fixed_texts)
    config = vars(args).copy()
    config.update(
        {
            "root": str(root),
            "fonts": str(fonts_dir),
            "output": str(output_dir),
            "fixed_text_count": len(fixed_texts),
            "text_sampler_preview": text_sampler.preview(),
            "scan_probe_texts": text_sampler.probe_texts(),
        }
    )
    paths.train_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.emit({"event": "font_scan_start", "fonts_dir": str(fonts_dir), "limit_fonts": args.limit_fonts})
    records = scan_font_records(
        fonts_dir,
        limit=args.limit_fonts,
        texts=text_sampler.probe_texts(),
        progress=logger.emit,
        log_every=args.scan_log_every,
    )
    if not records:
        raise RuntimeError("no trainable fonts found")

    dataset = FontRenderDataset(records, text_sampler, samples_per_font=args.samples_per_font, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(num_classes=len(records)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0

    if args.resume and paths.checkpoint.exists():
        ckpt = torch.load(paths.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    total_batches = len(loader)
    total_samples = total_batches * args.batch_size
    total_train_batches = total_batches * max(0, args.epochs - start_epoch)
    logger.emit(
        {
            "event": "train_start",
            "device": str(device),
            "cuda": cuda_status(),
            "fonts": len(records),
            "samples_per_font": args.samples_per_font,
            "dataset_samples": len(dataset),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "workers": args.workers,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "start_epoch": start_epoch,
            "epochs": args.epochs,
            "batches_per_epoch": total_batches,
            "train_batches_remaining": total_train_batches,
            "fixed_text_count": len(fixed_texts),
            "text_sampler_preview": text_sampler.preview(12),
        }
    )

    train_started = time.monotonic()
    completed_batches = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        epoch_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels) / args.grad_accum
            loss.backward()
            total_loss += float(loss.item()) * args.grad_accum
            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            completed_batches += 1
            if step % args.log_every == 0:
                elapsed = max(1e-9, time.monotonic() - train_started)
                batches_per_sec = completed_batches / elapsed
                samples_seen = completed_batches * args.batch_size
                eta_seconds = max(0, total_train_batches - completed_batches) / max(1e-9, batches_per_sec)
                logger.emit(
                    {
                        "event": "train_progress",
                        "epoch": epoch,
                        "epoch_step": step,
                        "epoch_batches": total_batches,
                        "progress_pct": round(100.0 * completed_batches / max(1, total_train_batches), 3),
                        "loss": total_loss / step,
                        "samples_seen": samples_seen,
                        "total_samples": total_samples * max(0, args.epochs - start_epoch),
                        "samples_per_sec": round(samples_seen / elapsed, 2),
                        "batches_per_sec": round(batches_per_sec, 3),
                        "epoch_elapsed_sec": round(time.monotonic() - epoch_started, 1),
                        "elapsed_sec": round(elapsed, 1),
                        "eta_sec": round(eta_seconds, 1),
                        "cuda": cuda_status(),
                    }
                )
        save_checkpoint(paths.checkpoint, model, optimizer, epoch, records, fixed_texts)
        write_metadata(paths.metadata, records, fixed_texts)
        logger.emit({"event": "checkpoint_saved", "epoch": epoch, "checkpoint": str(paths.checkpoint), "elapsed_sec": round(time.monotonic() - train_started, 1)})


def cuda_status() -> dict:
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(idx),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024 / 1024, 1),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024 / 1024, 1),
    }


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
    parser.add_argument("--fonts", default="fonts")
    parser.add_argument("--output", default="training/output")
    parser.add_argument("--texts", default="training/texts/zh_common.txt")
    parser.add_argument("--samples-per-font", type=int, default=1000)
    parser.add_argument("--limit-fonts", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--torch-interop-threads", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--scan-log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
