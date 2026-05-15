from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from .config import AIPaths, resolve_path
from .dataset import FontRenderDataset, balanced_family_order, coverage_summary, read_texts, scan_font_records, write_metadata, summarize_labels
from .deps import DataLoader, F, torch, require_torch
from .logging import JsonlLogger
from .texts import TextSampler


def train(args: argparse.Namespace) -> None:
    require_torch()
    from .model import create_model

    warnings.filterwarnings("ignore", category=FutureWarning)
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
            "record_count": 0,
        }
    )
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
    config["label_summary"] = summarize_labels(records)
    config["record_count"] = len(records)
    paths.train_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset = FontRenderDataset(records, text_sampler, samples_per_font=args.samples_per_font, seed=args.seed)
    drop_last = len(dataset) >= args.batch_size
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=drop_last,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    model = create_model(num_classes=len(records)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0

    if args.resume and paths.checkpoint.exists():
        ckpt = torch.load(paths.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        if ckpt.get("epoch_complete", False):
            start_epoch += 1

    total_batches = max(1, len(loader))
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
            "label_summary": summarize_labels(records),
            "coverage_summary": coverage_summary(records),
            "hard_negative_ratio": args.hard_negative_ratio,
        }
    )

    balanced_order = balanced_family_order(records, args.seed, hard_negative_ratio=args.hard_negative_ratio)
    logger.emit(
        {
            "event": "sampling_ready",
            "balanced_order_size": len(balanced_order),
            "hard_negative_tail_size": max(0, len(balanced_order) - len(records) * 2),
            "family_groups": len({record.get("family_name", "unknown") for record in records}),
            "coverage_summary": coverage_summary(records),
            "hard_negative_ratio": args.hard_negative_ratio,
        }
    )

    train_started = time.monotonic()
    completed_batches = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        epoch_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        accum_since_step = 0
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images, labels)
                loss = F.cross_entropy(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += float(loss.item()) * args.grad_accum
            accum_since_step += 1
            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_since_step = 0
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
                        "label_summary": summarize_labels(records),
                        "coverage_summary": coverage_summary(records),
                        "hard_negative_ratio": args.hard_negative_ratio,
                        "cuda": cuda_status(),
                    }
                )
            if args.checkpoint_every > 0 and completed_batches % args.checkpoint_every == 0:
                save_checkpoint(paths.checkpoint, model, optimizer, epoch, records, fixed_texts, epoch_complete=False, global_step=completed_batches)
                write_metadata(paths.metadata, records, fixed_texts)
                logger.emit(
                    {
                        "event": "checkpoint_saved",
                        "epoch": epoch,
                        "epoch_step": step,
                        "global_step": completed_batches,
                        "checkpoint": str(paths.checkpoint),
                        "elapsed_sec": round(time.monotonic() - train_started, 1),
                    }
                )
        if accum_since_step > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        save_checkpoint(paths.checkpoint, model, optimizer, epoch, records, fixed_texts, epoch_complete=True, global_step=completed_batches)
        write_metadata(paths.metadata, records, fixed_texts)
        logger.emit(
            {
                "event": "checkpoint_saved",
                "epoch": epoch,
                "epoch_step": total_batches,
                "global_step": completed_batches,
                "checkpoint": str(paths.checkpoint),
                "elapsed_sec": round(time.monotonic() - train_started, 1),
            }
        )


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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    records: list[dict],
    texts: list[str],
    epoch_complete: bool,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "global_step": global_step,
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
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
