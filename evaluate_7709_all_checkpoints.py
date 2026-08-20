#!/usr/bin/env python3
"""Evaluate every original 07709 DeepLOB checkpoint on the held-out test day."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_7709_deeplob import (
    DayData,
    DeepLOB,
    PreparedDay,
    date_from_path,
    eligible_targets,
    equation4_returns,
    make_loader,
    prepare_day,
    run_epoch,
)
from train_pytorch_mac import choose_device


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path.home() / "Desktop/7709_tickdata",
        help="directory containing seven dated 07709 CSV files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/7709",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/7709",
    )
    parser.add_argument("--test-date", default="2026-08-07")
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--security-id", type=int, default=7709)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--force-reconstruct", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/results/7709_all_checkpoints_test.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_candidates = [
        path
        for path in args.raw_dir.glob("hk07709_*.csv")
        if date_from_path(path) == args.test_date
    ]
    if len(raw_candidates) != 1:
        raise FileNotFoundError(
            f"Expected one raw CSV for {args.test_date}, found {raw_candidates}"
        )
    raw_path = raw_candidates[0]
    prepared_path, reconstruction = prepare_day(
        raw_path,
        args.processed_dir,
        args.snapshot_every,
        args.security_id,
        args.force_reconstruct,
    )
    archive = np.load(prepared_path, mmap_mode="r", allow_pickle=False)
    test_day = DayData(
        args.test_date,
        prepared_path,
        archive["features"].astype(np.float32, copy=False),
        archive["sessions"].astype(np.uint8, copy=False),
        reconstruction,
    )

    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    criterion = nn.CrossEntropyLoss()
    results: list[dict[str, object]] = []

    for k in args.ks:
        checkpoint_path = args.checkpoint_dir / f"deeplob_7709_k{k}_best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        checkpoint_k = int(checkpoint["labeling"]["k"])
        if checkpoint_k != k:
            raise ValueError(
                f"{checkpoint_path}: expected k={k}, checkpoint has k={checkpoint_k}"
            )
        alpha = float(checkpoint["labeling"]["alpha"])
        normalizer = checkpoint["preprocessing"]["normalizers"][args.test_date]
        test_day.mean = np.asarray(normalizer["mean"], dtype=np.float32)
        test_day.std = np.asarray(normalizer["std"], dtype=np.float32)
        test_day.normalization_sources = tuple(normalizer["source_dates"])
        returns = equation4_returns(test_day, k)
        indices = eligible_targets(
            test_day, returns, args.sequence_length, stride=1
        )
        values = returns[indices]
        labels = np.where(
            values < -alpha, 0, np.where(values > alpha, 2, 1)
        ).astype(np.int64)
        test_item = PreparedDay(test_day, indices, labels)
        loader = make_loader(
            [test_item],
            args.sequence_length,
            args.batch_size,
            False,
            args.num_workers,
            42,
        )
        model = DeepLOB().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loss, metrics = run_epoch(model, loader, criterion, device)

        embedded = checkpoint.get("test_metrics")
        consistency: dict[str, object] | None = None
        if embedded is not None:
            consistency = {
                name: abs(float(metrics[name]) - float(embedded[name]))
                for name in ("accuracy", "balanced_accuracy", "macro_f1")
            }
        record = {
            "checkpoint": str(checkpoint_path.resolve()),
            "k": k,
            "alpha": alpha,
            "loss": loss,
            **metrics,
            "label_counts": np.bincount(labels, minlength=3).tolist(),
            "normalization_sources": list(test_day.normalization_sources),
            "embedded_test_metric_absolute_differences": consistency,
        }
        results.append(record)
        print(
            f"k={k}: samples={metrics['samples']:,}, loss={loss:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    output = {
        "experiment": "All original DeepLOB 07709 checkpoints on held-out test day",
        "test_date": args.test_date,
        "device": str(device),
        "sequence_length": args.sequence_length,
        "snapshot_every_book_update_groups": args.snapshot_every,
        "raw_test_file": str(raw_path.resolve()),
        "results": results,
        "note": (
            "These post-hoc k=50 and k=100 test evaluations are diagnostic only. "
            "They must not be used to revise the original k selection, which was "
            "made using validation macro-F1 before the test day was opened."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output.resolve()}", flush=True)

    archive.close()


if __name__ == "__main__":
    main()
