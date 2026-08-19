#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "torch>=2.1",
# ]
# ///
"""Zero-shot evaluation of the public-LOBSTER checkpoint on HK 7709."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_7709_transfer import (
    WindowDataset,
    evaluate,
    forward_returns,
    labels_from_returns,
    majority_baseline,
    valid_target_indices,
)
from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the LOBSTER-sample DeepLOB checkpoint on HK 7709"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data/processed/hk07709_2026-07-09_lob.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/deeplob_lobster_5stocks_best.pt",
    )
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument(
        "--horizon-snapshots",
        type=int,
        default=10,
        help="10 snapshots approximate the checkpoint's 100 raw-event horizon",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--evaluation-alpha",
        type=float,
        default=None,
        help="override the checkpoint label threshold for a matched comparison",
    )
    parser.add_argument(
        "--cutoff-fraction",
        type=float,
        default=0.0,
        help="exclude an initial fraction to match another evaluation interval",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/results/lobster_to_7709_zero_shot.json",
    )
    return parser.parse_args()


def transform_7709(features: np.ndarray, feature_scale: float, checkpoint: dict) -> np.ndarray:
    transformed = features.astype(np.float32, copy=True)
    price_columns = np.arange(0, transformed.shape[1], 2)
    size_columns = np.arange(1, transformed.shape[1], 2)
    mid = (transformed[:, 0].astype(np.float64) + transformed[:, 2]) / 2.0
    transformed[:, price_columns] = (
        transformed[:, price_columns] / mid[:, None] - 1.0
    ) * 10_000.0
    # prepare_hkex_lob.py divided both prices and quantities by feature_scale.
    transformed[:, size_columns] = np.log1p(
        transformed[:, size_columns] * feature_scale
    )
    preprocessing = checkpoint["preprocessing"]
    mean = np.asarray(preprocessing["zscore_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["zscore_std"], dtype=np.float32)
    return np.ascontiguousarray((transformed - mean) / std)


def main() -> None:
    args = parse_args()
    if not args.data.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing data or checkpoint: {args.data}, {args.checkpoint}")
    if not 0.0 <= args.cutoff_fraction < 1.0:
        raise ValueError("--cutoff-fraction must be in [0, 1)")
    archive = np.load(args.data, allow_pickle=False)
    source_features = archive["features"].astype(np.float32, copy=False)
    sessions = archive["sessions"].astype(np.uint8, copy=False)
    metadata = json.loads(str(archive["metadata"]))
    if source_features.ndim != 2 or source_features.shape[1] != 40:
        raise ValueError(f"Expected [N, 40] features, found {source_features.shape}")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = DeepLOB().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_mid = (source_features[:, 0].astype(np.float64) + source_features[:, 2]) / 2.0
    returns = forward_returns(raw_mid, sessions, args.horizon_snapshots)
    checkpoint_alpha = float(checkpoint["labeling"]["alpha"])
    alpha = checkpoint_alpha if args.evaluation_alpha is None else args.evaluation_alpha
    labels = labels_from_returns(returns, alpha)
    cutoff = int(len(labels) * args.cutoff_fraction)
    target_indices = valid_target_indices(
        sessions, labels, args.sequence_length, cutoff
    )
    transformed = transform_7709(
        source_features, float(metadata["feature_scale"]), checkpoint
    )
    dataset = WindowDataset(transformed, labels, target_indices, args.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    torch.set_float32_matmul_precision("high")
    model_metrics, _ = evaluate(model, loader, device)
    evaluation_labels = labels[target_indices]
    result = {
        "experiment": "Public LOBSTER 5-stock checkpoint zero-shot transfer to HK 7709",
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "preprocessing": {
            "source_metadata": metadata,
            "price_transform": checkpoint["preprocessing"]["price_transform"],
            "size_transform": "restore raw HKEX quantity by multiplying feature_scale, then log1p",
            "zscore_source": "LOBSTER chronological training segments; no 7709 fitting",
            "sequence_length": args.sequence_length,
            "horizon_snapshots": args.horizon_snapshots,
            "approximate_horizon_book_update_groups": args.horizon_snapshots
            * int(metadata["snapshot_every_book_update_groups"]),
            "checkpoint_label_alpha": checkpoint_alpha,
            "evaluation_label_alpha": alpha,
            "evaluation_alpha_overridden": args.evaluation_alpha is not None,
            "evaluation_cutoff_fraction": args.cutoff_fraction,
            "label_counts": np.bincount(evaluation_labels, minlength=3).tolist(),
        },
        "model": model_metrics,
        "majority_baseline": majority_baseline(evaluation_labels),
        "limitations": [
            "The source model saw one NASDAQ day, while 7709 is a different exchange, instrument, tick regime, and date.",
            "Ten reconstructed 7709 snapshots only approximate 100 raw LOBSTER events.",
            "No 7709 observations were used to fit the checkpoint or its normalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
