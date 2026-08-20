#!/usr/bin/env python3
"""Frozen-checkpoint classification reports on train and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from reproduce_pytorch import DeepLOB as FIDeepLOB
from reproduce_pytorch import FI2010Windows, load_fi2010, predict
from train_lobster_samples import (
    CLASS_NAMES,
    DATE,
    DEFAULT_CHECKPOINT as LOBSTER_CHECKPOINT,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    SYMBOLS,
    download_file,
    fit_normalization,
    make_loader,
    make_prepared_splits,
    preprocess_orderbook,
    source_path,
    source_url,
    split_bounds,
    transformed_paths,
)
from train_pytorch_mac import DeepLOB as LobsterDeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fi-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/full_cpu/best_deeplob_state.pt")
    parser.add_argument("--fi-data-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--lobster-checkpoint", type=Path, default=LOBSTER_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/results/fi2010_lobster_split_reports.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="cpu")
    parser.add_argument("--skip-fi2010", action="store_true")
    parser.add_argument("--skip-lobster", action="store_true")
    return parser.parse_args()


def report_dict(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return classification_report(
        truth,
        prediction,
        labels=[0, 1, 2],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
        digits=4,
    )


def print_split(title: str, report: dict[str, object]) -> None:
    print(f"\n=== {title} ===", flush=True)
    header = f"{'':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    print(header, flush=True)
    for name in CLASS_NAMES:
        row = report[name]
        print(
            f"{name:<12} {row['precision']:10.4f} {row['recall']:10.4f} "
            f"{row['f1-score']:10.4f} {int(row['support']):10d}",
            flush=True,
        )
    print(
        f"{'accuracy':<12} {'':>10} {'':>10} {report['accuracy']:10.4f} "
        f"{int(report['macro avg']['support']):10d}",
        flush=True,
    )
    macro = report["macro avg"]
    print(
        f"{'macro avg':<12} {macro['precision']:10.4f} {macro['recall']:10.4f} "
        f"{macro['f1-score']:10.4f} {int(macro['support']):10d}",
        flush=True,
    )


def evaluate_fi2010(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    print("Evaluating FI-2010 frozen checkpoint on train and test...", flush=True)
    train_matrix, _, test_matrix = load_fi2010(args.fi_data_dir)
    train_loader = DataLoader(
        FI2010Windows(train_matrix, horizon_index=4, window=100),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        FI2010Windows(test_matrix, horizon_index=4, window=100),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    del train_matrix, test_matrix

    model = FIDeepLOB().to(device)
    state = torch.load(args.fi_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    result = {
        "checkpoint": str(args.fi_checkpoint.resolve()),
        "horizon_index": 4,
        "window": 100,
    }
    for split, loader in (("train", train_loader), ("test", test_loader)):
        truth, prediction = predict(model, loader, device)
        report = report_dict(truth, prediction)
        result[split] = {
            "samples": int(len(truth)),
            "classification_report": report,
        }
        print_split(f"FI-2010 {split}", report)
    return result


def evaluate_lobster(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    print("Evaluating LOBSTER frozen checkpoint on train and test...", flush=True)
    checkpoint = torch.load(args.lobster_checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    labeling = checkpoint["labeling"]
    preprocessing = checkpoint["preprocessing"]
    mean = np.asarray(preprocessing["zscore_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["zscore_std"], dtype=np.float32)
    alpha = float(labeling["alpha"])

    features_by_symbol: dict[str, np.ndarray] = {}
    mids_by_symbol: dict[str, np.ndarray] = {}
    bounds_by_symbol: dict[str, dict[str, tuple[int, int]]] = {}
    for symbol in SYMBOLS:
        filename = Path(source_path(symbol)).name
        raw_path = DEFAULT_RAW_DIR / filename
        download_file(source_url(symbol), raw_path, force=False)
        preprocess_orderbook(raw_path, DEFAULT_PROCESSED_DIR, symbol, force=False)
        feature_path, mid_path, _ = transformed_paths(DEFAULT_PROCESSED_DIR, symbol)
        features_by_symbol[symbol] = np.load(feature_path, mmap_mode="r")
        mids_by_symbol[symbol] = np.load(mid_path, mmap_mode="r")
        bounds_by_symbol[symbol] = split_bounds(
            len(features_by_symbol[symbol]),
            float(config["train_fraction"]),
            float(config["validation_fraction"]),
        )

    prepared, recomputed_alpha = make_prepared_splits(
        SYMBOLS,
        features_by_symbol,
        mids_by_symbol,
        bounds_by_symbol,
        int(config["sequence_length"]),
        int(config["horizon_events"]),
        int(config["sample_stride"]),
        float(config["stationary_share"]),
    )
    # Rebuild labels with the checkpoint alpha so train/test match the saved model.
    if abs(recomputed_alpha - alpha) > 1e-12:
        print(
            f"Using checkpoint alpha={alpha:.10g} instead of recomputed {recomputed_alpha:.10g}",
            flush=True,
        )
        from train_lobster_samples import PreparedSplit, eligible_local_indices, forward_returns

        rebuilt: dict[str, list] = {"train": [], "validation": [], "test": []}
        for symbol in SYMBOLS:
            for split in rebuilt:
                start, stop = bounds_by_symbol[symbol][split]
                returns = forward_returns(mids_by_symbol[symbol], start, stop, int(config["horizon_events"]))
                targets = eligible_local_indices(
                    len(returns),
                    int(config["sequence_length"]),
                    int(config["horizon_events"]),
                    int(config["sample_stride"]),
                )
                values = returns[targets]
                labels = np.where(values < -alpha, 0, np.where(values > alpha, 2, 1)).astype(np.int64)
                rebuilt[split].append(
                    PreparedSplit(symbol, split, features_by_symbol[symbol], start, targets, labels)
                )
        prepared = rebuilt

    train_mean, train_std = fit_normalization(features_by_symbol, bounds_by_symbol)
    if not np.allclose(train_mean, mean, atol=1e-5) or not np.allclose(train_std, std, atol=1e-5):
        print("Using checkpoint z-score statistics rather than freshly fitted values.", flush=True)

    model = LobsterDeepLOB().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    result = {
        "checkpoint": str(args.lobster_checkpoint.resolve()),
        "alpha": alpha,
        "sequence_length": int(config["sequence_length"]),
        "horizon_events": int(config["horizon_events"]),
        "sample_stride": int(config["sample_stride"]),
    }
    for split in ("train", "test"):
        loader = make_loader(
            prepared[split],
            int(config["sequence_length"]),
            mean,
            std,
            args.batch_size,
            False,
            0,
            int(config["seed"]),
        )
        truth, prediction = predict(model, loader, device)
        report = report_dict(truth, prediction)
        result[split] = {
            "samples": int(len(truth)),
            "classification_report": report,
        }
        print_split(f"LOBSTER {split}", report)
    return result


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    print(f"device={device}", flush=True)
    output: dict[str, object] = {"device": str(device)}
    if not args.skip_fi2010:
        output["fi2010"] = evaluate_fi2010(args, device)
    if not args.skip_lobster:
        output["lobster"] = evaluate_lobster(args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
