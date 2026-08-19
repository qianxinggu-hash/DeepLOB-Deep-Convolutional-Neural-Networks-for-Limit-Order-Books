#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "torch>=2.1",
# ]
# ///
"""Train DeepLOB on the public 10-level LOBSTER NASDAQ sample files.

The samples are one trading day (2012-06-21), not the proprietary one-year
London Stock Exchange data used by the DeepLOB paper.  The script downloads
the public files from a Hugging Face mirror, performs chronological splits,
fits preprocessing and label thresholds on training data only, evaluates the
best validation checkpoint on a held-out tail, and stores all inference
metadata alongside the PyTorch state dict.

Example:
    uv run --python 3.12 train_lobster_samples.py --epochs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "lobster_samples"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "lobster_samples"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "deeplob_lobster_5stocks_best.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "output" / "results" / "lobster_5stocks_training.json"
SYMBOLS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
DATE = "2012-06-21"
START = "34200000"
END = "57600000"
LEVELS = 10
NUM_FEATURES = 40
NUM_CLASSES = 3
CLASS_NAMES = ("down", "stationary", "up")
OFFICIAL_SAMPLE_PAGE = "https://data.lobsterdata.com/info/DataSamples.php"
MIRROR_REPOSITORY = "https://huggingface.co/datasets/totalorganfailure/lobster-data"


def source_path(symbol: str) -> str:
    folder = f"LOBSTER_SampleFile_{symbol}_{DATE}_{LEVELS}"
    filename = f"{symbol}_{DATE}_{START}_{END}_orderbook_{LEVELS}.csv"
    return f"{folder}/{filename}"


def source_url(symbol: str) -> str:
    return f"{MIRROR_REPOSITORY}/resolve/main/{source_path(symbol)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepLOB on public LOBSTER 10-level samples"
    )
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=list(SYMBOLS))
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--horizon-events", type=int, default=100)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=20,
        help="Use every Nth eligible target while each input retains 100 raw events",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--stationary-share",
        type=float,
        default=1 / 3,
        help="Training-set share inside +/- alpha used to fit the label threshold",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.sequence_length < 1 or args.horizon_events < 1 or args.sample_stride < 1:
        parser.error("sequence length, horizon, and stride must be positive")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1 - args.train_fraction:
        parser.error("validation fraction leaves no test data")
    if not 0 < args.stationary_share < 1:
        parser.error("--stationary-share must be between 0 and 1")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"Using downloaded file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"Downloading {url}")
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "DeepLOB-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    partial.replace(destination)
    print(
        f"Downloaded {destination.name}: {destination.stat().st_size / 1e6:.1f} MB "
        f"in {time.perf_counter() - started:.1f}s"
    )


def transformed_paths(processed_dir: Path, symbol: str) -> tuple[Path, Path, Path]:
    return (
        processed_dir / f"{symbol}_{DATE}_features.npy",
        processed_dir / f"{symbol}_{DATE}_mid_prices.npy",
        processed_dir / f"{symbol}_{DATE}_quality.json",
    )


def preprocess_orderbook(
    raw_path: Path,
    processed_dir: Path,
    symbol: str,
    force: bool,
) -> dict[str, object]:
    feature_path, mid_path, quality_path = transformed_paths(processed_dir, symbol)
    if all(path.is_file() for path in (feature_path, mid_path, quality_path)) and not force:
        return json.loads(quality_path.read_text())

    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading and checking {symbol}: {raw_path}")
    started = time.perf_counter()
    orderbook = np.loadtxt(raw_path, delimiter=",", dtype=np.float32)
    if orderbook.ndim != 2 or orderbook.shape[1] != NUM_FEATURES:
        raise ValueError(f"{symbol}: expected [N, 40], found {orderbook.shape}")

    price_columns = np.arange(0, NUM_FEATURES, 2)
    size_columns = np.arange(1, NUM_FEATURES, 2)
    finite_rows = np.isfinite(orderbook).all(axis=1)
    positive_prices = (orderbook[:, price_columns] > 0).all(axis=1)
    nonnegative_sizes = (orderbook[:, size_columns] >= 0).all(axis=1)
    uncrossed = orderbook[:, 0] > orderbook[:, 2]
    asks = orderbook[:, np.arange(0, NUM_FEATURES, 4)]
    bids = orderbook[:, np.arange(2, NUM_FEATURES, 4)]
    ask_monotone = (np.diff(asks, axis=1) >= 0).all(axis=1)
    bid_monotone = (np.diff(bids, axis=1) <= 0).all(axis=1)
    valid = finite_rows & positive_prices & nonnegative_sizes & uncrossed
    valid &= ask_monotone & bid_monotone
    invalid_count = int((~valid).sum())
    if invalid_count:
        orderbook = orderbook[valid]

    mid_prices = ((orderbook[:, 0].astype(np.float64) + orderbook[:, 2]) / 2.0)
    # Cross-instrument causal transform: prices become distance from the current
    # mid in basis points; volumes use log1p.  Global z-score parameters are fit
    # later using only the chronological training portions.
    orderbook[:, price_columns] = (
        orderbook[:, price_columns] / mid_prices[:, None] - 1.0
    ) * 10_000.0
    orderbook[:, size_columns] = np.log1p(orderbook[:, size_columns])
    np.save(feature_path, orderbook.astype(np.float32, copy=False))
    np.save(mid_path, mid_prices.astype(np.float64, copy=False))

    quality = {
        "symbol": symbol,
        "rows": int(len(orderbook)),
        "columns": int(orderbook.shape[1]),
        "invalid_rows_removed": invalid_count,
        "finite": bool(np.isfinite(orderbook).all()),
        "best_ask_strictly_above_best_bid": bool(uncrossed[valid].all()),
        "ask_levels_monotone": bool(ask_monotone[valid].all()),
        "bid_levels_monotone": bool(bid_monotone[valid].all()),
        "mid_price_min_fixed_point": float(mid_prices.min()),
        "mid_price_max_fixed_point": float(mid_prices.max()),
        "processed_feature_path": str(feature_path.resolve()),
        "processed_mid_price_path": str(mid_path.resolve()),
        "duration_seconds": time.perf_counter() - started,
    }
    quality_path.write_text(json.dumps(quality, indent=2) + "\n")
    print(f"Prepared {symbol}: {len(orderbook):,} rows in {quality['duration_seconds']:.1f}s")
    return quality


def split_bounds(rows: int, train_fraction: float, validation_fraction: float) -> dict[str, tuple[int, int]]:
    train_end = int(math.floor(rows * train_fraction))
    validation_end = int(math.floor(rows * (train_fraction + validation_fraction)))
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, rows),
    }


def fit_normalization(
    features_by_symbol: dict[str, np.ndarray],
    bounds_by_symbol: dict[str, dict[str, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_features = np.zeros(NUM_FEATURES, dtype=np.float64)
    sum_squares = np.zeros(NUM_FEATURES, dtype=np.float64)
    for symbol, features in features_by_symbol.items():
        start, stop = bounds_by_symbol[symbol]["train"]
        train = np.asarray(features[start:stop], dtype=np.float64)
        total += len(train)
        sum_features += train.sum(axis=0)
        sum_squares += np.square(train).sum(axis=0)
    mean = sum_features / total
    variance = np.maximum(sum_squares / total - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def forward_returns(mid_prices: np.ndarray, start: int, stop: int, horizon: int) -> np.ndarray:
    """DeepLOB Equation (3): mean of next h mid-prices vs current mid-price."""
    result = np.full(stop - start, np.nan, dtype=np.float64)
    values = np.asarray(mid_prices[start:stop], dtype=np.float64)
    if len(values) <= horizon:
        return result
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    local = np.arange(len(values) - horizon)
    future_mean = (cumulative[local + horizon + 1] - cumulative[local + 1]) / horizon
    result[local] = future_mean / values[local] - 1.0
    return result


def eligible_local_indices(length: int, sequence_length: int, horizon: int, stride: int) -> np.ndarray:
    first = sequence_length - 1
    stop = length - horizon
    if stop <= first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, stop, stride, dtype=np.int64)


@dataclass
class PreparedSplit:
    symbol: str
    split: str
    features: np.ndarray
    offset: int
    local_targets: np.ndarray
    labels: np.ndarray


class LOBSTERSplitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        prepared: PreparedSplit,
        sequence_length: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.prepared = prepared
        self.sequence_length = sequence_length
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.prepared.local_targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_target = int(self.prepared.local_targets[index])
        absolute_target = self.prepared.offset + local_target
        start = absolute_target - self.sequence_length + 1
        window = np.asarray(
            self.prepared.features[start : absolute_target + 1], dtype=np.float32
        )
        normalized = np.ascontiguousarray((window - self.mean) / self.std)
        target = int(self.prepared.labels[index])
        return torch.from_numpy(normalized).unsqueeze(0), torch.tensor(target)


def make_prepared_splits(
    symbols: Iterable[str],
    features_by_symbol: dict[str, np.ndarray],
    mids_by_symbol: dict[str, np.ndarray],
    bounds_by_symbol: dict[str, dict[str, tuple[int, int]]],
    sequence_length: int,
    horizon: int,
    stride: int,
    stationary_share: float,
) -> tuple[dict[str, list[PreparedSplit]], float]:
    return_cache: dict[tuple[str, str], np.ndarray] = {}
    calibration_returns: list[np.ndarray] = []
    for symbol in symbols:
        start, stop = bounds_by_symbol[symbol]["train"]
        returns = forward_returns(mids_by_symbol[symbol], start, stop, horizon)
        return_cache[(symbol, "train")] = returns
        targets = eligible_local_indices(len(returns), sequence_length, horizon, stride)
        calibration_returns.append(returns[targets])
    alpha = float(np.quantile(np.abs(np.concatenate(calibration_returns)), stationary_share))

    prepared_by_split: dict[str, list[PreparedSplit]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for symbol in symbols:
        for split in prepared_by_split:
            start, stop = bounds_by_symbol[symbol][split]
            returns = return_cache.get((symbol, split))
            if returns is None:
                returns = forward_returns(mids_by_symbol[symbol], start, stop, horizon)
            targets = eligible_local_indices(len(returns), sequence_length, horizon, stride)
            target_returns = returns[targets]
            labels = np.where(
                target_returns < -alpha,
                0,
                np.where(target_returns > alpha, 2, 1),
            ).astype(np.int64)
            prepared_by_split[split].append(
                PreparedSplit(symbol, split, features_by_symbol[symbol], start, targets, labels)
            )
    return prepared_by_split, alpha


def make_loader(
    prepared: list[PreparedSplit],
    sequence_length: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    datasets = [LOBSTERSplitDataset(item, sequence_length, mean, std) for item in prepared]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=generator,
        pin_memory=False,
    )


def confusion_metrics(confusion: np.ndarray) -> dict[str, object]:
    total = int(confusion.sum())
    per_class: list[dict[str, object]] = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        support = int(confusion[index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {"class": name, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    return {
        "samples": total,
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 0,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device, dtype=torch.float32)
            targets_device = targets.to(device, dtype=torch.int64)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets_device)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            truth = targets.numpy()
            confusion += np.bincount(truth * NUM_CLASSES + predictions, minlength=9).reshape(3, 3)
            total_loss += loss.item() * len(truth)
            if training and log_every and batch_number % log_every == 0:
                elapsed = time.perf_counter() - started
                seen = int(confusion.sum())
                print(f"  batch {batch_number}/{len(loader)} ({seen / elapsed:.0f} samples/s)")
    metrics = confusion_metrics(confusion)
    metrics["duration_seconds"] = time.perf_counter() - started
    return total_loss / max(1, int(confusion.sum())), metrics


def label_summary(prepared: list[PreparedSplit]) -> dict[str, object]:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    by_symbol: dict[str, list[int]] = {}
    for item in prepared:
        item_counts = np.bincount(item.labels, minlength=NUM_CLASSES)
        counts += item_counts
        by_symbol[item.symbol] = item_counts.tolist()
    return {
        "counts": counts.tolist(),
        "shares": (counts / counts.sum()).tolist(),
        "by_symbol": by_symbol,
    }


def majority_baseline(prepared: list[PreparedSplit], majority_class: int) -> dict[str, object]:
    truth_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for item in prepared:
        truth_counts += np.bincount(item.labels, minlength=NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[:, majority_class] = truth_counts
    result = confusion_metrics(confusion)
    result["predicted_class"] = CLASS_NAMES[majority_class]
    return result


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    source_records: list[dict[str, object]] = []
    quality_records: list[dict[str, object]] = []
    for symbol in args.symbols:
        filename = Path(source_path(symbol)).name
        raw_path = args.raw_dir / filename
        url = source_url(symbol)
        download_file(url, raw_path, args.force_download)
        quality = preprocess_orderbook(
            raw_path, args.processed_dir, symbol, args.force_preprocess
        )
        quality_records.append(quality)
        source_records.append(
            {
                "symbol": symbol,
                "date": DATE,
                "levels": LEVELS,
                "raw_path": str(raw_path.resolve()),
                "download_url": url,
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
        )

    features_by_symbol: dict[str, np.ndarray] = {}
    mids_by_symbol: dict[str, np.ndarray] = {}
    bounds_by_symbol: dict[str, dict[str, tuple[int, int]]] = {}
    for symbol in args.symbols:
        feature_path, mid_path, _ = transformed_paths(args.processed_dir, symbol)
        features_by_symbol[symbol] = np.load(feature_path, mmap_mode="r")
        mids_by_symbol[symbol] = np.load(mid_path, mmap_mode="r")
        bounds_by_symbol[symbol] = split_bounds(
            len(features_by_symbol[symbol]), args.train_fraction, args.validation_fraction
        )

    mean, std = fit_normalization(features_by_symbol, bounds_by_symbol)
    prepared, alpha = make_prepared_splits(
        args.symbols,
        features_by_symbol,
        mids_by_symbol,
        bounds_by_symbol,
        args.sequence_length,
        args.horizon_events,
        args.sample_stride,
        args.stationary_share,
    )
    summaries = {split: label_summary(items) for split, items in prepared.items()}
    for split in ("train", "validation", "test"):
        print(f"{split}: {sum(summaries[split]['counts']):,} samples, labels={summaries[split]['counts']}")
    print(f"Training-only alpha: {alpha:.10g}")

    train_loader = make_loader(
        prepared["train"], args.sequence_length, mean, std, args.batch_size,
        True, args.num_workers, args.seed,
    )
    validation_loader = make_loader(
        prepared["validation"], args.sequence_length, mean, std, args.batch_size,
        False, args.num_workers, args.seed,
    )
    test_loader = make_loader(
        prepared["test"], args.sequence_length, mean, std, args.batch_size,
        False, args.num_workers, args.seed,
    )

    train_counts = np.asarray(summaries["train"]["counts"], dtype=np.float64)
    class_weights = train_counts.sum() / (NUM_CLASSES * train_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = DeepLOB().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_val_loss = math.inf
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.log_every
        )
        validation_loss, validation_metrics = run_epoch(
            model, validation_loader, criterion, device
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        improved = validation_loss < best_val_loss
        if improved:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_name": "DeepLOB",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_loss": validation_loss,
                    "config": serializable_args(args),
                    "class_names": list(CLASS_NAMES),
                    "preprocessing": {
                        "feature_order": "ask_price,ask_size,bid_price,bid_size repeated for levels 1..10",
                        "price_transform": "(price / current_mid_price - 1) * 10000 basis points",
                        "size_transform": "log1p(size)",
                        "zscore_mean": mean.tolist(),
                        "zscore_std": std.tolist(),
                        "zscore_fit_scope": "pooled chronological training segments only",
                    },
                    "labeling": {
                        "method": "DeepLOB Equation (3): mean of next h mid-prices / current mid-price - 1",
                        "horizon_events": args.horizon_events,
                        "alpha": alpha,
                        "alpha_fit_scope": "pooled chronological training targets only",
                        "classes": {"0": "down", "1": "stationary", "2": "up"},
                    },
                    "data_sources": source_records,
                    "dataset_note": "Public one-day NASDAQ LOBSTER samples; not the paper's one-year LSE data.",
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
        marker = " saved" if improved else ""
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, "
            f"train_macro_f1={train_metrics['macro_f1']:.4f}, "
            f"val_loss={validation_loss:.4f}, "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}{marker}"
        )
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} unimproved epochs")
            break

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, device)
    majority_class = int(train_counts.argmax())
    baseline = majority_baseline(prepared["test"], majority_class)
    print(
        f"Test: loss={test_loss:.4f}, accuracy={test_metrics['accuracy']:.4f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    results = {
        "experiment": "DeepLOB trained on public LOBSTER NASDAQ samples",
        "status": "completed",
        "dataset_scope": {
            "symbols": args.symbols,
            "date": DATE,
            "levels": LEVELS,
            "official_sample_page": OFFICIAL_SAMPLE_PAGE,
            "mirror_repository": MIRROR_REPOSITORY,
            "not_equivalent_to_paper_lse_data": True,
        },
        "sources": source_records,
        "data_quality": quality_records,
        "split_method": "per-symbol chronological 70%/15%/15%; windows and targets stay inside each split",
        "split_bounds": {
            symbol: {name: list(bounds) for name, bounds in splits.items()}
            for symbol, splits in bounds_by_symbol.items()
        },
        "label_summaries": summaries,
        "preprocessing": checkpoint["preprocessing"],
        "labeling": checkpoint["labeling"],
        "training": {
            "device": str(device),
            "class_weights": class_weights.tolist(),
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_loss": float(checkpoint["best_validation_loss"]),
            "history": history,
        },
        "test": {"loss": test_loss, **test_metrics},
        "majority_baseline": baseline,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "limitations": [
            "All five symbols are from one NASDAQ trading day, so this is a pipeline and transfer baseline rather than a substitute for one year of LSE data.",
            "The sample stride reduces correlated target windows for runtime, although every input window still contains consecutive raw order-book events.",
            "The normalization differs from the paper because five preceding trading days are unavailable in the public sample.",
        ],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    # Enrich the checkpoint with the completed audit trail while keeping the
    # optimizer state so it can be used for continued training.
    checkpoint["training_history"] = history
    checkpoint["test_metrics"] = test_metrics
    checkpoint["majority_baseline"] = baseline
    checkpoint["results_path"] = str(args.results.resolve())
    torch.save(checkpoint, args.checkpoint)
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Results: {args.results.resolve()}")


if __name__ == "__main__":
    main()
