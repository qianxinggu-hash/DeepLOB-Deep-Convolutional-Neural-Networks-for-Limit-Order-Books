#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "sortedcontainers>=2.4",
#     "torch>=2.1",
# ]
# ///
"""Train and evaluate DeepLOB on the HK 07709 order-event files.

The experiment follows the LSE setup in the DeepLOB paper where the available
data permit it:

* reconstruct a 10-level book and sample it after every 10 update timestamps;
* use the most recent 100 book states as one input;
* use Equation (4), comparing the mean of the previous k mid-prices with the
  mean of the next k mid-prices, to create down/stationary/up labels;
* z-score each day using up to five preceding trading days;
* select k on a chronological validation day, then evaluate exactly once on
  the held-out 2026-08-07 test day.

The earliest available day has no preceding history.  It is normalised using
its own feature statistics as a documented warm start; every later day uses
only preceding dates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from prepare_hkex_lob import RECONSTRUCTION_VERSION, reconstruct
from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 3
NUM_FEATURES = 40
CLASS_NAMES = ("down", "stationary", "up")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronological DeepLOB experiment on HK 07709"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=PROJECT_ROOT / "7709_tickdata"
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed/7709"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints/7709"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "output/results/7709_deeplob_chronological.json",
    )
    parser.add_argument("--security-id", type=int, default=7709)
    parser.add_argument("--test-date", default="2026-08-07")
    parser.add_argument(
        "--validation-date",
        default=None,
        help="default: latest available date strictly before the test date",
    )
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument(
        "--stationary-share",
        type=float,
        default=1 / 3,
        help="training share inside +/- alpha",
    )
    parser.add_argument(
        "--train-target-stride",
        type=int,
        default=5,
        help="subsample highly overlapping training targets only",
    )
    parser.add_argument(
        "--validation-target-stride",
        type=int,
        default=5,
        help="subsample validation targets during model selection; test remains stride 1",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--force-reconstruct", action="store_true")
    args = parser.parse_args()

    positive_values = (
        args.snapshot_every,
        args.sequence_length,
        args.train_target_stride,
        args.validation_target_stride,
        args.epochs,
        args.batch_size,
    )
    if any(value < 1 for value in positive_values) or any(k < 1 for k in args.ks):
        parser.error("sampling, sequence, k, epoch, and batch values must be positive")
    if not 0.0 < args.stationary_share < 1.0:
        parser.error("--stationary-share must be between 0 and 1")
    if args.patience < 0 or args.num_workers < 0:
        parser.error("--patience and --num-workers cannot be negative")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def date_from_path(path: Path) -> str:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not infer YYYY-MM-DD date from {path}")
    return match.group(1)


def load_metadata(path: Path) -> dict[str, object] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(str(archive["metadata"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def compatible_cache(
    processed_dir: Path, raw_path: Path, snapshot_every: int, security_id: int
) -> Path | None:
    candidates = list(processed_dir.glob("*.npz"))
    candidates += list((PROJECT_ROOT / "data/processed").glob("hk07709*.npz"))
    for candidate in candidates:
        metadata = load_metadata(candidate)
        if metadata is None:
            continue
        if Path(str(metadata.get("source", ""))).name != raw_path.name:
            continue
        if int(metadata.get("snapshot_every_book_update_groups", -1)) != snapshot_every:
            continue
        if int(metadata.get("security_id", -1)) != security_id:
            continue
        if int(metadata.get("reconstruction_version", 1)) != RECONSTRUCTION_VERSION:
            continue
        return candidate
    return None


def prepare_day(
    raw_path: Path,
    processed_dir: Path,
    snapshot_every: int,
    security_id: int,
    force: bool,
) -> tuple[Path, dict[str, object]]:
    date = date_from_path(raw_path)
    destination = processed_dir / f"hk07709_{date}_s{snapshot_every}_lob.npz"
    cache = None if force else compatible_cache(
        processed_dir, raw_path, snapshot_every, security_id
    )
    if cache is not None:
        metadata = load_metadata(cache)
        assert metadata is not None
        print(f"Using reconstructed {date}: {cache} ({metadata['snapshot_count']:,} states)", flush=True)
        return cache, metadata

    print(f"Reconstructing {date}: {raw_path}", flush=True)
    started = time.perf_counter()
    features, send_times, sessions, metadata = reconstruct(
        raw_path, security_id, snapshot_every, 10
    )
    if not len(features):
        raise RuntimeError(f"No continuous-session snapshots reconstructed from {raw_path}")
    metadata["duration_seconds"] = time.perf_counter() - started
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=features,
        send_times=send_times,
        sessions=sessions,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    print(
        f"Prepared {date}: {len(features):,} states in {metadata['duration_seconds']:.1f}s",
        flush=True,
    )
    return destination, metadata


@dataclass
class DayData:
    date: str
    path: Path
    features: np.ndarray
    sessions: np.ndarray
    metadata: dict[str, object]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    normalization_sources: tuple[str, ...] = ()

    @property
    def mid_prices(self) -> np.ndarray:
        return (
            self.features[:, 0].astype(np.float64)
            + self.features[:, 2].astype(np.float64)
        ) / 2.0


def pooled_mean_std(days: list[DayData]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(NUM_FEATURES, dtype=np.float64)
    total_square = np.zeros(NUM_FEATURES, dtype=np.float64)
    for day in days:
        values = np.asarray(day.features, dtype=np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def assign_causal_normalizers(days: list[DayData]) -> None:
    for index, day in enumerate(days):
        history = days[max(0, index - 5) : index]
        if not history:
            history = [day]
            day.normalization_sources = (f"{day.date} (warm start)",)
        else:
            day.normalization_sources = tuple(item.date for item in history)
        day.mean, day.std = pooled_mean_std(history)


def equation4_returns(day: DayData, k: int) -> np.ndarray:
    """Symmetric k-state implementation of DeepLOB Equation (4).

    The paper prints a k denominator with a 0..k past sum.  We use k actual
    observations on both sides: [t-k+1, t] and [t+1, t+k].
    """
    mids = day.mid_prices
    result = np.full(len(mids), np.nan, dtype=np.float64)
    for session in np.unique(day.sessions):
        positions = np.flatnonzero(day.sessions == session)
        if len(positions) < 2 * k:
            continue
        if np.any(np.diff(positions) != 1):
            raise ValueError(f"{day.date}: session {session} is not contiguous")
        values = mids[positions]
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        local = np.arange(k - 1, len(values) - k, dtype=np.int64)
        past_mean = (cumulative[local + 1] - cumulative[local - k + 1]) / k
        future_mean = (cumulative[local + k + 1] - cumulative[local + 1]) / k
        result[positions[local]] = future_mean / past_mean - 1.0
    return result


def eligible_targets(
    day: DayData,
    returns: np.ndarray,
    sequence_length: int,
    stride: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for session in np.unique(day.sessions):
        positions = np.flatnonzero(day.sessions == session)
        first = sequence_length - 1
        candidates = positions[first::stride]
        valid = np.isfinite(returns[candidates])
        selected.append(candidates[valid])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected)


@dataclass
class PreparedDay:
    day: DayData
    indices: np.ndarray
    labels: np.ndarray


def prepare_labels(
    days: list[DayData],
    train_dates: set[str],
    validation_date: str,
    k: int,
    sequence_length: int,
    train_stride: int,
    validation_stride: int,
    stationary_share: float,
) -> tuple[dict[str, PreparedDay], float]:
    returns_by_date = {day.date: equation4_returns(day, k) for day in days}
    train_returns: list[np.ndarray] = []
    indices_by_date: dict[str, np.ndarray] = {}
    for day in days:
        if day.date in train_dates:
            stride = train_stride
        elif day.date == validation_date:
            stride = validation_stride
        else:
            stride = 1
        indices = eligible_targets(
            day, returns_by_date[day.date], sequence_length, stride
        )
        indices_by_date[day.date] = indices
        if day.date in train_dates:
            train_returns.append(returns_by_date[day.date][indices])
    if not train_returns or not any(len(values) for values in train_returns):
        raise RuntimeError(f"No eligible training targets for k={k}")
    alpha = float(
        np.quantile(np.abs(np.concatenate(train_returns)), stationary_share)
    )

    prepared: dict[str, PreparedDay] = {}
    for day in days:
        indices = indices_by_date[day.date]
        target_returns = returns_by_date[day.date][indices]
        labels = np.where(
            target_returns < -alpha,
            0,
            np.where(target_returns > alpha, 2, 1),
        ).astype(np.int64)
        prepared[day.date] = PreparedDay(day, indices, labels)
    return prepared, alpha


class DayWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, prepared: PreparedDay, sequence_length: int) -> None:
        self.prepared = prepared
        self.sequence_length = sequence_length
        if prepared.day.mean is None or prepared.day.std is None:
            raise ValueError("Normalisation parameters were not assigned")

    def __len__(self) -> int:
        return len(self.prepared.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = int(self.prepared.indices[index])
        start = target - self.sequence_length + 1
        window = np.asarray(
            self.prepared.day.features[start : target + 1], dtype=np.float32
        )
        normalized = np.ascontiguousarray(
            (window - self.prepared.day.mean) / self.prepared.day.std
        )
        label = int(self.prepared.labels[index])
        return torch.from_numpy(normalized).unsqueeze(0), torch.tensor(label)


def make_loader(
    items: list[PreparedDay],
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    datasets = [DayWindowDataset(item, sequence_length) for item in items]
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
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
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
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, (inputs, targets) in enumerate(loader, start=1):
            inputs_device = inputs.to(device, dtype=torch.float32)
            targets_device = targets.to(device, dtype=torch.int64)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs_device)
            loss = criterion(logits, targets_device)
            if training:
                loss.backward()
                optimizer.step()
            prediction = logits.argmax(dim=1).detach().cpu().numpy()
            truth = targets.numpy()
            confusion += np.bincount(
                truth * NUM_CLASSES + prediction, minlength=NUM_CLASSES**2
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            total_loss += loss.item() * len(truth)
            if training and log_every and batch_number % log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"    batch {batch_number}/{len(loader)} "
                    f"({int(confusion.sum()) / elapsed:.0f} samples/s)",
                    flush=True,
                )
    metrics = confusion_metrics(confusion)
    metrics["duration_seconds"] = time.perf_counter() - started
    return total_loss / max(1, int(confusion.sum())), metrics


def label_summary(items: list[PreparedDay]) -> dict[str, object]:
    total = np.zeros(NUM_CLASSES, dtype=np.int64)
    by_date: dict[str, list[int]] = {}
    for item in items:
        counts = np.bincount(item.labels, minlength=NUM_CLASSES)
        total += counts
        by_date[item.day.date] = counts.tolist()
    return {
        "counts": total.tolist(),
        "shares": (total / total.sum()).tolist(),
        "by_date": by_date,
    }


def majority_baseline(labels: np.ndarray, predicted_class: int) -> dict[str, object]:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[:, predicted_class] = counts
    result = confusion_metrics(confusion)
    result["predicted_class"] = CLASS_NAMES[predicted_class]
    return result


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    k: int,
    alpha: float,
    epoch: int,
    validation_loss: float,
    validation_metrics: dict[str, object],
    days: list[DayData],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model_name": "DeepLOB",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "config": serializable_args(args),
        "class_names": list(CLASS_NAMES),
        "labeling": {
            "method": "DeepLOB Equation (4), symmetric k-snapshot past/future means",
            "k": k,
            "alpha": alpha,
            "alpha_fit_scope": "training targets only",
        },
        "preprocessing": {
            "feature_order": "ask_price,ask_size,bid_price,bid_size repeated for levels 1..10",
            "snapshot_every_book_update_groups": args.snapshot_every,
            "sequence_length": args.sequence_length,
            "normalization": "per-feature z-score from up to five preceding available days",
            "normalizers": {
                day.date: {
                    "source_dates": list(day.normalization_sources),
                    "mean": day.mean.tolist() if day.mean is not None else None,
                    "std": day.std.tolist() if day.std is not None else None,
                }
                for day in days
            },
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    raw_files = sorted(args.raw_dir.glob("hk07709_*.csv"), key=date_from_path)
    if not raw_files:
        raise FileNotFoundError(f"No hk07709_*.csv files found under {args.raw_dir}")
    dates = [date_from_path(path) for path in raw_files]
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate trading dates in raw input files")
    if args.test_date not in dates:
        raise ValueError(f"Test date {args.test_date} is not among {dates}")
    dates_before_test = [date for date in dates if date < args.test_date]
    validation_date = args.validation_date or max(dates_before_test)
    if validation_date not in dates_before_test:
        raise ValueError("Validation date must exist and precede the test date")
    train_dates = [date for date in dates if date < validation_date]
    later_non_test = [date for date in dates if date > args.test_date]
    if later_non_test:
        raise ValueError(f"Dates after the test date are not assigned: {later_non_test}")
    if not train_dates:
        raise ValueError("At least one training date is required")
    print(
        f"Split: train={train_dates}, validation={validation_date}, test={args.test_date}",
        flush=True,
    )

    paths: dict[str, Path] = {}
    reconstruction_records: dict[str, dict[str, object]] = {}
    for raw_path in raw_files:
        date = date_from_path(raw_path)
        path, metadata = prepare_day(
            raw_path,
            args.processed_dir,
            args.snapshot_every,
            args.security_id,
            args.force_reconstruct,
        )
        paths[date] = path
        reconstruction_records[date] = metadata

    days: list[DayData] = []
    archives: list[np.lib.npyio.NpzFile] = []
    for date in dates:
        archive = np.load(paths[date], mmap_mode="r", allow_pickle=False)
        archives.append(archive)
        features = archive["features"].astype(np.float32, copy=False)
        sessions = archive["sessions"].astype(np.uint8, copy=False)
        if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
            raise ValueError(f"{date}: expected [N, 40], found {features.shape}")
        days.append(
            DayData(date, paths[date], features, sessions, reconstruction_records[date])
        )
    assign_causal_normalizers(days)

    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}", flush=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    candidates: list[dict[str, object]] = []
    prepared_cache: dict[int, dict[str, PreparedDay]] = {}

    for k in args.ks:
        seed_everything(args.seed)
        prepared, alpha = prepare_labels(
            days,
            set(train_dates),
            validation_date,
            k,
            args.sequence_length,
            args.train_target_stride,
            args.validation_target_stride,
            args.stationary_share,
        )
        prepared_cache[k] = prepared
        train_items = [prepared[date] for date in train_dates]
        validation_items = [prepared[validation_date]]
        summaries = {
            "train": label_summary(train_items),
            "validation": label_summary(validation_items),
        }
        print(
            f"k={k}: alpha={alpha:.10g}, train={summaries['train']['counts']}, "
            f"validation={summaries['validation']['counts']}",
            flush=True,
        )
        train_loader = make_loader(
            train_items,
            args.sequence_length,
            args.batch_size,
            True,
            args.num_workers,
            args.seed,
        )
        validation_loader = make_loader(
            validation_items,
            args.sequence_length,
            args.batch_size,
            False,
            args.num_workers,
            args.seed,
        )
        model = DeepLOB().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        checkpoint_path = args.checkpoint_dir / f"deeplob_7709_k{k}_best.pt"
        best_macro_f1 = -math.inf
        best_validation_loss = math.inf
        unimproved = 0
        history: list[dict[str, object]] = []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                args.log_every,
            )
            validation_loss, validation_metrics = run_epoch(
                model, validation_loader, criterion, device
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            score = float(validation_metrics["macro_f1"])
            improved = score > best_macro_f1 + 1e-8 or (
                abs(score - best_macro_f1) <= 1e-8
                and validation_loss < best_validation_loss
            )
            if improved:
                best_macro_f1 = score
                best_validation_loss = validation_loss
                unimproved = 0
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        args,
                        k,
                        alpha,
                        epoch,
                        validation_loss,
                        validation_metrics,
                        days,
                    ),
                    checkpoint_path,
                )
            else:
                unimproved += 1
            marker = " saved" if improved else ""
            print(
                f"  k={k} epoch={epoch:02d}: train_loss={train_loss:.4f}, "
                f"train_f1={train_metrics['macro_f1']:.4f}, "
                f"val_loss={validation_loss:.4f}, "
                f"val_f1={validation_metrics['macro_f1']:.4f}{marker}",
                flush=True,
            )
            if args.patience and unimproved >= args.patience:
                print(f"  k={k}: early stopping", flush=True)
                break

        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        candidates.append(
            {
                "k": k,
                "alpha": alpha,
                "checkpoint": str(checkpoint_path.resolve()),
                "best_epoch": int(saved["epoch"]),
                "validation_loss": float(saved["validation_loss"]),
                "validation": saved["validation_metrics"],
                "label_summaries": summaries,
                "history": history,
            }
        )
        del model, optimizer, train_loader, validation_loader
        if device.type == "mps":
            torch.mps.empty_cache()

    selected = max(
        candidates,
        key=lambda item: (
            float(item["validation"]["macro_f1"]),
            -float(item["validation_loss"]),
        ),
    )
    selected_k = int(selected["k"])
    print(
        f"Selected k={selected_k} by validation macro-F1; evaluating held-out test once",
        flush=True,
    )
    selected_checkpoint = Path(str(selected["checkpoint"]))
    saved = torch.load(selected_checkpoint, map_location=device, weights_only=True)
    model = DeepLOB().to(device)
    model.load_state_dict(saved["model_state_dict"])
    test_item = prepared_cache[selected_k][args.test_date]
    test_loader = make_loader(
        [test_item],
        args.sequence_length,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
    )
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, device)
    train_labels = np.concatenate(
        [prepared_cache[selected_k][date].labels for date in train_dates]
    )
    training_majority = int(np.bincount(train_labels, minlength=3).argmax())
    test_summary = label_summary([test_item])
    baseline = majority_baseline(test_item.labels, training_majority)
    print(
        f"Test k={selected_k}: loss={test_loss:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )

    result = {
        "experiment": "DeepLOB chronological HK 07709 evaluation",
        "status": "completed",
        "paper_protocol": {
            "levels": 10,
            "snapshot_sampling": f"one state per {args.snapshot_every} accepted update timestamps",
            "input_states": args.sequence_length,
            "label_method": "Equation (4), symmetric previous/future k-state mean mid-prices",
            "candidate_k": args.ks,
            "normalization": "z-score using up to five preceding available trading days",
            "model_selection": "validation macro-F1; test touched only after k selection",
        },
        "split": {
            "train_dates": train_dates,
            "validation_date": validation_date,
            "test_date": args.test_date,
        },
        "arguments": serializable_args(args),
        "reconstruction": reconstruction_records,
        "normalization_sources": {
            day.date: list(day.normalization_sources) for day in days
        },
        "candidate_results": candidates,
        "selected_k": selected_k,
        "selected_checkpoint": str(selected_checkpoint.resolve()),
        "test_label_summary": test_summary,
        "test": {"loss": test_loss, **test_metrics},
        "majority_baseline": baseline,
        "limitations": [
            "Only eight non-consecutive trading dates are available, unlike the paper's one-year LSE sample.",
            "The first date has no earlier normalisation history and therefore uses its own feature statistics as a warm start.",
            "Training and validation targets are subsampled by their configured strides; the final test uses every eligible target.",
            "A snapshot is emitted per accepted update timestamp group, because multiple HKEX rows can share one exchange timestamp.",
        ],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    saved["test_loss"] = test_loss
    saved["test_metrics"] = test_metrics
    saved["results_path"] = str(args.results.resolve())
    torch.save(saved, selected_checkpoint)
    print(f"Checkpoint: {selected_checkpoint.resolve()}", flush=True)
    print(f"Results: {args.results.resolve()}", flush=True)
    for archive in archives:
        archive.close()


if __name__ == "__main__":
    main()
