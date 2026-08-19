#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "sortedcontainers>=2.4",
#     "torch>=2.1",
# ]
# ///
"""Train DeepLOB with target-anchored book scaling and causal trend features."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from train_7709_deeplob import (
    CLASS_NAMES,
    DayData,
    confusion_metrics,
    eligible_targets,
    equation4_returns,
)
from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
PRICE_COLUMNS = np.arange(0, 40, 2)
SIZE_COLUMNS = np.arange(1, 40, 2)
NUM_CLASSES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed/7709"
    )
    parser.add_argument("--validation-date", default="2026-08-04")
    parser.add_argument("--test-date", default="2026-08-07")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--stationary-share", type=float, default=1 / 3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-4)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/7709/deeplob_7709_hybrid_k20_best.pt",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "output/results/7709_hybrid_deeplob.json",
    )
    args = parser.parse_args()
    if min(
        args.k,
        args.sequence_length,
        args.train_stride,
        args.validation_stride,
        args.epochs,
        args.batch_size,
    ) < 1:
        parser.error("k, sequence, stride, epochs, and batch size must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_days(processed_dir: Path) -> list[DayData]:
    days: list[DayData] = []
    for path in sorted(processed_dir.glob("*_s10_lob.npz")):
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        date = Path(str(metadata["source"])).stem.removeprefix("hk07709_")
        days.append(
            DayData(
                date,
                path,
                archive["features"].astype(np.float32, copy=False),
                archive["sessions"].astype(np.uint8, copy=False),
                metadata,
            )
        )
    if not days:
        raise FileNotFoundError(f"No reconstructed data under {processed_dir}")
    return days


def transform_rows(book: np.ndarray) -> np.ndarray:
    """Stationary book shape transform anchored to each row's current mid."""
    transformed = np.empty_like(book, dtype=np.float32)
    mid = (book[:, 0].astype(np.float64) + book[:, 2]) / 2.0
    transformed[:, PRICE_COLUMNS] = (
        book[:, PRICE_COLUMNS] / mid[:, None] - 1.0
    ) * 10_000.0
    transformed[:, SIZE_COLUMNS] = np.log1p(
        book[:, SIZE_COLUMNS] * 100_000.0
    )
    return transformed


def fit_book_scaler(days: list[DayData]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(40, dtype=np.float64)
    square = np.zeros(40, dtype=np.float64)
    for day in days:
        values = transform_rows(day.features).astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def transform_window(
    window: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Anchor all historic prices to the prediction-time mid-price.

    Unlike row-by-row centering, this retains the causal mid-price path inside
    the 100-state input while removing the absolute stock-price regime.
    """
    transformed = np.empty_like(window, dtype=np.float32)
    target_mid = float((window[-1, 0] + window[-1, 2]) / 2.0)
    transformed[:, PRICE_COLUMNS] = (
        window[:, PRICE_COLUMNS] / target_mid - 1.0
    ) * 10_000.0
    transformed[:, SIZE_COLUMNS] = np.log1p(
        window[:, SIZE_COLUMNS] * 100_000.0
    )
    normalized = (transformed - mean) / std
    return np.ascontiguousarray(np.clip(normalized, -20.0, 20.0))


def causal_features(day: DayData, indices: np.ndarray) -> np.ndarray:
    book = day.features
    mid = (book[:, 0].astype(np.float64) + book[:, 2]) / 2.0
    columns: list[np.ndarray] = []
    for horizon in (1, 2, 5, 10, 20, 50, 100):
        columns.append((mid[indices] / mid[indices - horizon] - 1.0) * 10_000.0)
    cumulative = np.concatenate(([0.0], np.cumsum(mid, dtype=np.float64)))
    for horizon in (5, 20, 50):
        trailing_mean = (
            cumulative[indices + 1] - cumulative[indices - horizon + 1]
        ) / horizon
        columns.append((mid[indices] / trailing_mean - 1.0) * 10_000.0)
    columns.append((book[indices, 0] - book[indices, 2]) / mid[indices] * 10_000.0)
    asks = book[indices][:, np.arange(1, 40, 4)].astype(np.float64)
    bids = book[indices][:, np.arange(3, 40, 4)].astype(np.float64)
    for levels in (1, 2, 3, 5, 10):
        ask_depth = asks[:, :levels].sum(axis=1)
        bid_depth = bids[:, :levels].sum(axis=1)
        columns.append((bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12))
    for level in range(5):
        columns.append(
            (bids[:, level] - asks[:, level])
            / (bids[:, level] + asks[:, level] + 1e-12)
        )
    columns.extend(
        [
            (book[indices, 4] - book[indices, 0]) / mid[indices] * 10_000.0,
            (book[indices, 2] - book[indices, 6]) / mid[indices] * 10_000.0,
            np.log1p(asks.sum(axis=1) * 100_000.0),
            np.log1p(bids.sum(axis=1) * 100_000.0),
        ]
    )
    return np.column_stack(columns).astype(np.float32)


@dataclass
class PreparedDay:
    day: DayData
    indices: np.ndarray
    labels: np.ndarray
    auxiliary: np.ndarray


class HybridDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        prepared: PreparedDay,
        sequence_length: int,
        book_mean: np.ndarray,
        book_std: np.ndarray,
        auxiliary_mean: np.ndarray,
        auxiliary_std: np.ndarray,
    ) -> None:
        self.prepared = prepared
        self.sequence_length = sequence_length
        self.book_mean = book_mean
        self.book_std = book_std
        self.auxiliary_mean = auxiliary_mean
        self.auxiliary_std = auxiliary_std

    def __len__(self) -> int:
        return len(self.prepared.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = int(self.prepared.indices[item])
        start = target - self.sequence_length + 1
        window = self.prepared.day.features[start : target + 1]
        book = transform_window(window, self.book_mean, self.book_std)
        auxiliary = np.ascontiguousarray(
            (self.prepared.auxiliary[item] - self.auxiliary_mean)
            / self.auxiliary_std
        )
        return (
            torch.from_numpy(book).unsqueeze(0),
            torch.from_numpy(auxiliary),
            torch.tensor(int(self.prepared.labels[item]), dtype=torch.int64),
        )


class HybridDeepLOB(nn.Module):
    def __init__(self, auxiliary_features: int) -> None:
        super().__init__()
        self.backbone = DeepLOB()
        self.lob_head = nn.Linear(64, NUM_CLASSES)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(auxiliary_features, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, NUM_CLASSES),
        )

    def lob_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        net = self.backbone
        features = net.conv3(net.conv2(net.conv1(inputs)))
        features = torch.cat(
            (
                net.inception1(features),
                net.inception2(features),
                net.inception3(features),
            ),
            dim=1,
        )
        features = features.permute(0, 2, 1, 3)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features, _ = net.lstm(features)
        return features[:, -1]

    def forward(self, book: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        return self.lob_head(self.lob_embedding(book)) + self.auxiliary_head(auxiliary)


def make_loader(
    items: list[PreparedDay],
    args: argparse.Namespace,
    book_mean: np.ndarray,
    book_std: np.ndarray,
    auxiliary_mean: np.ndarray,
    auxiliary_std: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    datasets = [
        HybridDataset(
            item,
            args.sequence_length,
            book_mean,
            book_std,
            auxiliary_mean,
            auxiliary_std,
        )
        for item in items
    ]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


def run_epoch(
    model: HybridDeepLOB,
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
        for number, (book, auxiliary, targets) in enumerate(loader, start=1):
            book = book.to(device, dtype=torch.float32)
            auxiliary = auxiliary.to(device, dtype=torch.float32)
            targets_device = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(book, auxiliary)
            loss = criterion(logits, targets_device)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            prediction = logits.argmax(dim=1).detach().cpu().numpy()
            truth = targets.numpy()
            confusion += np.bincount(
                truth * NUM_CLASSES + prediction, minlength=9
            ).reshape(3, 3)
            total_loss += loss.item() * len(truth)
            if training and log_every and number % log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  batch {number}/{len(loader)} "
                    f"({int(confusion.sum()) / elapsed:.0f} samples/s)",
                    flush=True,
                )
    metrics = confusion_metrics(confusion)
    metrics["duration_seconds"] = time.perf_counter() - started
    return total_loss / max(1, int(confusion.sum())), metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    days = load_days(args.processed_dir)
    train_dates = {day.date for day in days if day.date < args.validation_date}
    train_days = [day for day in days if day.date in train_dates]
    returns = {day.date: equation4_returns(day, args.k) for day in days}
    indices: dict[str, np.ndarray] = {}
    calibration: list[np.ndarray] = []
    for day in days:
        if day.date in train_dates:
            stride = args.train_stride
        elif day.date == args.validation_date:
            stride = args.validation_stride
        else:
            stride = 1
        indices[day.date] = eligible_targets(
            day, returns[day.date], args.sequence_length, stride
        )
        if day.date in train_dates:
            calibration.append(returns[day.date][indices[day.date]])
    alpha = float(np.quantile(np.abs(np.concatenate(calibration)), 1 / 3))

    prepared: dict[str, PreparedDay] = {}
    for day in days:
        values = returns[day.date][indices[day.date]]
        labels = np.where(
            values < -alpha, 0, np.where(values > alpha, 2, 1)
        ).astype(np.int64)
        prepared[day.date] = PreparedDay(
            day, indices[day.date], labels, causal_features(day, indices[day.date])
        )
    train_items = [prepared[date] for date in sorted(train_dates)]
    validation_items = [prepared[args.validation_date]]
    test_items = [prepared[args.test_date]]
    train_auxiliary = np.concatenate([item.auxiliary for item in train_items])
    auxiliary_mean = train_auxiliary.mean(axis=0).astype(np.float32)
    auxiliary_std = np.maximum(train_auxiliary.std(axis=0), 1e-6).astype(np.float32)
    book_mean, book_std = fit_book_scaler(train_days)

    train_loader = make_loader(
        train_items, args, book_mean, book_std, auxiliary_mean, auxiliary_std, True
    )
    validation_loader = make_loader(
        validation_items, args, book_mean, book_std, auxiliary_mean, auxiliary_std, False
    )
    test_loader = make_loader(
        test_items, args, book_mean, book_std, auxiliary_mean, auxiliary_std, False
    )
    print(
        f"train={len(train_loader.dataset):,}, val={len(validation_loader.dataset):,}, "
        f"test={len(test_loader.dataset):,}, alpha={alpha:.10g}",
        flush=True,
    )

    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    model = HybridDeepLOB(train_auxiliary.shape[1]).to(device)
    backbone_parameters = list(model.backbone.parameters()) + list(model.lob_head.parameters())
    auxiliary_parameters = list(model.auxiliary_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": auxiliary_parameters, "lr": args.aux_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -math.inf
    best_loss = math.inf
    unimproved = 0
    history: list[dict[str, object]] = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.log_every
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
        improved = score > best_f1 + 1e-8 or (
            abs(score - best_f1) <= 1e-8 and validation_loss < best_loss
        )
        if improved:
            best_f1 = score
            best_loss = validation_loss
            unimproved = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_name": "HybridDeepLOB",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_metrics": validation_metrics,
                    "labeling": {"method": "DeepLOB Equation (4)", "k": args.k, "alpha": alpha},
                    "preprocessing": {
                        "book": "target-mid anchored bps prices; log1p raw sizes; train-only z-score",
                        "book_mean": book_mean.tolist(),
                        "book_std": book_std.tolist(),
                        "auxiliary_mean": auxiliary_mean.tolist(),
                        "auxiliary_std": auxiliary_std.tolist(),
                        "auxiliary_features": int(train_auxiliary.shape[1]),
                    },
                    "config": {
                        key: str(value.resolve()) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                args.checkpoint,
            )
        else:
            unimproved += 1
        print(
            f"epoch={epoch:02d}: train_loss={train_loss:.4f}, "
            f"train_f1={train_metrics['macro_f1']:.4f}, "
            f"val_loss={validation_loss:.4f}, val_acc={validation_metrics['accuracy']:.4f}, "
            f"val_f1={validation_metrics['macro_f1']:.4f}"
            f"{' saved' if improved else ''}",
            flush=True,
        )
        if args.patience and unimproved >= args.patience:
            print("early stopping", flush=True)
            break

    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, device)
    print(
        f"test: loss={test_loss:.4f}, accuracy={test_metrics['accuracy']:.4f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    result = {
        "experiment": "Hybrid DeepLOB on HK 07709",
        "status": "completed",
        "split": {
            "train_dates": sorted(train_dates),
            "validation_date": args.validation_date,
            "test_date": args.test_date,
        },
        "labeling": saved["labeling"],
        "preprocessing": saved["preprocessing"],
        "training": {
            "device": str(device),
            "best_epoch": int(saved["epoch"]),
            "validation_loss": float(saved["validation_loss"]),
            "validation": saved["validation_metrics"],
            "history": history,
        },
        "test": {"loss": test_loss, **test_metrics},
        "checkpoint": str(args.checkpoint.resolve()),
        "caveats": [
            "The auxiliary branch uses only information available at the prediction timestamp.",
            "Overlapping labels reduce the effective independent sample count.",
            "Accuracy does not include fees, latency, spread crossing, or market impact.",
        ],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    saved["test_loss"] = test_loss
    saved["test_metrics"] = test_metrics
    saved["results_path"] = str(args.results.resolve())
    torch.save(saved, args.checkpoint)
    print(f"Checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"Results: {args.results.resolve()}", flush=True)


if __name__ == "__main__":
    main()
