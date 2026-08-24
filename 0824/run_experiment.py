#!/usr/bin/env python3
"""Single-day 70/30 chronological DeepLOB experiment for HK 07709."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_pytorch_mac import DeepLOB, choose_device  # noqa: E402
from lob_reconstruction import (  # noqa: E402
    audit_and_select_highest_volume,
    reconstruct_l2_npz,
    save_reconstruction,
)


CLASS_NAMES = ("down", "stationary", "up")
NUM_CLASSES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path.home() / "Desktop/7709_tickdata"
    )
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--stationary-share", type=float, default=1 / 3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--force-reconstruct", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error("train-ratio must be between zero and one")
    if not 0 < args.stationary_share < 1:
        parser.error("stationary-share must be between zero and one")
    if min(args.snapshot_every, args.sequence_length, args.horizon, args.epochs, args.batch_size) < 1:
        parser.error("sampling, sequence, horizon, epoch, and batch values must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transform_features(features: np.ndarray) -> np.ndarray:
    """Use current-mid anchored prices and log sizes before train-only z-score."""

    output = features.astype(np.float32, copy=True)
    mid = (features[:, 0].astype(np.float64) + features[:, 2]) / 2
    for level in range(10):
        base = level * 4
        output[:, base] = ((features[:, base] / mid) - 1.0) * 10_000
        output[:, base + 2] = ((features[:, base + 2] / mid) - 1.0) * 10_000
        output[:, base + 1] = np.log1p(features[:, base + 1])
        output[:, base + 3] = np.log1p(features[:, base + 3])
    return output


def symmetric_returns(mids: np.ndarray, segments: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(len(mids), np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        if len(positions) < 2 * horizon:
            continue
        if np.any(np.diff(positions) != 1):
            raise ValueError(f"segment {segment} is not contiguous")
        values = mids[positions]
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        local = np.arange(horizon - 1, len(values) - horizon, dtype=np.int64)
        past = (cumulative[local + 1] - cumulative[local - horizon + 1]) / horizon
        future = (cumulative[local + horizon + 1] - cumulative[local + 1]) / horizon
        result[positions[local]] = future / past - 1.0
    return result


def eligible_indices(
    returns: np.ndarray,
    segments: np.ndarray,
    split_index: int,
    sequence_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        start = int(positions[0])
        stop = int(positions[-1]) + 1
        candidates = np.arange(start + max(sequence_length, horizon) - 1, stop - horizon)
        candidates = candidates[np.isfinite(returns[candidates])]
        train.append(candidates[candidates + horizon < split_index])
        # Strict separation: no test input or label-history state may predate split.
        test.append(candidates[candidates - sequence_length + 1 >= split_index])
    return (
        np.concatenate(train) if train else np.empty(0, dtype=np.int64),
        np.concatenate(test) if test else np.empty(0, dtype=np.int64),
    )


@dataclass
class PreparedData:
    normalized: np.ndarray
    train_indices: np.ndarray
    train_labels: np.ndarray
    test_indices: np.ndarray
    test_labels: np.ndarray
    alpha: float
    mean: np.ndarray
    std: np.ndarray
    returns: np.ndarray


def prepare_data(
    features: np.ndarray,
    segments: np.ndarray,
    train_ratio: float,
    sequence_length: int,
    horizon: int,
    stationary_share: float,
) -> PreparedData:
    split_index = int(math.floor(len(features) * train_ratio))
    transformed = transform_features(features)
    mean = transformed[:split_index].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = transformed[:split_index].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = np.ascontiguousarray((transformed - mean) / std, dtype=np.float32)
    mids = (features[:, 0].astype(np.float64) + features[:, 2]) / 2
    returns = symmetric_returns(mids, segments, horizon)
    train_indices, test_indices = eligible_indices(
        returns, segments, split_index, sequence_length, horizon
    )
    if not len(train_indices) or not len(test_indices):
        raise RuntimeError("70/30 split produced an empty train or test target set")
    alpha = float(np.quantile(np.abs(returns[train_indices]), stationary_share))

    def labels(indices: np.ndarray) -> np.ndarray:
        values = returns[indices]
        return np.where(values < -alpha, 0, np.where(values > alpha, 2, 1)).astype(np.int64)

    return PreparedData(
        normalized=normalized,
        train_indices=train_indices,
        train_labels=labels(train_indices),
        test_indices=test_indices,
        test_labels=labels(test_indices),
        alpha=alpha,
        mean=mean,
        std=std,
        returns=returns,
    )


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, features: np.ndarray, indices: np.ndarray, labels: np.ndarray, sequence_length: int
    ) -> None:
        self.features = features
        self.indices = indices
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = int(self.indices[item])
        start = target - self.sequence_length + 1
        window = np.ascontiguousarray(self.features[start : target + 1])
        return torch.from_numpy(window).unsqueeze(0), torch.tensor(int(self.labels[item]))


def confusion_metrics(confusion: np.ndarray) -> dict[str, object]:
    total = int(confusion.sum())
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        support = int(confusion[index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {"class": name, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    return {
        "samples": total,
        "accuracy": float(np.trace(confusion) / total),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def run_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            targets_device = targets.to(device, dtype=torch.int64, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets_device)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            truth = targets.numpy()
            confusion += np.bincount(
                truth * NUM_CLASSES + predictions, minlength=NUM_CLASSES**2
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            total_loss += float(loss.item()) * len(truth)
            if training and batch % 50 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"    batch {batch}/{len(loader)}; {int(confusion.sum()) / elapsed:.0f} samples/s",
                    flush=True,
                )
    return total_loss / max(1, int(confusion.sum())), confusion_metrics(confusion)


def constant_baseline(test_labels: np.ndarray, predicted: int) -> dict[str, object]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion[:, predicted] = np.bincount(test_labels, minlength=NUM_CLASSES)
    result = confusion_metrics(confusion)
    result["predicted_class"] = CLASS_NAMES[predicted]
    return result


def majority_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> dict[str, object]:
    predicted = int(np.bincount(train_labels, minlength=NUM_CLASSES).argmax())
    return constant_baseline(test_labels, predicted)


def label_summary(labels: np.ndarray) -> dict[str, object]:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {"counts": counts.tolist(), "shares": (counts / counts.sum()).tolist()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "checkpoints").mkdir(exist_ok=True)

    print("Auditing daily printed trade volume...", flush=True)
    audit, selected = audit_and_select_highest_volume(args.raw_dir)
    audit_path = HERE / "output/volume_audit.json"
    audit_path.write_text(
        json.dumps({"metric": "sum of MsgType=50 Quantity", "days": audit, "selected": selected}, indent=2),
        encoding="utf-8",
    )
    selected_date = str(selected["date"])
    source_npz = args.raw_dir / f"hk07709_{selected_date}.npz"
    if not source_npz.is_file():
        raise FileNotFoundError(
            f"Highest-volume day is {selected_date}, but its L2 source is missing: {source_npz}"
        )
    print(
        f"Selected {selected_date}: {int(selected['trade_quantity']):,} printed shares",
        flush=True,
    )

    reconstruction_path = HERE / f"data/hk07709_{selected_date}_l2_s{args.snapshot_every}.npz"
    if args.force_reconstruct or not reconstruction_path.is_file():
        print(f"Reconstructing {source_npz}...", flush=True)
        reconstructed = reconstruct_l2_npz(source_npz, args.snapshot_every)
        save_reconstruction(reconstruction_path, reconstructed)
    else:
        with np.load(reconstruction_path, allow_pickle=False) as archive:
            from lob_reconstruction import ReconstructedLOB

            reconstructed = ReconstructedLOB(
                features=archive["features"],
                exchange_ms=archive["exchange_ms"],
                send_times=archive["send_times"],
                segments=archive["segments"],
                metadata=json.loads(str(archive["metadata"])),
            )
    print(
        f"Snapshots: {len(reconstructed.features):,}; quality={reconstructed.metadata['quality_counts']}",
        flush=True,
    )

    prepared = prepare_data(
        reconstructed.features,
        reconstructed.segments,
        args.train_ratio,
        args.sequence_length,
        args.horizon,
        args.stationary_share,
    )
    split_index = int(math.floor(len(reconstructed.features) * args.train_ratio))
    print(
        f"Split at snapshot {split_index:,}/{len(reconstructed.features):,} "
        f"({int(reconstructed.send_times[split_index])}); "
        f"train targets={len(prepared.train_indices):,}, test targets={len(prepared.test_indices):,}, "
        f"alpha={prepared.alpha:.8g}",
        flush=True,
    )

    train_dataset = WindowDataset(
        prepared.normalized, prepared.train_indices, prepared.train_labels, args.sequence_length
    )
    test_dataset = WindowDataset(
        prepared.normalized, prepared.test_indices, prepared.test_labels, args.sequence_length
    )
    generator = torch.Generator().manual_seed(args.seed)
    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        persistent_workers=args.num_workers > 0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=pin_memory,
    )

    model = DeepLOB(NUM_CLASSES).to(device)
    counts = np.bincount(prepared.train_labels, minlength=NUM_CLASSES).astype(np.float64)
    class_weights = counts.sum() / (NUM_CLASSES * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    print(f"Training on {device} for fixed {args.epochs} epochs...", flush=True)
    for epoch in range(1, args.epochs + 1):
        loss, metrics = run_loader(model, train_loader, criterion, device, optimizer)
        row = {"epoch": epoch, "train_loss": loss, **metrics}
        history.append(row)
        print(
            f"epoch {epoch:02d}: loss={loss:.4f}, acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    test_loss, test_metrics = run_loader(model, test_loader, criterion, device)
    test_metrics["loss"] = test_loss
    baseline = majority_baseline(prepared.train_labels, prepared.test_labels)
    stationary_baseline = constant_baseline(prepared.test_labels, 1)
    oracle_test_majority = constant_baseline(
        prepared.test_labels,
        int(np.bincount(prepared.test_labels, minlength=NUM_CLASSES).argmax()),
    )
    checkpoint_path = HERE / f"checkpoints/deeplob_7709_{selected_date}_70_30.pt"
    results_path = HERE / "output/results.json"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "selected_date": selected_date,
        "arguments": vars(args),
        "preprocessing": {
            "mean": prepared.mean.tolist(),
            "std": prepared.std.tolist(),
            "price": "current-mid anchored basis points",
            "size": "log1p raw aggregate quantity",
        },
        "labeling": {"horizon": args.horizon, "alpha": prepared.alpha},
        "split_index": split_index,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, checkpoint_path)

    payload = {
        "experiment": "HK 07709 highest-volume single-day chronological 70/30 DeepLOB",
        "status": "complete",
        "selected_day": selected,
        "volume_audit": str(audit_path.resolve()),
        "source_l2": str(source_npz.resolve()),
        "reconstruction_cache": str(reconstruction_path.resolve()),
        "reconstruction": reconstructed.metadata,
        "split": {
            "method": "chronological snapshot split with disjoint input and label windows",
            "ratio": [args.train_ratio, 1 - args.train_ratio],
            "split_index": split_index,
            "split_send_time_hkt": int(reconstructed.send_times[split_index]),
            "train_last_target_send_time_hkt": int(
                reconstructed.send_times[prepared.train_indices[-1]]
            ),
            "test_first_target_send_time_hkt": int(
                reconstructed.send_times[prepared.test_indices[0]]
            ),
            "shared_snapshots_between_train_and_test_inputs": 0,
        },
        "features": {
            "sequence_length": args.sequence_length,
            "book_levels": 10,
            "price_transform": "(level_price / current_mid - 1) * 10000",
            "size_transform": "log1p(aggregate_quantity)",
            "normalization": "per-feature z-score fitted only on first 70% snapshots",
        },
        "labeling": {
            "method": "symmetric past/future mean mid-price return",
            "horizon_snapshots": args.horizon,
            "alpha": prepared.alpha,
            "alpha_fit": "quantile of absolute returns on eligible training targets only",
            "target_stationary_share": args.stationary_share,
        },
        "train": {
            "targets": len(prepared.train_indices),
            "labels": label_summary(prepared.train_labels),
            "class_weights": class_weights.tolist(),
            "fixed_epochs_no_test_selection": args.epochs,
            "history": history,
        },
        "test": {
            "targets": len(prepared.test_indices),
            "labels": label_summary(prepared.test_labels),
            "metrics": test_metrics,
        },
        "majority_baseline": baseline,
        "stationary_baseline": stationary_baseline,
        "oracle_test_majority_diagnostic": oracle_test_majority,
        "uniform_random_expected_accuracy": 1 / 3,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "seed": args.seed,
        "limitations": [
            "single trading day; test observations are strongly autocorrelated overlapping windows",
            "fixed 70/30 split provides no independent validation day",
            "classification metrics do not include fees, fill probability, latency, or market impact",
        ],
    }
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"TEST: loss={test_loss:.4f}, acc={test_metrics['accuracy']:.4f}, "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}; "
        f"train-majority_acc={baseline['accuracy']:.4f}, "
        f"stationary_acc={stationary_baseline['accuracy']:.4f}",
        flush=True,
    )
    print(f"Saved {results_path} and {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
