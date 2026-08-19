#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "torch>=2.1",
# ]
# ///
"""Evaluate a FI-2010 DeepLOB checkpoint on reconstructed HK 7709 data."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("down", "stationary", "up")
FI_TEST_CLASS_COUNTS = np.array([47_915, 48_050, 43_523], dtype=np.int64)
FI_STATIONARY_SHARE = float(FI_TEST_CLASS_COUNTS[1] / FI_TEST_CLASS_COUNTS.sum())


class deeplob(nn.Module):
    """Compatibility class for the trusted model saved by the repo notebook."""

    def __init__(self, y_len: int = 3) -> None:
        super().__init__()
        reference = DeepLOB(y_len)
        self.conv1 = reference.conv1
        self.conv2 = reference.conv2
        self.conv3 = reference.conv3
        self.inp1 = reference.inception1
        self.inp2 = reference.inception2
        self.inp3 = reference.inception3
        self.lstm = reference.lstm
        self.fc1 = reference.classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv3(self.conv2(self.conv1(inputs)))
        features = torch.cat(
            (self.inp1(features), self.inp2(features), self.inp3(features)), dim=1
        )
        features = features.permute(0, 2, 1, 3)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features, _ = self.lstm(features)
        return self.fc1(features[:, -1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of FI-2010 DeepLOB on HK 7709"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data/processed/hk07709_2026-07-09_lob.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "jupyter_pytorch/best_val_model_pytorch",
    )
    parser.add_argument(
        "--checkpoint-format",
        choices=("legacy-model", "state-dict"),
        default="legacy-model",
    )
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=10,
        help="10 sampled states corresponds to roughly 100 raw book updates",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help="early fraction used only to set the stationary-label threshold",
    )
    parser.add_argument(
        "--stationary-share",
        type=float,
        default=FI_STATIONARY_SHARE,
        help="target stationary share on the calibration interval",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/results/fi2010_to_7709_zero_shot.json",
    )
    return parser.parse_args()


def load_model(path: Path, checkpoint_format: str, device: torch.device) -> nn.Module:
    if checkpoint_format == "legacy-model":
        # This repository-owned checkpoint contains a pickled nn.Module.  The
        # compatibility class above is intentionally named as in the notebook.
        model = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module in {path}")
        return model.to(device)

    saved = torch.load(path, map_location=device, weights_only=True)
    state = saved.get("model_state_dict", saved)
    model = DeepLOB().to(device)
    model.load_state_dict(state)
    return model


def forward_returns(mid_prices: np.ndarray, sessions: np.ndarray, horizon: int) -> np.ndarray:
    """Equation (3): future mean relative to the current mid-price."""
    returns = np.full(len(mid_prices), np.nan, dtype=np.float64)
    for current_session in np.unique(sessions):
        positions = np.flatnonzero(sessions == current_session)
        if len(positions) <= horizon:
            continue
        values = mid_prices[positions]
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        local = np.arange(len(values) - horizon)
        future_mean = (
            cumulative[local + horizon + 1] - cumulative[local + 1]
        ) / horizon
        returns[positions[local]] = future_mean / values[local] - 1.0
    return returns


def labels_from_returns(returns: np.ndarray, alpha: float) -> np.ndarray:
    labels = np.full(len(returns), -1, dtype=np.int8)
    valid = np.isfinite(returns)
    labels[valid] = np.where(
        returns[valid] < -alpha,
        0,
        np.where(returns[valid] > alpha, 2, 1),
    )
    return labels


def calibration_cutoff(send_times: np.ndarray, fraction: float) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1")
    return int(math.floor(len(send_times) * fraction))


def valid_target_indices(
    sessions: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    cutoff: int,
) -> np.ndarray:
    candidates = np.arange(max(cutoff, sequence_length - 1), len(labels))
    starts = candidates - sequence_length + 1
    valid = labels[candidates] >= 0
    valid &= sessions[starts] == sessions[candidates]
    return candidates[valid]


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        target_indices: np.ndarray,
        sequence_length: int,
    ) -> None:
        self.features = features
        self.labels = labels
        self.target_indices = target_indices
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_index = int(self.target_indices[index])
        start = target_index - self.sequence_length + 1
        window = np.ascontiguousarray(self.features[start : target_index + 1])
        return (
            torch.from_numpy(window).unsqueeze(0),
            torch.tensor(int(self.labels[target_index]), dtype=torch.int64),
        )


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    total = int(confusion.sum())
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        support = int(confusion[index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    prediction_counts = np.zeros(3, dtype=np.int64)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in loader:
            predictions = model(inputs.to(device, dtype=torch.float32)).argmax(dim=1).cpu()
            target_array = targets.numpy()
            prediction_array = predictions.numpy()
            prediction_counts += np.bincount(prediction_array, minlength=3)
            indices = target_array * 3 + prediction_array
            confusion += np.bincount(indices, minlength=9).reshape(3, 3)
    result = metrics_from_confusion(confusion)
    result["duration_seconds"] = time.perf_counter() - started
    result["prediction_counts"] = prediction_counts.tolist()
    return result, confusion


def majority_baseline(labels: np.ndarray) -> dict[str, object]:
    counts = np.bincount(labels, minlength=3)
    majority = int(counts.argmax())
    confusion = np.zeros((3, 3), dtype=np.int64)
    confusion[:, majority] = counts
    result = metrics_from_confusion(confusion)
    result["majority_class"] = CLASS_NAMES[majority]
    return result


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0.0 < args.stationary_share < 1.0:
        raise ValueError("--stationary-share must be between 0 and 1")

    archive = np.load(args.data, allow_pickle=False)
    features = archive["features"].astype(np.float32, copy=False)
    send_times = archive["send_times"].astype(np.int64, copy=False)
    sessions = archive["sessions"].astype(np.uint8, copy=False)
    metadata = json.loads(str(archive["metadata"]))
    if features.ndim != 2 or features.shape[1] != 40:
        raise ValueError(f"Expected [N, 40] features, found {features.shape}")

    mid_prices = (features[:, 0].astype(np.float64) + features[:, 2]) / 2.0
    returns = forward_returns(mid_prices, sessions, args.horizon_steps)
    cutoff = calibration_cutoff(send_times, args.calibration_fraction)
    calibration_returns = returns[:cutoff]
    calibration_returns = calibration_returns[np.isfinite(calibration_returns)]
    alpha = float(np.quantile(np.abs(calibration_returns), args.stationary_share))
    labels = labels_from_returns(returns, alpha)
    target_indices = valid_target_indices(
        sessions, labels, args.sequence_length, cutoff
    )
    if not len(target_indices):
        raise RuntimeError("No valid evaluation windows")

    dataset = WindowDataset(features, labels, target_indices, args.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    model = load_model(args.checkpoint, args.checkpoint_format, device)
    model_metrics, _ = evaluate(model, loader, device)
    evaluation_labels = labels[target_indices]

    result = {
        "experiment": "FI-2010 checkpoint zero-shot transfer to HK 7709",
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_format": args.checkpoint_format,
        "device": str(device),
        "preprocessing": {
            "source_metadata": metadata,
            "sequence_length": args.sequence_length,
            "horizon_sampled_steps": args.horizon_steps,
            "approximate_horizon_book_update_groups": args.horizon_steps
            * int(metadata["snapshot_every_book_update_groups"]),
            "label_method": "DeepLOB Equation (3): future mean vs current mid-price",
            "calibration_fraction": args.calibration_fraction,
            "target_calibration_stationary_share": args.stationary_share,
            "calibrated_alpha": alpha,
            "evaluation_start_send_time": int(send_times[target_indices[0]]),
            "evaluation_end_send_time": int(send_times[target_indices[-1]]),
            "evaluation_label_counts": np.bincount(
                evaluation_labels, minlength=3
            ).tolist(),
        },
        "model": model_metrics,
        "majority_baseline": majority_baseline(evaluation_labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
