#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "joblib>=1.3",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
#     "sortedcontainers>=2.4",
#     "torch>=2.1",
# ]
# ///
"""Causal-feature predictability diagnostic for the HK 07709 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_7709_deeplob import DayData, eligible_targets, equation4_returns


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("down", "stationary", "up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed/7709"
    )
    parser.add_argument("--validation-date", default="2026-08-04")
    parser.add_argument("--test-date", default="2026-08-07")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--stationary-share", type=float, default=1 / 3)
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "output/results/7709_causal_baselines.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/7709/7709_logistic_causal.joblib",
    )
    return parser.parse_args()


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
        raise FileNotFoundError(f"No reconstructed 07709 days under {processed_dir}")
    return days


def causal_features(day: DayData, indices: np.ndarray) -> np.ndarray:
    """Features available at the prediction timestamp; no future values."""
    book = day.features
    mid = (book[:, 0].astype(np.float64) + book[:, 2]) / 2.0
    columns: list[np.ndarray] = []
    for horizon in (1, 2, 5, 10, 20, 50, 100):
        previous = indices - horizon
        columns.append((mid[indices] / mid[previous] - 1.0) * 10_000.0)
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
        columns.append(
            (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
        )
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


def metric_record(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return {
        "samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "prediction_counts": np.bincount(prediction, minlength=3).tolist(),
    }


def main() -> None:
    args = parse_args()
    days = load_days(args.processed_dir)
    train_dates = {day.date for day in days if day.date < args.validation_date}
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
    alpha = float(
        np.quantile(np.abs(np.concatenate(calibration)), args.stationary_share)
    )

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for day in days:
        features[day.date] = causal_features(day, indices[day.date])
        values = returns[day.date][indices[day.date]]
        labels[day.date] = np.where(
            values < -alpha, 0, np.where(values > alpha, 2, 1)
        ).astype(np.int64)
    train_x = np.concatenate([features[date] for date in sorted(train_dates)])
    train_y = np.concatenate([labels[date] for date in sorted(train_dates)])
    validation_x = features[args.validation_date]
    validation_y = labels[args.validation_date]
    test_x = features[args.test_date]
    test_y = labels[args.test_date]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, max_iter=500, class_weight="balanced", random_state=42
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=31,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    model_results: dict[str, object] = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        model_results[name] = {
            "train": metric_record(train_y, model.predict(train_x)),
            "validation": metric_record(validation_y, model.predict(validation_x)),
            "test": metric_record(test_y, model.predict(test_x)),
        }
        print(
            f"{name}: val_acc={model_results[name]['validation']['accuracy']:.4f}, "
            f"test_acc={model_results[name]['test']['accuracy']:.4f}"
        )

    # This model is saved as a diagnostic baseline, not a trading strategy.
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["logistic_regression"], args.model)
    result = {
        "experiment": "HK 07709 causal-feature predictability diagnostic",
        "status": "completed",
        "split": {
            "train_dates": sorted(train_dates),
            "validation_date": args.validation_date,
            "test_date": args.test_date,
        },
        "labeling": {
            "method": "DeepLOB Equation (4)",
            "k": args.k,
            "alpha": alpha,
            "classes": list(CLASS_NAMES),
        },
        "features": {
            "count": int(train_x.shape[1]),
            "causal_only": True,
            "summary": "lagged mid returns, trailing-mean returns, spread, multi-level depth imbalance, queue imbalance, book gaps, log depth",
        },
        "models": model_results,
        "saved_logistic_model": str(args.model.resolve()),
        "note": "Test metrics diagnose whether the label is predictable; no execution costs or overlap-adjusted uncertainty are included.",
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Results: {args.results.resolve()}")


if __name__ == "__main__":
    main()
