#!/usr/bin/env python3
"""Same 25-feature causal logistic regression as HK 07709, on FI-2010 and LOBSTER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reproduce_pytorch import load_fi2010
from train_lobster_samples import (
    CLASS_NAMES,
    DATE,
    DEFAULT_CHECKPOINT as LOBSTER_CHECKPOINT,
    DEFAULT_PROCESSED_DIR,
    SYMBOLS,
    eligible_local_indices,
    forward_returns,
    make_prepared_splits,
    split_bounds,
    transformed_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FI_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "fi2010"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fi-data-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--lobster-checkpoint", type=Path, default=LOBSTER_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/results/fi2010_lobster_logistic.json",
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--horizon-index", type=int, default=4)
    parser.add_argument("--skip-fi2010", action="store_true")
    parser.add_argument("--skip-lobster", action="store_true")
    return parser.parse_args()


def causal_features(
    book: np.ndarray,
    indices: np.ndarray,
    min_index: int = 0,
) -> np.ndarray:
    """Features available at the prediction timestamp; no future values.

    Same 25 columns as analyze_7709_causal_baselines.py. Book layout is
    ask_price, ask_size, bid_price, bid_size for 10 levels.
    Lookbacks stay inside the current split (min_index).
    """
    mid = (book[:, 0].astype(np.float64) + book[:, 2]) / 2.0
    columns: list[np.ndarray] = []
    for horizon in (1, 2, 5, 10, 20, 50, 100):
        previous = np.clip(indices - horizon, min_index, None)
        columns.append((mid[indices] / mid[previous] - 1.0) * 10_000.0)
    cumulative = np.concatenate(([0.0], np.cumsum(mid, dtype=np.float64)))
    for horizon in (5, 20, 50):
        start = np.clip(indices - horizon + 1, min_index, None)
        trailing_mean = (cumulative[indices + 1] - cumulative[start]) / np.maximum(
            indices - start + 1, 1
        )
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
            (bids[:, level] - asks[:, level]) / (bids[:, level] + asks[:, level] + 1e-12)
        )
    columns.extend(
        [
            (book[indices, 4] - book[indices, 0]) / mid[indices] * 10_000.0,
            (book[indices, 2] - book[indices, 6]) / mid[indices] * 10_000.0,
            np.log1p(np.maximum(asks.sum(axis=1), 0.0) * 100_000.0),
            np.log1p(np.maximum(bids.sum(axis=1), 0.0) * 100_000.0),
        ]
    )
    return np.column_stack(columns).astype(np.float32)


def invert_lobster_book(features: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """Undo bps-from-mid / log1p so causal features match the 7709 recipe."""
    book = np.asarray(features, dtype=np.float64)
    prices = book[:, 0::2]
    sizes = book[:, 1::2]
    restored = np.empty_like(book)
    restored[:, 0::2] = mids[:, None] * (1.0 + prices / 10_000.0)
    restored[:, 1::2] = np.expm1(sizes)
    return restored


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
    print(f"{'':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}", flush=True)
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


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
    )
    print(f"Fitting logistic regression on {len(train_y):,} samples x {train_x.shape[1]} features...", flush=True)
    model.fit(train_x, train_y)
    return model


def evaluate_model(model, splits: dict[str, tuple[np.ndarray, np.ndarray]], title_prefix: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for split, (features, labels) in splits.items():
        prediction = model.predict(features)
        report = report_dict(labels, prediction)
        result[split] = {"samples": int(len(labels)), "classification_report": report}
        print_split(f"{title_prefix} {split}", report)
    return result


def load_fi2010_books(data_dir: Path, horizon_index: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache = {
        "train": (FI_CACHE_DIR / "train_features.npy", FI_CACHE_DIR / "train_labels.npy"),
        "test": (FI_CACHE_DIR / "test_features.npy", FI_CACHE_DIR / "test_labels.npy"),
    }
    if all(path.is_file() for pair in cache.values() for path in pair):
        print("Loading cached FI-2010 features...", flush=True)
        return {
            split: (np.load(paths[0]), np.load(paths[1]))
            for split, paths in cache.items()
        }

    print("Loading FI-2010 text files (slow on first run)...", flush=True)
    train_matrix, _, test_matrix = load_fi2010(data_dir)
    books = {
        "train": (
            np.ascontiguousarray(train_matrix[:40, :].T, dtype=np.float32),
            np.ascontiguousarray(train_matrix[-5:, :].T[:, horizon_index] - 1, dtype=np.int64),
        ),
        "test": (
            np.ascontiguousarray(test_matrix[:40, :].T, dtype=np.float32),
            np.ascontiguousarray(test_matrix[-5:, :].T[:, horizon_index] - 1, dtype=np.int64),
        ),
    }
    del train_matrix, test_matrix
    FI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split, (features, labels) in books.items():
        np.save(cache[split][0], features)
        np.save(cache[split][1], labels)
    return books


def fi2010_split_arrays(
    book: np.ndarray,
    labels: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(window - 1, len(book), dtype=np.int64)
    return causal_features(book, indices), labels[indices]


def run_fi2010(args: argparse.Namespace) -> dict[str, object]:
    print("FI-2010 causal logistic regression...", flush=True)
    books = load_fi2010_books(args.fi_data_dir, args.horizon_index)
    train_x, train_y = fi2010_split_arrays(*books["train"], args.window)
    test_x, test_y = fi2010_split_arrays(*books["test"], args.window)
    del books
    model = fit_logistic(train_x, train_y)
    result = {
        "checkpoint_labels": "FI-2010 supplied labels, horizon_index=4 (k=100)",
        "window": args.window,
        "features": 25,
        **evaluate_model(model, {"train": (train_x, train_y), "test": (test_x, test_y)}, "FI-2010 logistic"),
    }
    return result


def run_lobster(args: argparse.Namespace) -> dict[str, object]:
    print("LOBSTER causal logistic regression...", flush=True)
    checkpoint = torch.load(args.lobster_checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    labeling = checkpoint["labeling"]
    alpha = float(labeling["alpha"])

    features_by_symbol: dict[str, np.ndarray] = {}
    mids_by_symbol: dict[str, np.ndarray] = {}
    books_by_symbol: dict[str, np.ndarray] = {}
    bounds_by_symbol: dict[str, dict[str, tuple[int, int]]] = {}
    for symbol in SYMBOLS:
        feature_path, mid_path, _ = transformed_paths(DEFAULT_PROCESSED_DIR, symbol)
        features = np.load(feature_path)
        mids = np.load(mid_path)
        features_by_symbol[symbol] = features
        mids_by_symbol[symbol] = mids
        books_by_symbol[symbol] = invert_lobster_book(features, mids)
        bounds_by_symbol[symbol] = split_bounds(
            len(features),
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
    used_alpha = alpha
    if abs(recomputed_alpha - alpha) > 1e-12:
        print(f"Using checkpoint alpha={alpha:.10g} instead of recomputed {recomputed_alpha:.10g}", flush=True)
        rebuilt: dict[str, list] = {"train": [], "validation": [], "test": []}
        from train_lobster_samples import PreparedSplit

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

    def stack_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        feature_parts = []
        label_parts = []
        for item in prepared[split]:
            absolute = item.offset + item.local_targets
            feature_parts.append(
                causal_features(books_by_symbol[item.symbol], absolute, min_index=item.offset)
            )
            label_parts.append(item.labels)
        return np.concatenate(feature_parts), np.concatenate(label_parts)

    train_x, train_y = stack_split("train")
    test_x, test_y = stack_split("test")
    model = fit_logistic(train_x, train_y)
    return {
        "checkpoint": str(args.lobster_checkpoint.resolve()),
        "alpha": used_alpha,
        "sequence_length": int(config["sequence_length"]),
        "horizon_events": int(config["horizon_events"]),
        "sample_stride": int(config["sample_stride"]),
        "features": 25,
        **evaluate_model(model, {"train": (train_x, train_y), "test": (test_x, test_y)}, "LOBSTER logistic"),
    }


def main() -> None:
    args = parse_args()
    output: dict[str, object] = {
        "experiment": "FI-2010 and LOBSTER causal logistic regression",
        "model": "StandardScaler + LogisticRegression(C=0.1, class_weight=balanced)",
        "features": "25 causal columns: lagged mid returns, trailing-mean returns, spread, depth imbalance, queue imbalance, book gaps, log depth",
    }
    if not args.skip_fi2010:
        output["fi2010"] = run_fi2010(args)
    if not args.skip_lobster:
        output["lobster"] = run_lobster(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
