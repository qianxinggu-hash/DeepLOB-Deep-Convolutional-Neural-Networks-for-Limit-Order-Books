#!/usr/bin/env python3
"""Audit whether the Hybrid label measures a tradable forward price move.

The DeepLOB Equation (4) label compares a future mean with a past mean.  Its
past-to-current component is already known at prediction time.  This script
measures how well that component alone predicts the classification label and
how little it predicts the point-to-point move used by the GP drift replay.
Realized t -> t+20 timestamps are read solely for this ex-post diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_drift_adapter import prediction_horizon_seconds  # noqa: E402
from train_7709_deeplob import (  # noqa: E402
    DayData,
    confusion_metrics,
    eligible_targets,
    equation4_returns,
)


DEFAULT_TRAINING_RESULT = HERE / "output/hybrid_month/pre_july_training.json"
DEFAULT_PROCESSED_DIR = ROOT / "data/processed/7709"
DEFAULT_OUTPUT = HERE / "output/hybrid_month/label_target_diagnostic.json"


def load_day(processed_dir: Path, date: str) -> tuple[DayData, np.ndarray]:
    path = processed_dir / f"hk07709_{date}_s10_lob.npz"
    with np.load(path, allow_pickle=False) as archive:
        day = DayData(
            date=date,
            path=path,
            features=archive["features"].astype(np.float32, copy=False),
            sessions=archive["sessions"].astype(np.uint8, copy=False),
            metadata=json.loads(str(archive["metadata"])),
        )
        send_times = archive["send_times"].astype(np.int64, copy=False)
    return day, send_times


def past_mean(day: DayData, k: int) -> np.ndarray:
    """Mean of the k prices ending at t, matching equation4_returns."""
    mid = day.mid_prices
    result = np.full(len(mid), np.nan, dtype=np.float64)
    for session in np.unique(day.sessions):
        positions = np.flatnonzero(day.sessions == session)
        values = mid[positions]
        cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
        local = np.arange(k - 1, len(values), dtype=np.int64)
        result[positions[local]] = (
            cumulative[local + 1] - cumulative[local - k + 1]
        ) / k
    return result


def labels(returns: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(returns < -alpha, 0, np.where(returns > alpha, 2, 1))


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def duration_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(len(values)),
        "p25_seconds": float(np.quantile(values, 0.25)),
        "median_seconds": float(np.median(values)),
        "p75_seconds": float(np.quantile(values, 0.75)),
    }


def analyze_day(
    day: DayData,
    send_times: np.ndarray,
    k: int,
    sequence_length: int,
    stride: int,
    alpha: float,
    split: str,
    hybrid_metrics: dict | None,
) -> tuple[dict, np.ndarray]:
    returns = equation4_returns(day, k)
    indices = eligible_targets(day, returns, sequence_length, stride)
    mid = day.mid_prices
    known = mid[indices] / past_mean(day, k)[indices] - 1.0
    actual = returns[indices]
    forward_endpoint = mid[indices + k] - mid[indices]
    true_labels = labels(actual, alpha)
    known_labels = labels(known, alpha)
    confusion = np.bincount(
        true_labels * 3 + known_labels, minlength=9
    ).reshape(3, 3)
    metrics = confusion_metrics(confusion)
    durations = prediction_horizon_seconds(
        send_times, day.sessions, indices, k
    )
    if hybrid_metrics is not None and len(indices) != hybrid_metrics["samples"]:
        raise AssertionError(
            f"{day.date}: diagnostic samples {len(indices)} differ from "
            f"Hybrid report {hybrid_metrics['samples']}"
        )
    result = {
        "split": split,
        "processed_source": str(day.path.relative_to(ROOT)),
        "sample_stride": stride,
        "samples": int(len(indices)),
        "known_component_baseline": {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "confusion_matrix": confusion.tolist(),
        },
        "hybrid_report": (
            {
                "accuracy": float(hybrid_metrics["accuracy"]),
                "macro_f1": float(hybrid_metrics["macro_f1"]),
                "confusion_matrix": hybrid_metrics["confusion_matrix"],
            }
            if hybrid_metrics is not None
            else None
        ),
        "known_component_vs_equation4_label_correlation": correlation(
            known, actual
        ),
        "known_component_vs_forward_endpoint_correlation": correlation(
            known, forward_endpoint
        ),
        "equation4_label_vs_forward_endpoint_correlation": correlation(
            actual, forward_endpoint
        ),
        "realized_forward_20_snapshot_duration": duration_summary(durations),
    }
    return result, durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--training-result", type=Path, default=DEFAULT_TRAINING_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    training_result = json.loads(args.training_result.read_text(encoding="utf-8"))
    split = training_result["split"]
    k = int(training_result["labeling"]["k"])
    alpha = float(training_result["labeling"]["alpha"])
    # The checkpoint's auxiliary feature reaches 100 snapshots back, so the
    # first eligible target is at session offset 100 (sequence_length=101).
    sequence_length = 101
    train_dates = split["train_dates"]
    selected_dates = [*train_dates, split["validation_date"], split["test_date"]]
    results = {}
    training_durations = []
    for date in selected_dates:
        if date in train_dates:
            split_name = "train"
            stride = 5
            hybrid_metrics = None
        elif date == split["validation_date"]:
            split_name = "validation"
            stride = 5
            hybrid_metrics = training_result["training"]["validation"]
        else:
            split_name = "test"
            stride = 1
            hybrid_metrics = training_result["test"]
        day, send_times = load_day(args.processed_dir, date)
        results[date], durations = analyze_day(
            day, send_times, k, sequence_length, stride, alpha,
            split_name, hybrid_metrics,
        )
        if split_name == "train":
            training_durations.append(durations)

    pooled_training = duration_summary(np.concatenate(training_durations))
    july_test = results[split["test_date"]][
        "realized_forward_20_snapshot_duration"
    ]
    output = {
        "purpose": "Audit Hybrid classification label against the GP drift endpoint target",
        "label_definition": "mean(mid[t+1:t+k]) / mean(mid[t-k+1:t]) - 1",
        "known_component_definition": "mid[t] / mean(mid[t-k+1:t]) - 1",
        "gp_drift_calibration_target": "(mid[t+k] - mid[t]) / tick_hkd",
        "note": "The known component uses only information available at t. Realized t->t+k timestamps are used only for ex-post duration comparison.",
        "k_snapshots": k,
        "alpha_return": alpha,
        "training_report_source": str(args.training_result.relative_to(ROOT)),
        "dates": results,
        "pooled_training_realized_forward_20_snapshot_duration": pooled_training,
        "july_test_realized_forward_20_snapshot_duration": july_test,
        "training_to_july_median_duration_ratio": (
            pooled_training["median_seconds"] / july_test["median_seconds"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_known_macro_f1": results[split["validation_date"]][
                    "known_component_baseline"
                ]["macro_f1"],
                "validation_hybrid_macro_f1": results[split["validation_date"]][
                    "hybrid_report"
                ]["macro_f1"],
                "july_known_forward_endpoint_correlation": results[
                    split["test_date"]
                ]["known_component_vs_forward_endpoint_correlation"],
                "duration_ratio": output["training_to_july_median_duration_ratio"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
