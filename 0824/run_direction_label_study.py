#!/usr/bin/env python3
"""Walk-forward comparison of symmetric and genuinely forward direction labels.

The legacy label compares the next 20-snapshot average mid to the preceding
20-snapshot average mid.  The forward label compares the same next-20 average
mid to the mid available at the prediction timestamp.  Both use the same
causal features, eligible indices, model, dates, and rolling training window.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from directional_market_maker import classifier
from run_experiment import forward_returns, symmetric_returns
from run_improved_experiment import (
    SEQUENCE_LENGTH,
    causal_features,
    interval_indices,
    make_labels,
    metrics_from_prediction,
)
from run_july_month_strategy import MODEL_TRAIN_DAYS, _book_cache
from run_multiday_strategy import load_day


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
HORIZON = 20
STATIONARY_SHARE = 1.0 / 3.0
POLICY = "tolerant"
LABELS: dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "symmetric_past20_to_future20": symmetric_returns,
    "forward_current_to_future20": forward_returns,
}


@dataclass
class LabelDay:
    date: str
    book: np.ndarray
    segments: np.ndarray
    labels_source: np.ndarray
    eligible: np.ndarray


def _mid(book: np.ndarray) -> np.ndarray:
    return (book[:, 0].astype(np.float64) + book[:, 2]) / 2.0


def _observed_past_component(
    mids: np.ndarray, segments: np.ndarray, horizon: int
) -> np.ndarray:
    """Current mid relative to the past-horizon mean; known at time t."""

    result = np.full(len(mids), np.nan, dtype=np.float64)
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        if len(positions) < horizon:
            continue
        if np.any(np.diff(positions) != 1):
            raise ValueError(f"segment {segment} is not contiguous")
        values = mids[positions]
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        local = np.arange(horizon - 1, len(values), dtype=np.int64)
        past = (cumulative[local + 1] - cumulative[local - horizon + 1]) / horizon
        result[positions[local]] = values[local] / past - 1.0
    return result


def _load_usable_dates() -> list[str]:
    results_path = OUTPUT_DIR / "july_2026_tolerant_as_gp_drift_results.json"
    result = json.loads(results_path.read_text())
    return list(result["source_audit"]["clean_trading_dates"])


def _build_label_day(date: str, label_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray]) -> LabelDay:
    loaded = load_day(date, "tolerant_mbo", _book_cache(date, POLICY))
    values = label_function(_mid(loaded.book), loaded.segments, HORIZON)
    eligible = interval_indices(
        values,
        loaded.segments,
        0,
        len(loaded.book),
        SEQUENCE_LENGTH,
        HORIZON,
    )
    return LabelDay(date, loaded.book, loaded.segments, values, eligible)


def _fit(train_days: list[LabelDay]) -> tuple[object, float]:
    train_returns = np.concatenate([day.labels_source[day.eligible] for day in train_days])
    alpha = float(np.quantile(np.abs(train_returns), STATIONARY_SHARE))
    features = np.concatenate([causal_features(day.book, day.eligible) for day in train_days])
    labels = np.concatenate(
        [make_labels(day.labels_source, day.eligible, alpha) for day in train_days]
    )
    estimator = classifier()
    estimator.fit(features, labels)
    return estimator, alpha


def _per_day_split(day: LabelDay) -> tuple[np.ndarray, np.ndarray]:
    split = int(np.floor(len(day.book) * 0.70))
    train = interval_indices(
        day.labels_source, day.segments, 0, split, SEQUENCE_LENGTH, HORIZON
    )
    validation = interval_indices(
        day.labels_source,
        day.segments,
        split,
        len(day.book),
        SEQUENCE_LENGTH,
        HORIZON,
    )
    return train, validation


def _predict(estimator: object, alpha: float, day: LabelDay) -> dict[str, Any]:
    probabilities = estimator.predict_proba(causal_features(day.book, day.eligible))
    classes = estimator.named_steps["logisticregression"].classes_
    prediction = classes[np.argmax(probabilities, axis=1)].astype(np.int64)
    truth = make_labels(day.labels_source, day.eligible, alpha)
    metrics = metrics_from_prediction(truth, prediction)
    metrics.update(
        {
            "alpha": alpha,
            "samples": int(len(day.eligible)),
            "label_counts": np.bincount(truth, minlength=3).tolist(),
            "mean_probability_edge": float(
                np.mean(np.abs(probabilities[:, list(classes).index(2)] - probabilities[:, list(classes).index(0)]))
            ),
        }
    )
    return metrics


def _observed_path_proxy(alpha: float, day: LabelDay) -> dict[str, Any]:
    past_component = _observed_past_component(_mid(day.book), day.segments, HORIZON)
    truth = make_labels(day.labels_source, day.eligible, alpha)
    proxy = make_labels(past_component, day.eligible, alpha)
    metrics = metrics_from_prediction(truth, proxy)
    metrics["correlation_with_label_return"] = float(
        np.corrcoef(day.labels_source[day.eligible], past_component[day.eligible])[0, 1]
    )
    return metrics


def _aggregate(tests: list[dict[str, Any]], label_name: str) -> dict[str, Any]:
    rows = [test["labels"][label_name] for test in tests]
    total = sum(int(row["samples"]) for row in rows)
    result = {
        "test_days": len(rows),
        "samples": total,
        "weighted_accuracy": sum(float(row["accuracy"]) * int(row["samples"]) for row in rows) / total,
        "weighted_balanced_accuracy": sum(
            float(row["balanced_accuracy"]) * int(row["samples"]) for row in rows
        )
        / total,
        "weighted_macro_f1": sum(float(row["macro_f1"]) * int(row["samples"]) for row in rows) / total,
        "mean_daily_probability_edge": float(
            np.mean([float(row["mean_probability_edge"]) for row in rows])
        ),
    }
    if label_name == "symmetric_past20_to_future20":
        proxy_rows = [test["observed_past_proxy"] for test in tests]
        result["observed_past_proxy"] = {
            "weighted_accuracy": sum(
                float(row["accuracy"]) * int(test["labels"][label_name]["samples"])
                for row, test in zip(proxy_rows, tests)
            )
            / total,
            "weighted_macro_f1": sum(
                float(row["macro_f1"]) * int(test["labels"][label_name]["samples"])
                for row, test in zip(proxy_rows, tests)
            )
            / total,
            "mean_correlation_with_label_return": float(
                np.mean([float(row["correlation_with_label_return"]) for row in proxy_rows])
            ),
        }
    return result


def _validate(output: dict[str, Any]) -> dict[str, bool]:
    tests = output["tests"]
    same_indices = all(
        int(test["labels"]["symmetric_past20_to_future20"]["samples"])
        == int(test["labels"]["forward_current_to_future20"]["samples"])
        for test in tests
    )
    chronological = all(
        all(train_date < test["test_date"] for train_date in test["train_dates"])
        for test in tests
    )
    valid_probabilities = all(
        0.0 <= float(metric) <= 1.0
        for test in tests
        for label in test["labels"].values()
        for metric in (label["accuracy"], label["balanced_accuracy"], label["macro_f1"])
    )
    return {
        "same_eligible_samples_for_both_labels": same_indices,
        "chronological_training_only": chronological,
        "classification_metrics_are_bounded": valid_probabilities,
        "all_passed": same_indices and chronological and valid_probabilities,
    }


def _write_daily_csv(tests: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for test in tests:
        row: dict[str, Any] = {
            "date": test["test_date"],
            "train_dates": ";".join(test["train_dates"]),
        }
        for label_name, metrics in test["labels"].items():
            prefix = label_name.replace("_to_", "_")
            for key in ("samples", "alpha", "accuracy", "balanced_accuracy", "macro_f1", "mean_probability_edge"):
                row[f"{prefix}_{key}"] = metrics[key]
        proxy = test["observed_past_proxy"]
        row["symmetric_observed_past_proxy_accuracy"] = proxy["accuracy"]
        row["symmetric_observed_past_proxy_macro_f1"] = proxy["macro_f1"]
        row["symmetric_observed_past_proxy_correlation"] = proxy[
            "correlation_with_label_return"
        ]
        rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    usable_dates = _load_usable_dates()
    if len(usable_dates) < 2:
        raise RuntimeError("fewer than two usable dates")
    all_days = {
        label_name: {
            date: _build_label_day(date, label_function)
            for date in usable_dates
        }
        for label_name, label_function in LABELS.items()
    }
    initial: dict[str, Any] = {}
    tests: list[dict[str, Any]] = []
    for label_name, days in all_days.items():
        first_day = days[usable_dates[0]]
        fit_indices, validation_indices = _per_day_split(first_day)
        initial_train = LabelDay(
            first_day.date,
            first_day.book,
            first_day.segments,
            first_day.labels_source,
            fit_indices,
        )
        initial_validation = LabelDay(
            first_day.date,
            first_day.book,
            first_day.segments,
            first_day.labels_source,
            validation_indices,
        )
        estimator, alpha = _fit([initial_train])
        initial[label_name] = _predict(estimator, alpha, initial_validation)

    for position, test_date in enumerate(usable_dates[1:], start=1):
        train_dates = usable_dates[max(0, position - MODEL_TRAIN_DAYS) : position]
        test: dict[str, Any] = {"test_date": test_date, "train_dates": train_dates, "labels": {}}
        for label_name, days in all_days.items():
            estimator, alpha = _fit([days[date] for date in train_dates])
            test["labels"][label_name] = _predict(estimator, alpha, days[test_date])
        test["observed_past_proxy"] = _observed_path_proxy(
            float(test["labels"]["symmetric_past20_to_future20"]["alpha"]),
            all_days["symmetric_past20_to_future20"][test_date],
        )
        tests.append(test)
        symmetric = test["labels"]["symmetric_past20_to_future20"]
        forward = test["labels"]["forward_current_to_future20"]
        print(
            "TEST",
            test_date,
            f"symmetric_acc={symmetric['accuracy']:.4f}",
            f"forward_acc={forward['accuracy']:.4f}",
            flush=True,
        )

    output: dict[str, Any] = {
        "as_of": "2026-08-31",
        "instrument": "HK 07709",
        "design": {
            "reconstruction_policy": POLICY,
            "usable_dates": usable_dates,
            "initial_period": f"{usable_dates[0]} first 70% fit / last 30% initial validation only",
            "full_day_out_of_sample_tests": usable_dates[1:],
            "horizon_snapshots": HORIZON,
            "features": "same 25 causal LOB features for both labels",
            "model": "same class-balanced multinomial logistic regression for both labels",
            "rolling_training_days": MODEL_TRAIN_DAYS,
            "class_threshold": "training-only 33.33rd percentile of absolute return separately for each label",
            "label_definitions": {
                "symmetric_past20_to_future20": "mean(mid[t+1:t+20]) / mean(mid[t-19:t]) - 1",
                "forward_current_to_future20": "mean(mid[t+1:t+20]) / mid[t] - 1",
                "observed_past_proxy": "mid[t] / mean(mid[t-19:t]) - 1; this is available at t and is reported only as a leakage/overlap diagnostic",
            },
        },
        "initial_validation": initial,
        "tests": tests,
        "aggregate": {
            label_name: _aggregate(tests, label_name) for label_name in LABELS
        },
    }
    output["validation"] = _validate(output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "july_2026_tolerant_direction_label_comparison_results.json"
    daily_path = OUTPUT_DIR / "july_2026_tolerant_direction_label_comparison_daily.csv"
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    _write_daily_csv(tests, daily_path)
    print("RESULT", result_path, flush=True)
    print("DAILY", daily_path, flush=True)
    print("VALID", json.dumps(output["validation"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
