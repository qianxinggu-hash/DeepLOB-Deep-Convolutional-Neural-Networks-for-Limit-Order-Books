#!/usr/bin/env python3
"""Independent data, split, checkpoint, and metric validation for 0824."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from run_experiment import (
    CLASS_NAMES,
    WindowDataset,
    choose_device,
    constant_baseline,
    prepare_data,
    run_loader,
)
from train_pytorch_mac import DeepLOB


HERE = Path(__file__).resolve().parent


def main() -> None:
    results_path = HERE / "output/results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    cache_path = Path(results["reconstruction_cache"])
    checkpoint_path = Path(results["checkpoint"])
    with np.load(cache_path, allow_pickle=False) as archive:
        features = archive["features"]
        send_times = archive["send_times"]
        segments = archive["segments"]

    price_order_violations = 0
    for level in range(9):
        price_order_violations += int(
            np.sum(features[:, level * 4] >= features[:, (level + 1) * 4])
        )
        price_order_violations += int(
            np.sum(features[:, level * 4 + 2] <= features[:, (level + 1) * 4 + 2])
        )
    crossed_or_locked = int(np.sum(features[:, 2] >= features[:, 0]))
    nonpositive_sizes = int(
        np.sum(features[:, 1::4] <= 0) + np.sum(features[:, 3::4] <= 0)
    )
    segment_contiguous = all(
        np.all(np.diff(np.flatnonzero(segments == segment)) == 1)
        for segment in np.unique(segments)
    )

    ratio = float(results["split"]["ratio"][0])
    sequence_length = int(results["features"]["sequence_length"])
    horizon = int(results["labeling"]["horizon_snapshots"])
    stationary_share = float(results["labeling"]["target_stationary_share"])
    prepared = prepare_data(
        features, segments, ratio, sequence_length, horizon, stationary_share
    )
    split_index = int(results["split"]["split_index"])
    leakage_checks = {
        "train_label_future_before_split": bool(
            int(prepared.train_indices[-1]) + horizon < split_index
        ),
        "test_input_starts_at_or_after_split": bool(
            int(prepared.test_indices[0]) - sequence_length + 1 >= split_index
        ),
        "train_test_target_disjoint": bool(
            not np.intersect1d(prepared.train_indices, prepared.test_indices).size
        ),
        "alpha_matches": bool(
            np.isclose(prepared.alpha, float(results["labeling"]["alpha"]), rtol=0, atol=1e-15)
        ),
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DeepLOB(3)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = WindowDataset(
        prepared.normalized, prepared.test_indices, prepared.test_labels, sequence_length
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    train_counts = np.bincount(prepared.train_labels, minlength=3).astype(np.float64)
    weights = train_counts.sum() / (3 * train_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    loss, metrics = run_loader(model, loader, criterion, device)
    metrics["loss"] = loss
    recorded = results["test"]["metrics"]
    metric_checks = {
        key: bool(np.isclose(metrics[key], recorded[key], rtol=0, atol=1e-12))
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "loss")
    }
    metric_checks["confusion_matrix_exact"] = (
        metrics["confusion_matrix"] == recorded["confusion_matrix"]
    )

    constants = {
        CLASS_NAMES[index]: constant_baseline(prepared.test_labels, index)
        for index in range(3)
    }
    validation = {
        "overall_assessment": "share with caveats; valid diagnostic, not a usable trading model",
        "data_quality": {
            "snapshots": int(len(features)),
            "finite_values": bool(np.isfinite(features).all()),
            "strictly_increasing_sample_times": bool(np.all(np.diff(send_times) > 0)),
            "price_order_violations_across_levels": price_order_violations,
            "crossed_or_locked_snapshots": crossed_or_locked,
            "nonpositive_resting_sizes": nonpositive_sizes,
            "segments_contiguous": bool(segment_contiguous),
        },
        "split_and_label_checks": leakage_checks,
        "checkpoint_reload_metric_checks": metric_checks,
        "recomputed_test_metrics": metrics,
        "constant_baselines": constants,
        "uniform_random_expected_accuracy": 1 / 3,
        "interpretation": {
            "model_accuracy_minus_stationary_constant": metrics["accuracy"]
            - constants["stationary"]["accuracy"],
            "model_macro_f1_minus_stationary_constant": metrics["macro_f1"]
            - constants["stationary"]["macro_f1"],
            "model_balanced_accuracy_minus_uniform_random_expectation": metrics[
                "balanced_accuracy"
            ]
            - 1 / 3,
        },
        "blockers_for_trading_claim": [
            "one day only",
            "no independent validation day",
            "strongly overlapping and autocorrelated windows",
            "no fees, fill model, latency, or market-impact backtest",
            "test performance is near chance and below the stationary constant on accuracy",
        ],
    }
    output = HERE / "output/validation.json"
    output.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
