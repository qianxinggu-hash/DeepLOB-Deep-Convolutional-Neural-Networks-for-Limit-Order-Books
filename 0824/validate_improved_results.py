#!/usr/bin/env python3
"""Independent checkpoint/model reload validation for the improved experiment."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from run_experiment import symmetric_returns, transform_features
from run_improved_experiment import (
    MODEL_FACTORIES,
    PreparedPartition,
    causal_features,
    evaluate,
    interval_indices,
    log_softmax_numpy,
    make_labels,
    make_loader,
    metrics_from_prediction,
)
from train_pytorch_mac import choose_device


HERE = Path(__file__).resolve().parent


def close_metric(left: dict[str, object], right: dict[str, object]) -> dict[str, bool]:
    checks = {
        key: bool(np.isclose(left[key], right[key], rtol=0, atol=1e-12))
        for key in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    checks["confusion_matrix_exact"] = left["confusion_matrix"] == right["confusion_matrix"]
    return checks


def main() -> None:
    results_path = HERE / "output/improved_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(results["artifacts"]["deeplob_checkpoint"])
    logistic_path = Path(results["artifacts"]["logistic_model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with np.load(Path(results["source_cache"]), allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        segments = archive["segments"].astype(np.int16, copy=False)
        send_times = archive["send_times"].astype(np.int64, copy=False)

    split_index = int(checkpoint["split_index"])
    horizon = int(checkpoint["horizon"])
    alpha = float(checkpoint["alpha"])
    mids = (features[:, 0].astype(np.float64) + features[:, 2]) / 2
    returns = symmetric_returns(mids, segments, horizon)
    test_indices = interval_indices(
        returns, segments, split_index, len(features), 100, horizon
    )
    test_labels = make_labels(returns, test_indices, alpha)
    test_auxiliary = causal_features(features, test_indices)
    transformed = transform_features(features)
    normalized = np.ascontiguousarray(
        (transformed - checkpoint["book_mean"]) / checkpoint["book_std"],
        dtype=np.float32,
    )
    partition = PreparedPartition(test_indices, test_labels, test_auxiliary)
    loader = make_loader(
        normalized,
        partition,
        checkpoint["auxiliary_mean"],
        checkpoint["auxiliary_std"],
        256,
        False,
        int(checkpoint["seed"]),
    )
    device = choose_device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_FACTORIES[str(checkpoint["model_name"])]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    loss, neural_metrics, neural_logits, truth = evaluate(model, loader, criterion, device)

    logistic = joblib.load(logistic_path)
    logistic_log_probability = logistic.predict_log_proba(test_auxiliary)
    logistic_metrics = metrics_from_prediction(
        truth, logistic_log_probability.argmax(axis=1)
    )
    calibration = checkpoint["calibration"]
    combined = (
        float(calibration["neural_weight"]) * log_softmax_numpy(neural_logits)
        + float(calibration["logistic_weight"]) * logistic_log_probability
    )
    combined[:, 1] += float(calibration["stationary_logit_bias"])
    ensemble_metrics = metrics_from_prediction(truth, combined.argmax(axis=1))

    recorded = results["final_test"]
    checks = {
        "neural": close_metric(neural_metrics, recorded["selected_deeplob"]),
        "logistic": close_metric(logistic_metrics, recorded["causal_logistic"]),
        "ensemble": close_metric(
            ensemble_metrics, recorded["validation_selected_ensemble"]
        ),
        "neural_loss": bool(
            np.isclose(loss, recorded["neural_loss"], rtol=0, atol=1e-7)
        ),
        "test_targets_exact": len(test_indices)
        == int(results["final_data"]["test_targets"]),
        "test_input_after_split": int(test_indices[0]) - 99 >= split_index,
        "test_future_before_segment_end": bool(np.isfinite(returns[test_indices]).all()),
        "split_timestamp_exact": int(send_times[split_index])
        == int(results["outer_split"]["split_send_time_hkt"]),
    }
    all_passed = all(
        value
        for item in checks.values()
        for value in (item.values() if isinstance(item, dict) else [item])
    )
    original = json.loads((HERE / "output/results.json").read_text(encoding="utf-8"))
    validation = {
        "overall_assessment": "ready as a diagnostic model comparison; not a trading claim",
        "all_reproduction_checks_passed": all_passed,
        "checks": checks,
        "recomputed": {
            "neural_loss": loss,
            "selected_deeplob": neural_metrics,
            "causal_logistic": logistic_metrics,
            "validation_selected_ensemble": ensemble_metrics,
        },
        "improvement_vs_original_deeplob_percentage_points": {
            "accuracy": 100
            * (
                neural_metrics["accuracy"]
                - original["test"]["metrics"]["accuracy"]
            ),
            "balanced_accuracy": 100
            * (
                neural_metrics["balanced_accuracy"]
                - original["test"]["metrics"]["balanced_accuracy"]
            ),
            "macro_f1": 100
            * (
                neural_metrics["macro_f1"]
                - original["test"]["metrics"]["macro_f1"]
            ),
        },
        "remaining_blockers": [
            "earlier baseline work already exposed this final 30%, so a new day is required for a pristine confirmation",
            "single-day overlapping samples are not independent",
            "model metrics do not include executable-price or cost assumptions",
        ],
    }
    output = HERE / "output/improved_validation.json"
    output.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
