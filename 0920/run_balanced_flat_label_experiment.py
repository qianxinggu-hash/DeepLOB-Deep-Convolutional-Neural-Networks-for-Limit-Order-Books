#!/usr/bin/env python3
"""Walk-forward three-class labels with a prior-only flat band.

The frozen DeepLOB embeddings and aligned quote starts are unchanged. For each
test day, select one symmetric tick threshold on the preceding training days,
then refit the same three softmax heads used by the aligned comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss)

from hybrid_labeling import CLASSES, class_counts, labels, prior_flat_threshold
from run_aligned_direction_comparison import FEATURES, DATA, fit, x_of


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output/balanced_flat_labels"


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = CLASSES[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    return {
        "samples": int(len(y)),
        "true_counts": class_counts(y),
        "predicted_counts": class_counts(pred),
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES,
                                   average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
        "confidence_median": float(np.median(confidence)),
        "confidence_p99": float(np.quantile(confidence, 0.99)),
        "confidence_max": float(np.max(confidence)),
        "confidence_above_0_9": int(np.count_nonzero(confidence > 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-test-days", type=int)
    args = parser.parse_args()
    manifest = json.loads((DATA / "manifest.json").read_text())
    dates = [row["date"] for row in manifest["dates"]]
    if args.limit_test_days is not None:
        dates = dates[:1 + args.limit_test_days]
    data = {}
    for date in dates:
        with np.load(DATA / f"day_{date}.npz", allow_pickle=False) as archive:
            data[date] = {key: archive[key].copy() for key in archive.files}
    out = args.output_dir
    (out / "per_day").mkdir(parents=True, exist_ok=True)
    daily = []
    all_truth, all_probs = {name: [] for name in FEATURES}, {name: [] for name in FEATURES}
    baseline_hits = baseline_log_loss_weighted = total_test = 0
    for ordinal, date in enumerate(dates[1:], 1):
        test = data[date]
        clean = bool(test["l3_available"])
        prior = [d for d in dates[:ordinal] if bool(data[d]["l3_available"])][-5:]
        if not prior or any(d >= date for d in prior):
            raise AssertionError("training dates must precede the test date")
        train = [data[d] for d in prior]
        move_train = np.concatenate([d["move_20_ticks"] for d in train])
        threshold = prior_flat_threshold(move_train)
        y_train = labels(move_train, threshold)
        y_test = labels(test["move_20_ticks"], threshold)
        if any(count == 0 for count in class_counts(y_train)):
            raise AssertionError("training class missing")
        prior_shares = np.asarray(class_counts(y_train), dtype=np.float64) / len(y_train)
        baseline_class = int(CLASSES[np.argmax(prior_shares)])
        baseline_hits += int(np.count_nonzero(y_test == baseline_class))
        baseline_log_loss_weighted += float(log_loss(
            y_test, np.tile(prior_shares, (len(y_test), 1)), labels=CLASSES
        )) * len(y_test)
        total_test += len(y_test)
        results, probabilities = {}, {}
        for name in FEATURES:
            if name == "new_l3_hybrid" and not clean:
                probabilities[name] = probabilities["old_hybrid"].copy()
                results[name] = {**results["old_hybrid"], "fallback_to_old_hybrid": True}
            else:
                model = fit(train, 20, name, flat_threshold_ticks=threshold)
                p_train = model.predict_proba(np.concatenate([x_of(d, name) for d in train]))
                p_test = model.predict_proba(x_of(test, name))
                probabilities[name] = p_test
                results[name] = {
                    "train": metrics(y_train, p_train),
                    "test": metrics(y_test, p_test),
                    "fallback_to_old_hybrid": False,
                }
            all_truth[name].append(y_test)
            all_probs[name].append(probabilities[name])
        np.savez_compressed(
            out / "per_day" / f"{date}.npz",
            quote_indices=test["quote_indices"], actual=y_test,
            threshold_ticks=np.asarray(threshold),
            **{f"p_{name}": probabilities[name].astype(np.float32)
               for name in FEATURES},
        )
        row = {
            "date": date, "prior_dates": prior, "l3_available": clean,
            "flat_threshold_ticks": threshold,
            "train_counts": class_counts(y_train),
            "test_counts": class_counts(y_test),
            "prior_majority_class": baseline_class,
            "models": results,
        }
        daily.append(row)
        print("DONE", date, "threshold", threshold, flush=True)
    summary = {name: metrics(np.concatenate(all_truth[name]),
                             np.concatenate(all_probs[name]))
               for name in FEATURES}
    result = {
        "design": "prior-only symmetric flat band on 20-snapshot endpoint mid-price move; same frozen DeepLOB encoder and rolling heads as aligned comparison",
        "threshold_rule": "choose half-tick threshold minimizing the largest distance of prior down/flat/up shares from one-third; ties use smaller total distance then threshold",
        "class_order": CLASSES.tolist(),
        "causal_constant_baseline": {
            "majority_accuracy": baseline_hits / total_test,
            "prior_probability_log_loss": baseline_log_loss_weighted / total_test,
        },
        "test_dates": dates[1:], "summary": summary, "daily": daily,
    }
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
