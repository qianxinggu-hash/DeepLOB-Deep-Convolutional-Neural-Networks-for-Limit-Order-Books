#!/usr/bin/env python3
"""Evaluate up/flat/down prediction without conditioning on a future move."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import benchmark_direction_methods as base


HERE = Path(__file__).resolve().parent
DATA = HERE / "output/direction_dataset"
OUTPUT = HERE / "output/direction_three_way"
HORIZONS = (5, 20)
LABELS = (-1, 0, 1)


def measure(actual: np.ndarray, prediction: np.ndarray) -> dict:
    recalls = recall_score(actual, prediction, labels=LABELS, average=None,
                           zero_division=0)
    return {
        "quotes": int(len(actual)),
        "accuracy": float(accuracy_score(actual, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, labels=LABELS,
                                   average="macro", zero_division=0)),
        "down_recall": float(recalls[0]),
        "flat_recall": float(recalls[1]),
        "up_recall": float(recalls[2]),
        "true_flat_share": float(np.mean(actual == 0)),
        "predicted_flat_share": float(np.mean(prediction == 0)),
    }


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    dates = [r["date"] for r in manifest["dates"]]
    data = base.load_data(dates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_day = OUTPUT / "per_day"
    per_day.mkdir(exist_ok=True)
    daily = []
    for ordinal, date in enumerate(dates[1:], start=1):
        prior = [d for d in dates[:ordinal] if bool(data[d]["l3_available"])][-5:]
        train_days = [data[d] for d in prior]
        test = data[date]
        available = bool(test["l3_available"])
        for horizon in HORIZONS:
            out = per_day / f"{date}_h{horizon}"
            if out.with_suffix(".json").exists() and out.with_suffix(".npz").exists():
                daily.append(json.loads(out.with_suffix(".json").read_text()))
                continue
            train_move = np.concatenate([d[f"move_{horizon}_ticks"] for d in train_days])
            train_y = np.sign(train_move).astype(np.int8)
            true = np.sign(test[f"move_{horizon}_ticks"]).astype(np.int8)
            predictions = {}
            for balancing in (False, True):
                for feature in ("l2", "l2_l3_6"):
                    name = ("balanced" if balancing else "unweighted") + "_" + feature
                    if feature != "l2" and not available:
                        predictions[name] = predictions[
                            ("balanced" if balancing else "unweighted") + "_l2"
                        ].copy()
                        continue
                    x_train = base.features(train_days, feature)
                    x_test = base.features([test], feature)
                    model = make_pipeline(
                        StandardScaler(), LogisticRegression(
                            C=0.05, solver="lbfgs", max_iter=200,
                            class_weight="balanced" if balancing else None,
                            random_state=23,
                        ),
                    )
                    model.fit(x_train, train_y)
                    predictions[name] = model.predict(x_test).astype(np.int8)
                    print("FIT", date, horizon, name, flush=True)
            np.savez_compressed(out.with_suffix(".npz"), actual=true,
                                **{f"prediction_{k}": v for k, v in predictions.items()})
            row = {"date": date, "horizon_snapshots": horizon,
                   "l3_available": available, "prior_dates": prior,
                   "methods": {name: measure(true, prediction)
                               for name, prediction in predictions.items()}}
            out.with_suffix(".json").write_text(
                json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            daily.append(row)
            print("DONE", date, horizon, flush=True)
    summary = []
    for horizon in HORIZONS:
        for subset in ("all", "l3_available", "l3_unavailable"):
            chosen = [r for r in daily if r["horizon_snapshots"] == horizon
                      and (subset == "all" or r["l3_available"] ==
                           (subset == "l3_available"))]
            for name in chosen[0]["methods"]:
                ys, predictions = [], []
                for r in chosen:
                    with np.load(per_day / f"{r['date']}_h{horizon}.npz",
                                 allow_pickle=False) as archive:
                        ys.append(archive["actual"])
                        predictions.append(archive[f"prediction_{name}"])
                summary.append({"horizon_snapshots": horizon, "subset": subset,
                                "method": name, "days": len(chosen),
                                **measure(np.concatenate(ys), np.concatenate(predictions))})
    result = {
        "train_protocol": "at most five prior complete-MBO days",
        "labels": {"-1": "down", "0": "flat", "1": "up"},
        "model": "StandardScaler + 3-class LogisticRegression(C=0.05); optional balanced class weights",
        "summary": summary, "daily": daily,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps([s for s in summary if s["subset"] == "all"],
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
