#!/usr/bin/env python3
"""Test whether ignoring or upweighting large prior moves improves direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import benchmark_direction_methods as base


HERE = Path(__file__).resolve().parent
DATA = HERE / "output/direction_dataset"
OUTPUT = HERE / "output/direction_label_variants"
HORIZONS = (5, 10, 20)
VARIANTS = ("abs1_training", "magnitude_weighted")


def fit(x_train: np.ndarray, move_train: np.ndarray, x_test: np.ndarray,
        variant: str) -> np.ndarray:
    selected = move_train != 0
    if variant == "abs1_training":
        selected &= np.abs(move_train) >= 1
    y = (move_train[selected] > 0).astype(np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError("one direction in training")
    sample_weight = None
    if variant == "magnitude_weighted":
        sample_weight = np.clip(np.abs(move_train[selected]), 0.5, 5.0)
        sample_weight = sample_weight / np.mean(sample_weight)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(
            C=0.05, solver="liblinear", max_iter=200, random_state=23
        ),
    )
    if sample_weight is None:
        model.fit(x_train[selected], y)
    else:
        model.fit(x_train[selected], y,
                  logisticregression__sample_weight=sample_weight)
    return model.predict_proba(x_test)[:, 1]


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
            result_path = per_day / f"{date}_h{horizon}.json"
            score_path = per_day / f"{date}_h{horizon}.npz"
            if result_path.exists() and score_path.exists():
                daily.append(json.loads(result_path.read_text()))
                continue
            move_train = np.concatenate([d[f"move_{horizon}_ticks"] for d in train_days])
            move = test[f"move_{horizon}_ticks"]
            x_l2 = base.features(train_days, "l2")
            t_l2 = base.features([test], "l2")
            x_fused = base.features(train_days, "l2_l3_6") if available else None
            t_fused = base.features([test], "l2_l3_6") if available else None
            scores = {}
            metrics = {}
            for variant in VARIANTS:
                book = fit(x_l2, move_train, t_l2, variant)
                fused = (fit(x_fused, move_train, t_fused, variant)
                         if available else book.copy())
                for feature, score in (("l2", book), ("l2_l3_6", fused)):
                    name = f"{variant}_{feature}"
                    scores[name] = score.astype(np.float32)
                    metrics[name] = base.direction_metrics(move, score, True)
            np.savez_compressed(score_path, move_ticks=move,
                                **{f"score_{k}": v for k, v in scores.items()})
            row = {"date": date, "horizon_snapshots": horizon,
                   "l3_available": available, "prior_dates": prior,
                   "methods": metrics}
            result_path.write_text(json.dumps(row, ensure_ascii=False,
                                              allow_nan=False, indent=2) + "\n")
            daily.append(row)
            print("DONE", date, horizon, flush=True)
    summary = []
    for horizon in HORIZONS:
        for subset in ("all", "l3_available", "l3_unavailable"):
            chosen = [r for r in daily if r["horizon_snapshots"] == horizon
                      and (subset == "all" or r["l3_available"] ==
                           (subset == "l3_available"))]
            for variant in VARIANTS:
                for feature in ("l2", "l2_l3_6"):
                    name = f"{variant}_{feature}"
                    ys, scores = [], []
                    for r in chosen:
                        with np.load(per_day / f"{r['date']}_h{horizon}.npz",
                                     allow_pickle=False) as archive:
                            ys.append(archive["move_ticks"])
                            scores.append(archive[f"score_{name}"])
                    metric = base.direction_metrics(np.concatenate(ys),
                                                    np.concatenate(scores), True)
                    summary.append({"horizon_snapshots": horizon, "subset": subset,
                                    "method": name, "days": len(chosen), **metric})
    result = {
        "target": "endpoint sign; flat excluded from binary metrics",
        "train_dates": "at most five prior complete-MBO days",
        "variants": {
            "abs1_training": "train on prior abs move >= 1 tick; evaluate all nonflat moves",
            "magnitude_weighted": "train on all prior nonflat moves with abs-tick weights clipped to 0.5–5 and mean-normalized",
        },
        "daily": daily, "summary": summary,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps([s for s in summary if s["subset"] == "all"],
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
