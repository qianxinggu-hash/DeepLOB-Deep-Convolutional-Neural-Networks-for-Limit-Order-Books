#!/usr/bin/env python3
"""Walk-forward single-feature direction ablation on complete OrderID days."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import benchmark_direction_methods as base
from l3_order_signal import FEATURE_NAMES


HERE = Path(__file__).resolve().parent
DATA = HERE / "output/direction_dataset"
OUTPUT = HERE / "output/direction_l3_ablation"
HORIZONS = (5, 20)


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    dates = [r["date"] for r in manifest["dates"]]
    data = base.load_data(dates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_day = OUTPUT / "per_day"
    per_day.mkdir(exist_ok=True)
    daily = []
    for ordinal, date in enumerate(dates[1:], start=1):
        if not bool(data[date]["l3_available"]):
            continue
        prior = [d for d in dates[:ordinal] if bool(data[d]["l3_available"])][-5:]
        for horizon in HORIZONS:
            out = per_day / f"{date}_h{horizon}"
            if out.with_suffix(".json").exists() and out.with_suffix(".npz").exists():
                daily.append(json.loads(out.with_suffix(".json").read_text()))
                continue
            move_train = np.concatenate([data[d][f"move_{horizon}_ticks"] for d in prior])
            valid = move_train != 0
            target = (move_train[valid] > 0).astype(np.int8)
            move = data[date][f"move_{horizon}_ticks"]
            train_l3 = np.concatenate([data[d]["l3"] for d in prior])[valid]
            test_l3 = data[date]["l3"]
            scores = {}
            for index, name in enumerate(FEATURE_NAMES):
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=0.05, solver="liblinear", max_iter=200, random_state=23
                ))
                model.fit(train_l3[:, index:index+1], target)
                scores[name] = model.predict_proba(test_l3[:, index:index+1])[:, 1]
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=0.05, solver="liblinear", max_iter=200, random_state=23
            ))
            model.fit(train_l3, target)
            scores["all_six"] = model.predict_proba(test_l3)[:, 1]
            np.savez_compressed(out.with_suffix(".npz"), move_ticks=move,
                                **{f"score_{k}": v.astype(np.float32)
                                   for k, v in scores.items()})
            row = {"date": date, "horizon_snapshots": horizon,
                   "prior_dates": prior,
                   "methods": {name: base.direction_metrics(move, score, True)
                               for name, score in scores.items()}}
            out.with_suffix(".json").write_text(
                json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            daily.append(row)
            print("DONE", date, horizon, flush=True)
    summary = []
    for horizon in HORIZONS:
        chosen = [r for r in daily if r["horizon_snapshots"] == horizon]
        for name in (*FEATURE_NAMES, "all_six"):
            ys, scores = [], []
            for r in chosen:
                with np.load(per_day / f"{r['date']}_h{horizon}.npz",
                             allow_pickle=False) as archive:
                    ys.append(archive["move_ticks"])
                    scores.append(archive[f"score_{name}"])
            summary.append({"horizon_snapshots": horizon, "feature": name,
                            "days": len(chosen),
                            **base.direction_metrics(np.concatenate(ys),
                                                     np.concatenate(scores), True)})
    result = {"train_protocol": "at most five prior complete-MBO days",
              "test_dates": [d for d in dates[1:] if bool(data[d]["l3_available"])],
              "feature_names": FEATURE_NAMES,
              "summary": summary, "daily": daily}
    (OUTPUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
