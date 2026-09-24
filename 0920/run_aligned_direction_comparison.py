#!/usr/bin/env python3
"""Matched July DeepLOB / old Hybrid / L3 Hybrid direction heads.

The three models share one frozen pre-July DeepLOB encoder.  Their separately
refitted softmax heads use respectively its 64 coordinates, those coordinates
plus 28 causal L2 features, and those coordinates plus six OrderID features.
The July endpoint labels and quote starts are identical across models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hybrid_drift_adapter import fit_probability_move_calibrator  # noqa: E402

DATA = HERE / "output/direction_dataset"
SOURCE = HERE / "output/new_hybrid_signal/predictions"
OUT = HERE / "output/aligned_direction_comparison"
HORIZONS = (5, 20)
FEATURES = ("deeplob", "old_hybrid", "new_l3_hybrid")


def x_of(day: dict, name: str) -> np.ndarray:
    if name == "deeplob":
        return day["l2"][:, :64]
    if name == "old_hybrid":
        return day["l2"]
    if name == "new_l3_hybrid":
        return np.column_stack((day["l2"], day["l3"]))
    raise ValueError(name)


def fit(train: list[dict], horizon: int, name: str):
    x = np.concatenate([x_of(day, name) for day in train])
    y = np.concatenate([np.sign(day[f"move_{horizon}_ticks"]).astype(np.int8)
                        for day in train])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, solver="lbfgs", max_iter=300,
                           random_state=23),
    )
    model.fit(x, y)
    if not np.array_equal(model[-1].classes_, np.array([-1, 0, 1])):
        raise AssertionError("training lacks a direction class")
    return model


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = np.array([-1, 0, 1], dtype=np.int8)[np.argmax(p, axis=1)]
    directional = y != 0
    return {
        "quotes": int(len(y)),
        "accuracy_3class": float(accuracy_score(y, pred)),
        "balanced_accuracy_3class": float(balanced_accuracy_score(y, pred)),
        "macro_f1_3class": float(f1_score(y, pred, labels=[-1, 0, 1],
                                           average="macro", zero_division=0)),
        "log_loss_3class": float(log_loss(y, p, labels=[-1, 0, 1])),
        "flat_recall": float(np.mean(pred[y == 0] == 0)),
        "flat_share": float(np.mean(y == 0)),
        "directional_quotes": int(directional.sum()),
        "binary_score_accuracy_nonflat": float(np.mean(
            (p[directional, 2] >= p[directional, 0]) == (y[directional] > 0)
        )),
        "binary_score_accuracy_all": float(np.mean(
            (p[:, 2] >= p[:, 0]) == (y > 0)
        )),
    }


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    dates = [row["date"] for row in manifest["dates"]]
    data = {}
    for date in dates:
        with np.load(DATA / f"day_{date}.npz", allow_pickle=False) as a:
            data[date] = {k: a[k].copy() for k in a.files}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_day").mkdir(exist_ok=True)
    (OUT / "signals").mkdir(exist_ok=True)
    daily = []
    for ordinal, date in enumerate(dates[1:], 1):
        test = data[date]
        clean = bool(test["l3_available"])
        prior = [d for d in dates[:ordinal] if bool(data[d]["l3_available"])][-5:]
        if not prior or any(d >= date for d in prior):
            raise AssertionError("invalid rolling prior")
        train = [data[d] for d in prior]
        per_horizon = {}
        for horizon in HORIZONS:
            y = np.sign(test[f"move_{horizon}_ticks"]).astype(np.int8)
            probabilities = {}
            for name in FEATURES:
                if name == "new_l3_hybrid" and not clean:
                    probabilities[name] = probabilities["old_hybrid"].copy()
                    continue
                model = fit(train, horizon, name)
                probabilities[name] = model.predict_proba(x_of(test, name))
                if horizon == 20:
                    past_probs = np.concatenate([
                        model.predict_proba(x_of(data[d], name)) for d in prior
                    ])
                    past_moves = np.concatenate([
                        data[d]["move_20_ticks"] for d in prior
                    ])
                    calibrator = fit_probability_move_calibrator(
                        past_probs, past_moves, clip_ticks=3.0
                    )
                    per_horizon[name] = {
                        "current_ticks": calibrator.transform(probabilities[name]),
                        "prior_ticks": calibrator.transform(past_probs),
                        "current_hard_ticks": np.array([-1.0, 0.0, 1.0])[
                            np.argmax(probabilities[name], axis=1)
                        ],
                        "prior_hard_ticks": np.array([-1.0, 0.0, 1.0])[
                            np.argmax(past_probs, axis=1)
                        ],
                        "calibrator": {
                            "intercept": calibrator.intercept_ticks,
                            "coefficients": calibrator.coefficients_ticks.tolist(),
                            "samples": calibrator.samples,
                        },
                    }
                print("FIT", date, horizon, name, flush=True)
            if horizon == 20 and not clean:
                per_horizon["new_l3_hybrid"] = per_horizon["old_hybrid"]
            np.savez_compressed(
                OUT / "per_day" / f"{date}_h{horizon}.npz",
                actual=y,
                **{f"p_{name}": p.astype(np.float32)
                   for name, p in probabilities.items()},
            )
            daily.append({
                "date": date, "horizon": horizon, "prior_dates": prior,
                "l3_available": clean,
                "models": {name: metrics(y, p) for name, p in probabilities.items()},
            })
        with np.load(SOURCE / f"day_{date}.npz", allow_pickle=False) as a:
            source = {k: a[k].copy() for k in a.files}
        valid = source["has_100_snapshot_history"]
        if not np.array_equal(source["quote_indices"][valid], test["quote_indices"]):
            raise AssertionError(f"{date}: quote starts differ")
        if list(source["prior_dates"].astype(str)) != prior:
            raise AssertionError(f"{date}: prior dates differ")
        signals = {k: source[k] for k in (
            "quote_indices", "has_100_snapshot_history",
            "causal_horizon_seconds", "tick_hkd", "l3_available",
            "prior_dates", "prior_quote_indices",
            "prior_causal_horizon_seconds", "prior_tick_hkd",
            "prior_segment_ids", "prior_day_ordinals", "prior_day_quote_counts",
            "prior_quote_intervals_seconds",
        )}
        for name in FEATURES:
            branch = per_horizon[name]
            for suffix, current_key, prior_key in (
                ("", "current_ticks", "prior_ticks"),
                ("_hard", "current_hard_ticks", "prior_hard_ticks"),
            ):
                full = np.zeros(len(source["quote_indices"]), dtype=np.float64)
                full[valid] = branch[current_key]
                signals[f"expected_{name}{suffix}_ticks"] = full
                past_full = []
                for i, d in enumerate(prior):
                    count = int(source["prior_day_quote_counts"][i])
                    past = np.zeros(count, dtype=np.float64)
                    mask = data[d]["quote_indices"]
                    begin = int(sum(source["prior_day_quote_counts"][:i]))
                    prior_indices = source["prior_quote_indices"][begin:begin + count]
                    loc = np.searchsorted(prior_indices, mask)
                    if not np.array_equal(prior_indices[loc], mask):
                        raise AssertionError(f"{date}: prior quote alignment differs")
                    part_start = sum(len(data[z]["quote_indices"]) for z in prior[:i])
                    part_end = part_start + len(mask)
                    past[loc] = branch[prior_key][part_start:part_end]
                    past_full.append(past)
                signals[f"prior_{name}{suffix}_ticks"] = np.concatenate(past_full)
        if not clean and not np.array_equal(signals["expected_old_hybrid_ticks"],
                                            signals["expected_new_l3_hybrid_ticks"]):
            raise AssertionError("L3 gap fallback differs")
        np.savez_compressed(OUT / "signals" / f"day_{date}.npz", **signals)
        print("DAY", date, "done", flush=True)
    summary = []
    for horizon in HORIZONS:
        for subset in ("all", "l3_available", "l3_unavailable"):
            selected = [row for row in daily if row["horizon"] == horizon
                        and (subset == "all" or row["l3_available"] ==
                             (subset == "l3_available"))]
            for name in FEATURES:
                y, p = [], []
                for row in selected:
                    with np.load(OUT / "per_day" /
                                 f"{row['date']}_h{horizon}.npz", allow_pickle=False) as a:
                        y.append(a["actual"])
                        p.append(a[f"p_{name}"])
                summary.append({"horizon": horizon, "subset": subset,
                                "model": name, "days": len(selected),
                                **metrics(np.concatenate(y), np.concatenate(p))})
    result = {
        "protocol": "same quote starts and endpoint labels; at most five prior complete days",
        "architecture": "shared frozen pre-July 64-D DeepLOB encoder; separately retrained 3-class logistic heads",
        "features": {
            "deeplob": "64-D DeepLOB embedding",
            "old_hybrid": "DeepLOB embedding + 28 causal L2 engineering features",
            "new_l3_hybrid": "old Hybrid + six OrderID lifecycle features, with exact old-Hybrid gap fallback",
        },
        "horizons": HORIZONS,
        "probability_to_ticks": "past-only in-sample ridge calibration, predictions clipped to +/-3 ticks",
        "summary": summary, "daily": daily,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([s for s in summary if s["subset"] == "all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
