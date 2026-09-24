#!/usr/bin/env python3
"""Audit causal direction replays and bootstrap paired day-level lifts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCES = {
    5: HERE / "output/direction_benchmark_short",
    10: HERE / "output/direction_benchmark_short",
    20: HERE / "output/direction_benchmark",
}
COMPARISONS = (
    (5, "logit_l2_l3_6", "logit_l2"),
    (10, "logit_l2_l3_6", "logit_l2"),
    (20, "logit_l2_l3_6", "logit_l2"),
    (20, "hgb_l2_l3_6", "hgb_l2"),
    (5, "logit_l2_l3_6", "microprice_sign"),
    (20, "hgb_l2_l3_6", "microprice_sign"),
)


def correctness(path: Path, a: str, b: str) -> tuple[int, int, int]:
    with np.load(path, allow_pickle=False) as archive:
        move = archive["move_ticks"]
        positive = move > 0
        selected = move != 0
        scores = []
        for name in (a, b):
            score = archive[f"score_{name}"]
            threshold = 0.5 if name.startswith(("logit", "hgb", "prior")) else 0.0
            scores.append(score >= threshold)
    return (int(np.count_nonzero(selected & (scores[0] == positive))),
            int(np.count_nonzero(selected & (scores[1] == positive))),
            int(np.count_nonzero(selected)))


def main() -> None:
    rng = np.random.default_rng(20260924)
    results = {}
    for horizon, path in SOURCES.items():
        result = json.loads((path / "results.json").read_text())
        if len(result["test_dates"]) != 19:
            raise AssertionError(f"incomplete horizon {horizon}")
        results[horizon] = result
        for day in result["daily"]:
            if day["horizon_snapshots"] != horizon:
                continue
            date = day["date"]
            if any(prior >= date for prior in day["prior_dates"]):
                raise AssertionError(f"nonprior train date on {date}")
            if not day["l3_available"]:
                with np.load(path / "per_day" / f"{date}_h{horizon}.npz",
                             allow_pickle=False) as archive:
                    for name, base in (("ridge_l2_l3_6", "ridge_l2"),
                                       ("logit_l2_l3_6", "logit_l2")):
                        if not np.array_equal(archive[f"score_{name}"],
                                              archive[f"score_{base}"]):
                            raise AssertionError(f"L3 fallback differs on {date}")
                    if horizon == 20:
                        for name, base in (("svm_l2_l3_6", "svm_l2"),
                                           ("hgb_l2_l3_6", "hgb_l2"),
                                           ("logit_l2_l3_3", "logit_l2")):
                            if not np.array_equal(archive[f"score_{name}"],
                                                  archive[f"score_{base}"]):
                                raise AssertionError(f"L3 fallback differs on {date}")
    comparisons = []
    for horizon, a, b in COMPARISONS:
        source = SOURCES[horizon]
        daily = [d for d in results[horizon]["daily"]
                 if d["horizon_snapshots"] == horizon]
        for subset in ("all", "l3_available"):
            selected = [d for d in daily if subset == "all" or d["l3_available"]]
            blocks = np.asarray([
                correctness(source / "per_day" / f"{d['date']}_h{horizon}.npz", a, b)
                for d in selected
            ], dtype=np.int64)
            lift = float((blocks[:, 0].sum() - blocks[:, 1].sum()) /
                         blocks[:, 2].sum())
            draws = rng.integers(0, len(blocks), size=(20_000, len(blocks)))
            sampled = blocks[draws]
            lift_draws = ((sampled[:, :, 0].sum(axis=1) -
                           sampled[:, :, 1].sum(axis=1)) /
                          sampled[:, :, 2].sum(axis=1))
            comparisons.append({
                "horizon_snapshots": horizon, "subset": subset,
                "method": a, "baseline": b,
                "days": len(blocks), "directional_quotes": int(blocks[:, 2].sum()),
                "accuracy_method": float(blocks[:, 0].sum() / blocks[:, 2].sum()),
                "accuracy_baseline": float(blocks[:, 1].sum() / blocks[:, 2].sum()),
                "accuracy_lift_percentage_points": 100 * lift,
                "day_bootstrap_ci95_percentage_points": (
                    100 * np.quantile(lift_draws, [0.025, 0.975])
                ).tolist(),
                "positive_days": int(np.count_nonzero(blocks[:, 0] > blocks[:, 1])),
                "negative_days": int(np.count_nonzero(blocks[:, 0] < blocks[:, 1])),
                "zero_difference_days": int(np.count_nonzero(blocks[:, 0] == blocks[:, 1])),
            })
    output = {
        "checks": {
            "all_train_dates_prior": True,
            "gap_day_l3_fallback_exact": True,
            "test_dates": 19,
            "bootstrap_unit": "whole trading day",
            "bootstrap_draws": 20_000,
        },
        "comparisons": comparisons,
        "warning": "July dates are already inspected; CIs describe these sampled days, not untouched-date generalization.",
    }
    out = HERE / "output/direction_benchmark/validation.json"
    out.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
