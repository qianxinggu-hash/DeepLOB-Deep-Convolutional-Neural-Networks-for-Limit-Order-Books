#!/usr/bin/env python3
"""Paired trading-day uncertainty for extra direction model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCES = {
    "short_full": HERE / "output/direction_benchmark_short_full",
    "full": HERE / "output/direction_benchmark",
    "ablation": HERE / "output/direction_l3_ablation",
    "variants": HERE / "output/direction_label_variants",
}
COMPARISONS = (
    (5, "short_full", "logit_l3_6_only", "short_full", "logit_l2"),
    (5, "short_full", "logit_l3_6_only", "short_full", "logit_l2_l3_6"),
    (20, "full", "logit_l3_6_only", "full", "logit_l2"),
    (5, "short_full", "hgb_l2_l3_6", "short_full", "hgb_l2"),
    (10, "short_full", "hgb_l2_l3_6", "short_full", "hgb_l2"),
    (5, "short_full", "logit_l2_l3_6", "variants", "abs1_training_l2_l3_6"),
    (5, "short_full", "logit_l2_l3_6", "variants", "magnitude_weighted_l2_l3_6"),
    (20, "full", "logit_l2_l3_6", "variants", "abs1_training_l2_l3_6"),
    (5, "ablation", "all_six", "ablation", "best_order_count_imbalance"),
)


def correctness(path: Path, method: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        move = archive["move_ticks"].copy()
        score = archive[f"score_{method}"].copy()
    directional = move != 0
    predicted = score >= 0.5
    return move, ((predicted == (move > 0)) & directional)


def main() -> None:
    dates = [r["date"] for r in json.loads(
        (HERE / "output/direction_dataset/manifest.json").read_text()
    )["dates"][1:]]
    rows = []
    for horizon, source_a, a, source_b, b in COMPARISONS:
        blocks = []
        per_day = []
        for date in dates:
            path_a = SOURCES[source_a] / "per_day" / f"{date}_h{horizon}.npz"
            path_b = SOURCES[source_b] / "per_day" / f"{date}_h{horizon}.npz"
            if not path_a.exists() or not path_b.exists():
                continue
            with np.load(path_a, allow_pickle=False) as archive:
                if f"score_{a}" not in archive:
                    continue
            y_a, correct_a = correctness(path_a, a)
            y_b, correct_b = correctness(path_b, b)
            if not np.array_equal(y_a, y_b):
                raise AssertionError(f"labels differ on {date}")
            n = int(np.count_nonzero(y_a))
            block = (int(correct_a.sum()), int(correct_b.sum()), n)
            blocks.append(block)
            per_day.append({"date": date, "method_correct": block[0],
                            "baseline_correct": block[1], "directional_quotes": n})
        arr = np.asarray(blocks, dtype=np.int64)
        rng = np.random.default_rng(20260924 + horizon + len(a))
        draws = rng.integers(0, len(arr), size=(20_000, len(arr)))
        sampled = arr[draws]
        delta = ((sampled[:, :, 0].sum(axis=1) - sampled[:, :, 1].sum(axis=1)) /
                 sampled[:, :, 2].sum(axis=1)) * 100
        rows.append({
            "horizon_snapshots": horizon,
            "method": f"{source_a}:{a}",
            "baseline": f"{source_b}:{b}",
            "days": len(arr), "directional_quotes": int(arr[:, 2].sum()),
            "method_accuracy": float(arr[:, 0].sum() / arr[:, 2].sum()),
            "baseline_accuracy": float(arr[:, 1].sum() / arr[:, 2].sum()),
            "lift_percentage_points": float(
                (arr[:, 0].sum() - arr[:, 1].sum()) / arr[:, 2].sum() * 100
            ),
            "day_bootstrap_ci95_percentage_points": np.quantile(delta, [0.025, 0.975]).tolist(),
            "days_better": int(np.count_nonzero(arr[:, 0] > arr[:, 1])),
            "days_worse": int(np.count_nonzero(arr[:, 0] < arr[:, 1])),
            "daily": per_day,
        })
    output = {"bootstrap_unit": "whole trading day", "draws": 20_000,
              "comparisons": rows,
              "warning": "Model/horizon choices inspected on the same July dates; intervals do not correct for multiple comparisons."}
    path = HERE / "output/direction_benchmark/extra_comparisons.json"
    path.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "daily"} for row in rows],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
