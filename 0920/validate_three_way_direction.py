#!/usr/bin/env python3
"""Day-block uncertainty for the three-class L3 versus L2 comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "output/direction_three_way"


def main() -> None:
    result = json.loads((SOURCE / "results.json").read_text())
    rows = []
    for horizon in (5, 20):
        for subset in ("all", "l3_available"):
            blocks = []
            for day in result["daily"]:
                if day["horizon_snapshots"] != horizon:
                    continue
                if subset == "l3_available" and not day["l3_available"]:
                    continue
                with np.load(SOURCE / "per_day" /
                             f"{day['date']}_h{horizon}.npz", allow_pickle=False) as archive:
                    actual = archive["actual"]
                    fused = archive["prediction_unweighted_l2_l3_6"]
                    book = archive["prediction_unweighted_l2"]
                    if not day["l3_available"] and not np.array_equal(fused, book):
                        raise AssertionError(f"L3 fallback differs on {day['date']}")
                    blocks.append((int(np.count_nonzero(fused == actual)),
                                   int(np.count_nonzero(book == actual)), len(actual)))
            blocks = np.asarray(blocks, dtype=np.int64)
            rng = np.random.default_rng(20260924 + horizon)
            draws = rng.integers(0, len(blocks), size=(20_000, len(blocks)))
            sampled = blocks[draws]
            lift = ((sampled[:, :, 0].sum(axis=1) - sampled[:, :, 1].sum(axis=1)) /
                    sampled[:, :, 2].sum(axis=1)) * 100
            rows.append({
                "horizon_snapshots": horizon, "subset": subset,
                "days": len(blocks), "quotes": int(blocks[:, 2].sum()),
                "accuracy_lift_percentage_points": float(
                    (blocks[:, 0].sum() - blocks[:, 1].sum()) / blocks[:, 2].sum() * 100
                ),
                "day_bootstrap_ci95_percentage_points": np.quantile(
                    lift, [0.025, 0.975]
                ).tolist(),
                "days_better": int(np.count_nonzero(blocks[:, 0] > blocks[:, 1])),
                "days_worse": int(np.count_nonzero(blocks[:, 0] < blocks[:, 1])),
            })
    output = {"bootstrap_unit": "whole trading day", "draws": 20_000,
              "gap_day_l3_fallback_exact": True, "comparisons": rows,
              "warning": "July dates are already inspected; intervals do not correct for model selection."}
    (SOURCE / "validation.json").write_text(
        json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
