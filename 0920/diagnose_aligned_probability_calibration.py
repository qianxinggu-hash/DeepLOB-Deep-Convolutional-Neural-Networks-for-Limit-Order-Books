#!/usr/bin/env python3
"""Evaluate past-only probability-to-tick maps before GP/AS replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "output/aligned_direction_comparison"
DATA = HERE / "output/direction_dataset"
MODELS = ("deeplob", "old_hybrid", "new_l3_hybrid")


def main() -> None:
    results = json.loads((ROOT / "results.json").read_text())
    rows = []
    for row in results["daily"]:
        if row["horizon"] != 20:
            continue
        date = row["date"]
        with np.load(DATA / f"day_{date}.npz", allow_pickle=False) as a:
            y = a["move_20_ticks"].copy()
        with np.load(ROOT / "signals" / f"day_{date}.npz", allow_pickle=False) as a:
            valid = a["has_100_snapshot_history"]
            assert int(valid.sum()) == len(y)
            preds = {name: a[f"expected_{name}_ticks"][valid].copy()
                     for name in MODELS}
        rows.append({"date": date, "l3_available": row["l3_available"],
                     "y": y, "predictions": preds})
    summary = []
    for subset in ("all", "l3_available", "l3_unavailable"):
        chosen = [r for r in rows if subset == "all" or r["l3_available"] ==
                  (subset == "l3_available")]
        y = np.concatenate([r["y"] for r in chosen])
        for name in MODELS:
            p = np.concatenate([r["predictions"][name] for r in chosen])
            summary.append({
                "subset": subset, "model": name, "days": len(chosen),
                "quotes": len(y), "rmse_ticks": float(np.sqrt(np.mean((y-p)**2))),
                "zero_rmse_ticks": float(np.sqrt(np.mean(y**2))),
                "correlation": float(np.corrcoef(y, p)[0, 1]),
                "prediction_std_ticks": float(np.std(p)),
                "prediction_mean_ticks": float(np.mean(p)),
            })
    out = {"description": "past-only probability-to-tick calibration; same 20-snapshot endpoint target",
           "summary": summary}
    (ROOT / "calibration_diagnostic.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps([r for r in summary if r["subset"] == "all"], indent=2))


if __name__ == "__main__":
    main()
