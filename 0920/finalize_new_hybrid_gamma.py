#!/usr/bin/env python3
"""Combine complete per-day GP gamma sweep files after parallel replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sweep_new_hybrid_gamma as sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=sweep.DEFAULT_OUTPUT)
    parser.add_argument("--signal-dir", type=Path, default=sweep.month.DEFAULT_SIGNAL)
    parser.add_argument("--gammas", type=float, nargs="+", default=sweep.GAMMAS)
    args = parser.parse_args()
    gammas = sorted(set(args.gammas), reverse=True)
    manifest = json.loads(sweep.month.MANIFEST.read_text())
    dates = manifest["date_order"]
    rows = []
    for date in dates[1:]:
        for gamma in gammas:
            path = args.output_dir / "per_day" / f"{date}_gamma_{gamma:g}.json"
            saved = json.loads(path.read_text())
            if saved["date"] != date or saved["gamma"] != gamma or len(saved["rows"]) != 4:
                raise AssertionError(f"invalid per-day file: {path}")
            rows.extend(saved["rows"])
    summary = [sweep.summarize(rows, gamma, mode, treatment, subset)
               for gamma in gammas for mode in sweep.gp.FILL_MODES
               for treatment in sweep.TREATMENTS
               for subset in ("all", "l3_available", "l3_unavailable")]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "daily.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    anchors = {
        "carry": sweep.HERE / "output/new_hybrid_month/results.json",
        "forced_flat": sweep.HERE / "output/new_hybrid_month_forced_flat/results.json",
    }
    prior = json.loads(anchors["carry"].read_text())
    result = {
        "name": "new hybrid GP gamma sweep", "dates": dates, "test_dates": dates[1:],
        "gammas": gammas, "gap_policy": prior["gap_policy"],
        "signal_result": str((args.signal_dir / "results.json").resolve()),
        "anchor_results": {key: str(path.resolve()) for key, path in anchors.items()},
        "source_note": "5 and 3.125 from reconciled month replay; all other gamma values replayed here",
        "fee_and_execution": prior["fee_and_execution"],
        "test_labels_not_used_in_current_day_fit": True,
        "summary": summary, "daily": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps([s for s in summary if s["subset"] == "all"
                      and s["gap_inventory"] == "carry"],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
