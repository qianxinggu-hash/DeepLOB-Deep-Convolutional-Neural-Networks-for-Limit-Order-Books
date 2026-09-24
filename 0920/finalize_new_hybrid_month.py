#!/usr/bin/env python3
"""Insert daily-tick-correct AS results into the completed GP month replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def merge(treatment: str) -> None:
    month_dir = HERE / ("output/new_hybrid_month" if treatment == "carry" else
                        "output/new_hybrid_month_forced_flat")
    as_dir = HERE / f"output/new_hybrid_as_corrected_{treatment}"
    month = json.loads((month_dir / "results.json").read_text())
    corrected = json.loads((as_dir / "results.json").read_text())
    if month.get("gap_inventory", "carry") != treatment or corrected["gap_inventory"] != treatment:
        raise ValueError("gap inventory treatment differs")
    if not corrected["as_only"]:
        raise ValueError("expected AS-only corrected result")
    as_rows = {(r["date"], r["fill_mode"]): r for r in corrected["daily"]}
    old_keys = {(r["date"], r["fill_mode"]) for r in month["daily"] if r["family"] == "AS"}
    if old_keys != set(as_rows) or len(old_keys) != 38:
        raise ValueError("AS daily keys do not match 19 days × 2 modes")
    month["daily"] = [as_rows[(r["date"], r["fill_mode"])]
                      if r["family"] == "AS" else r for r in month["daily"]]
    as_summary = {(r["fill_mode"], r["subset"]): r for r in corrected["summary"]}
    summary_keys = {(r["fill_mode"], r["subset"])
                    for r in month["summary"] if r["family"] == "AS"}
    if summary_keys != set(as_summary):
        raise ValueError("AS summary keys do not match")
    month["summary"] = [as_summary[(r["fill_mode"], r["subset"])]
                        if r["family"] == "AS" else r for r in month["summary"]]
    month["as_tick_corrected"] = True
    month["as_tick_hkd"] = "per-day inferred exchange tick (HK$0.05 or HK$0.02)"
    month["as_corrected_source"] = str((as_dir / "results.json").resolve())
    month["fee_and_execution"] += "; AS quotes and through fills use the daily tick"
    (month_dir / "results.json").write_text(
        json.dumps(month, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (month_dir / "daily.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(month["daily"][0]))
        writer.writeheader()
        writer.writerows(month["daily"])
    print("MERGED", treatment, month_dir / "results.json")


def main() -> None:
    for treatment in ("carry", "forced_flat"):
        merge(treatment)


if __name__ == "__main__":
    main()
