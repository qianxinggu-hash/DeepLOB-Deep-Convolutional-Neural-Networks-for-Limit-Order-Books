#!/usr/bin/env python3
"""Re-run the July gamma=5 no-drift GP policy with issue-free prior days.

The test-day books and fill proxies are identical to the original 0920 study.
Only the rolling GP calibration set changes: use the last five available dates
with no *recorded* MBO reconstruction order-state error or feed gap.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.append(str(ROOT / "0824"))

import gp_qvi_model as gp  # noqa: E402
from paper_gp_replay import simulate_gp_day  # noqa: E402
from run_gp_qvi_experiment import _load_inputs  # noqa: E402


QUALITY = HERE / "output/combination_study/source_quality.json"
REFERENCE = HERE / "output/gamma_daily.csv"
OUTPUT = HERE / "output/combination_study/clean_prior_gamma5.json"


def run_day(task: tuple) -> dict:
    date, day, trades, inputs = task
    config = gp.GPConfig(inventory_penalty_gamma=5.0)
    tick = gp.infer_tick_hkd(day)
    outcomes = {}
    for mode in gp.FILL_MODES:
        policies = {
            bucket: gp.solve_gp_policy(
                inputs, tick, mode, bucket, config, market_orders_enabled=True
            )
            for bucket in range(len(gp.SESSION_BUCKETS))
        }
        result = simulate_gp_day(day, trades, policies, config, mode, "gp_qvi")
        outcomes[mode] = {
            key: result[key] for key in (
                "net_pnl_hkd", "gross_pnl_hkd", "all_in_fees_hkd",
                "maker_fills", "market_order_lots", "quoted_bid", "quoted_ask",
                "max_abs_inventory_lots", "max_drawdown_hkd",
                "squared_inventory_lot2_seconds", "quote_decisions",
                "end_inventory_lots",
            )
        }
    return {"date": date, "calibration_dates": list(inputs.calibration_dates),
            "outcomes": outcomes}


def summarize(rows: list[dict], mode: str) -> dict:
    return {
        "test_days": len(rows),
        "original_net_pnl_hkd": sum(row[mode]["original_net_pnl_hkd"] for row in rows),
        "clean_prior_net_pnl_hkd": sum(row[mode]["clean_prior_net_pnl_hkd"] for row in rows),
        "net_pnl_delta_hkd": sum(row[mode]["net_pnl_delta_hkd"] for row in rows),
        "original_maker_fills": sum(row[mode]["original_maker_fills"] for row in rows),
        "clean_prior_maker_fills": sum(row[mode]["clean_prior_maker_fills"] for row in rows),
        "changed_pnl_days": sum(abs(row[mode]["net_pnl_delta_hkd"]) > 1e-6 for row in rows),
        "clean_prior_profitable_days": sum(row[mode]["clean_prior_net_pnl_hkd"] > 0 for row in rows),
        "clean_prior_worst_intraday_drawdown_hkd": max(
            row[mode]["clean_prior_max_drawdown_hkd"] for row in rows
        ),
    }


def main() -> None:
    source, dates, days, trades = _load_inputs()
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    quality_by_date = {row["date"]: row for row in quality["days"]}
    issue_dates = {
        row["date"] for row in quality["days"]
        if row["quality_group"] == "recorded_reconstruction_issue"
    }
    assert dates[1:] == list(quality_by_date)
    assert dates[0] not in issue_dates

    config = gp.GPConfig(inventory_penalty_gamma=5.0)
    stats = {
        date: gp.collect_daily_calibration_stats(days[date], trades[date], config)
        for date in dates if date not in issue_dates
    }
    tasks = []
    for position, date in enumerate(dates[1:], start=1):
        prior = [prior for prior in dates[:position] if prior not in issue_dates][-5:]
        assert prior and all(prior_date < date for prior_date in prior)
        inputs = gp.aggregate_calibration_stats([stats[prior_date] for prior_date in prior], config)
        tasks.append((date, days[date], trades[date], inputs))

    with REFERENCE.open(encoding="utf-8-sig", newline="") as stream:
        reference = {
            (row["date"], row["fill_mode"]): row
            for row in csv.DictReader(stream) if float(row["gamma"]) == 5.0
        }
    output_rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for day_result in pool.map(run_day, tasks):
            date = day_result["date"]
            calibration_dates = day_result["calibration_dates"]
            assert all(prior not in issue_dates for prior in calibration_dates)
            row = {
                "date": date,
                "test_day_quality_group": quality_by_date[date]["quality_group"],
                "phase": quality_by_date[date]["phase"],
                "clean_prior_calibration_dates": calibration_dates,
                "clean_prior_calibration_day_count": len(calibration_dates),
            }
            for mode in gp.FILL_MODES:
                result = day_result["outcomes"][mode]
                old = reference[(date, mode)]
                old_net = float(old["net_pnl_hkd"])
                net = float(result["net_pnl_hkd"])
                assert math.isclose(
                    net, float(result["gross_pnl_hkd"]) - float(result["all_in_fees_hkd"]),
                    abs_tol=1e-6,
                )
                assert result["end_inventory_lots"] == 0
                assert result["quote_decisions"] == int(old["quote_decisions"])
                if date == dates[1]:
                    assert math.isclose(net, old_net, abs_tol=1e-6)
                    assert result["maker_fills"] == int(old["maker_fills"])
                row[mode] = {
                    "original_calibration_dates": old["calibration_dates"].split("|"),
                    "original_net_pnl_hkd": old_net,
                    "clean_prior_net_pnl_hkd": net,
                    "net_pnl_delta_hkd": net - old_net,
                    "original_maker_fills": int(old["maker_fills"]),
                    "clean_prior_maker_fills": int(result["maker_fills"]),
                    "original_gross_pnl_hkd": float(old["gross_pnl_hkd"]),
                    "clean_prior_gross_pnl_hkd": float(result["gross_pnl_hkd"]),
                    "original_all_in_fees_hkd": float(old["all_in_fees_hkd"]),
                    "clean_prior_all_in_fees_hkd": float(result["all_in_fees_hkd"]),
                    "clean_prior_market_order_lots": int(result["market_order_lots"]),
                    "clean_prior_max_abs_inventory_lots": int(result["max_abs_inventory_lots"]),
                    "clean_prior_max_drawdown_hkd": float(result["max_drawdown_hkd"]),
                    "clean_prior_squared_inventory_lot2_seconds": float(
                        result["squared_inventory_lot2_seconds"]
                    ),
                }
            output_rows.append(row)
            print("DONE", date, len(calibration_dates),
                  *(f"{mode}={row[mode]['clean_prior_net_pnl_hkd']:.2f}" for mode in gp.FILL_MODES),
                  flush=True)

    groups = {
        "all": output_rows,
        "test_day_no_recorded_issue": [row for row in output_rows if row["test_day_quality_group"] == "no_recorded_reconstruction_issue"],
        "test_day_recorded_issue": [row for row in output_rows if row["test_day_quality_group"] == "recorded_reconstruction_issue"],
        "exploration_2026_07_03_to_17": [row for row in output_rows if row["phase"] == "exploration_2026_07_03_to_17"],
        "late_check_2026_07_20_to_30": [row for row in output_rows if row["phase"] == "late_check_2026_07_20_to_30"],
    }
    summary = {group: {mode: summarize(rows, mode) for mode in gp.FILL_MODES}
               for group, rows in groups.items()}
    result = {
        "experiment": "gamma5_no_drift_clean_prior_calibration_sensitivity",
        "instrument": "HK 07709",
        "source_quality": str(QUALITY.relative_to(ROOT)),
        "original_baseline": str(REFERENCE.relative_to(ROOT)),
        "calibration_rule": "last up to five previous replay-eligible July dates with no recorded missing-order error, feed gap, or final taint",
        "gamma": 5.0,
        "fill_modes": list(gp.FILL_MODES),
        "limitations": [
            "No recorded reconstruction issue does not establish perfectly correct market data.",
            "Changing the calibration dates also changes sample recency and size; PnL differences cannot be attributed solely to feed quality.",
            "Test-day books and proxy fills are unchanged, including nine issue-flagged dates.",
            "The through and touch modes are fill proxies, not observed queue-position fills.",
            "The existing GP replay infers the tick from the complete test-day book; this sensitivity retains that pre-existing potential look-ahead.",
            "The full July period has been used in prior gamma research; this is a sensitivity analysis, not a fresh holdout.",
        ],
        "summary": summary,
        "daily": output_rows,
        "validation": {
            "test_days": len(output_rows),
            "all_clean_prior_calibrations_strictly_past": all(
                all(prior < row["date"] and prior not in issue_dates
                    for prior in row["clean_prior_calibration_dates"])
                for row in output_rows
            ),
            "reference_first_test_day_reproduced": True,
            "all_days_end_flat_and_cash_reconciles": True,
            "quote_decisions_match_original": True,
        },
    }
    assert len(output_rows) == 19
    assert len(groups["test_day_recorded_issue"]) == 9
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(summary["all"], indent=2))


if __name__ == "__main__":
    main()
