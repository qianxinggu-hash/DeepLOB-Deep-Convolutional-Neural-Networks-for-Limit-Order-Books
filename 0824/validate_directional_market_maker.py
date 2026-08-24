#!/usr/bin/env python3
"""Independent accounting and causality checks for the market-maker output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from directional_market_maker import (
    ALL_IN_FEE_RATE,
    LATENCY_MS,
    LOT_SIZE,
    OFFICIAL_FIXED_FEE_RATE,
    compact_time_to_day_ms,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    result_path = HERE / "output" / "directional_market_maker_results.json"
    fill_path = HERE / "output" / "directional_market_maker_fills.csv"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    fills = pd.read_csv(fill_path)
    checks: dict[str, bool] = {}

    expected_fee = fills["price_hkd"].to_numpy() * LOT_SIZE * ALL_IN_FEE_RATE
    checks["fill_fees_recomputed"] = bool(
        np.allclose(expected_fee, fills["fee_hkd"].to_numpy(), atol=1e-9)
    )
    signal_ms = compact_time_to_day_ms(fills["signal_time"].to_numpy(np.int64))
    checks["fills_strictly_after_quote_latency"] = bool(
        np.all(fills["trade_time_ms"].to_numpy() > signal_ms + LATENCY_MS)
    )
    key = ["strategy", "fill_mode", "date", "quote_number", "side"]
    checks["at_most_one_fill_per_quote_side"] = not bool(fills.duplicated(key).any())

    summary_counts_match = True
    accounting_matches = True
    inventory_limits_hold = True
    passive_modes_present = set(fills["fill_mode"]) == {"through", "touch"}
    for experiment in results["experiments"]:
        date = experiment["test_date"]
        for strategy, by_mode in experiment["test"]["strategies"].items():
            for fill_mode, summary in by_mode.items():
                selected = fills[
                    (fills["date"] == date)
                    & (fills["strategy"] == strategy)
                    & (fills["fill_mode"] == fill_mode)
                ]
                summary_counts_match &= len(selected) == int(summary["maker_fills"])
                summary_counts_match &= int((selected["side"] == "buy").sum()) == int(
                    summary["buy_fills"]
                )
                summary_counts_match &= int((selected["side"] == "sell").sum()) == int(
                    summary["sell_fills"]
                )
                maximum = int(summary["parameters"]["max_inventory_lots"])
                inventory_limits_hold &= bool(
                    (selected["inventory_after_lots"].abs() <= maximum).all()
                )
                turnover = float(summary["feeable_turnover_hkd"])
                gross = float(summary["gross_pnl_hkd"])
                accounting_matches &= np.isclose(
                    float(summary["fees_hkd"]), turnover * ALL_IN_FEE_RATE, atol=1e-7
                )
                accounting_matches &= np.isclose(
                    float(summary["official_only_fees_hkd"]),
                    turnover * OFFICIAL_FIXED_FEE_RATE,
                    atol=1e-7,
                )
                accounting_matches &= np.isclose(
                    float(summary["net_pnl_hkd"]),
                    gross - turnover * ALL_IN_FEE_RATE,
                    atol=1e-7,
                )
                accounting_matches &= np.isclose(
                    float(summary["net_pnl_official_only_hkd"]),
                    gross - turnover * OFFICIAL_FIXED_FEE_RATE,
                    atol=1e-7,
                )
    checks["summary_fill_counts_match_log"] = bool(summary_counts_match)
    checks["inventory_limits_hold"] = bool(inventory_limits_hold)
    checks["summary_accounting_recomputed"] = bool(accounting_matches)
    checks["both_queue_sensitivity_modes_present"] = bool(passive_modes_present)

    aggregate_matches = True
    for fill_mode, by_strategy in results["aggregate_test_results"].items():
        for strategy, aggregate in by_strategy.items():
            net_values = [
                float(item["test"]["strategies"][strategy][fill_mode]["net_pnl_hkd"])
                for item in results["experiments"]
            ]
            gross_values = [
                float(item["test"]["strategies"][strategy][fill_mode]["gross_pnl_hkd"])
                for item in results["experiments"]
            ]
            official_values = [
                float(
                    item["test"]["strategies"][strategy][fill_mode][
                        "net_pnl_official_only_hkd"
                    ]
                )
                for item in results["experiments"]
            ]
            aggregate_matches &= np.isclose(
                sum(net_values), float(aggregate["total_net_pnl_hkd"]), atol=1e-7
            )
            aggregate_matches &= np.isclose(
                sum(gross_values), float(aggregate["total_gross_pnl_hkd"]), atol=1e-7
            )
            aggregate_matches &= np.isclose(
                sum(official_values),
                float(aggregate["total_net_pnl_official_only_hkd"]),
                atol=1e-7,
            )
    checks["aggregate_results_recomputed"] = bool(aggregate_matches)

    chronological = all(
        max(item["model_train_dates"]) < item["test_date"]
        for item in results["experiments"]
    )
    checks["model_training_precedes_test_day"] = chronological
    checks["all_source_books_untainted"] = all(
        not record["mbo_state_tainted"] and not record["order_errors"]
        for record in results["data_quality"].values()
    )
    output = {
        "as_of": results["as_of"],
        "fill_rows_checked": int(len(fills)),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "assessment": "Share with caveats" if all(checks.values()) else "Needs revision",
        "required_caveat": "The source does not reveal the hypothetical order's exact queue position; touch/through are sensitivity bounds, not exact fills.",
    }
    output_path = HERE / "output" / "directional_market_maker_validation.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
