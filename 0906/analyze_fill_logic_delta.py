#!/usr/bin/env python3
"""Isolate the PnL impact of adding best-quote OR fill evidence.

The paired replay holds each day's original classic-AS parameters fixed and
changes only the fill rule from trade-price evidence to trade-price OR opposite
best-quote evidence.  This separates execution-rule effects from walk-forward
parameter retuning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LEGACY_DIR = ROOT / "0824"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(LEGACY_DIR) not in sys.path:
    sys.path.append(str(LEGACY_DIR))

import directional_market_maker as new_model  # noqa: E402
from run_multiday_strategy import HORIZON, load_day  # noqa: E402


OLD_RESULT = LEGACY_DIR / "output/july_2026_tolerant_as_gp_drift_results.json"
NEW_RESULT = HERE / "output/july_2026_tolerant_as_gp_drift_results.json"
OUTPUT = HERE / "output/fill_logic_diagnostic.json"


def build_diagnostic() -> dict[str, object]:
    old = json.loads(OLD_RESULT.read_text(encoding="utf-8"))
    new = json.loads(NEW_RESULT.read_text(encoding="utf-8"))
    legacy_model = new_model._legacy
    daily: list[dict[str, object]] = []

    for test in old["tests"]:
        date = test["test_date"]
        day = load_day(
            date,
            "tolerant_mbo",
            LEGACY_DIR
            / f"data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz",
        )
        trades = legacy_model.load_or_extract_trades(
            Path("/unused"),
            LEGACY_DIR / f"data/july_2026/hk07709_{date}_trades.npz",
        )
        quote_indices = new_model.select_quote_indices(day, day.eligible, HORIZON)
        alpha = np.zeros(len(quote_indices), dtype=np.float64)
        parameters = new_model.ASParameters(**test["parameters"]["classic_as"])

        for fill_mode in ("through", "touch"):
            old_row, _ = legacy_model.simulate_market_maker(
                day,
                trades,
                quote_indices,
                alpha,
                parameters,
                fill_mode,
                "classic_as",
            )
            new_row, _ = new_model.simulate_market_maker(
                day,
                trades,
                quote_indices,
                alpha,
                parameters,
                fill_mode,
                "classic_as",
            )
            reported = test["strategies"]["classic_as"][fill_mode]
            if not np.isclose(old_row["net_pnl_hkd"], reported["net_pnl_hkd"]):
                raise AssertionError(f"old result does not reproduce: {date} {fill_mode}")
            daily.append(
                {
                    "date": date,
                    "fill_mode": fill_mode,
                    "old_fills": int(old_row["maker_fills"]),
                    "new_fills": int(new_row["maker_fills"]),
                    "new_trade_price_fills": int(new_row["trade_price_fills"]),
                    "new_best_quote_fills": int(new_row["best_quote_fills"]),
                    "new_best_ask_fills": int(new_row["best_ask_fills"]),
                    "new_best_bid_fills": int(new_row["best_bid_fills"]),
                    "old_gross_pnl_hkd": float(old_row["gross_pnl_hkd"]),
                    "new_gross_pnl_hkd": float(new_row["gross_pnl_hkd"]),
                    "old_fees_hkd": float(old_row["fees_hkd"]),
                    "new_fees_hkd": float(new_row["fees_hkd"]),
                    "old_net_pnl_hkd": float(old_row["net_pnl_hkd"]),
                    "new_net_pnl_hkd": float(new_row["net_pnl_hkd"]),
                }
            )

    summary: list[dict[str, object]] = []
    for fill_mode in ("through", "touch"):
        selected = [row for row in daily if row["fill_mode"] == fill_mode]
        count_keys = (
            "old_fills",
            "new_fills",
            "new_trade_price_fills",
            "new_best_quote_fills",
            "new_best_ask_fills",
            "new_best_bid_fills",
        )
        value_keys = (
            "old_gross_pnl_hkd",
            "new_gross_pnl_hkd",
            "old_fees_hkd",
            "new_fees_hkd",
            "old_net_pnl_hkd",
            "new_net_pnl_hkd",
        )
        totals = {key: sum(int(row[key]) for row in selected) for key in count_keys}
        totals.update(
            {key: sum(float(row[key]) for row in selected) for key in value_keys}
        )
        added_fills = totals["new_fills"] - totals["old_fills"]
        gross_delta = totals["new_gross_pnl_hkd"] - totals["old_gross_pnl_hkd"]
        fee_delta = totals["new_fees_hkd"] - totals["old_fees_hkd"]
        net_delta = totals["new_net_pnl_hkd"] - totals["old_net_pnl_hkd"]
        full_old = old["aggregate"][fill_mode]["classic_as"]
        full_new = new["aggregate"][fill_mode]["classic_as"]
        summary.append(
            {
                "fill_mode": fill_mode,
                **totals,
                "added_fills": int(added_fills),
                "new_trade_price_fill_share": (
                    totals["new_trade_price_fills"] / totals["new_fills"]
                ),
                "new_best_quote_fill_share": (
                    totals["new_best_quote_fills"] / totals["new_fills"]
                ),
                "gross_pnl_delta_hkd": gross_delta,
                "additional_fees_hkd": fee_delta,
                "net_pnl_delta_hkd": net_delta,
                "gross_delta_per_added_fill_hkd": gross_delta / added_fills,
                "fee_delta_per_added_fill_hkd": fee_delta / added_fills,
                "net_delta_per_added_fill_hkd": net_delta / added_fills,
                "days_new_rule_better": sum(
                    float(row["new_net_pnl_hkd"]) > float(row["old_net_pnl_hkd"])
                    for row in selected
                ),
                "days_new_rule_worse": sum(
                    float(row["new_net_pnl_hkd"]) < float(row["old_net_pnl_hkd"])
                    for row in selected
                ),
                "full_walk_forward_old_net_pnl_hkd": float(full_old["net_pnl_hkd"]),
                "full_walk_forward_new_net_pnl_hkd": float(full_new["net_pnl_hkd"]),
                "full_walk_forward_net_delta_hkd": float(full_new["net_pnl_hkd"])
                - float(full_old["net_pnl_hkd"]),
            }
        )

    return {
        "as_of": "2026-09-06",
        "instrument": "HK 07709",
        "test_days": len(old["tests"]),
        "comparison": "classic AS; original daily parameters held fixed",
        "old_fill_rule": "qualifying future trade price",
        "new_fill_rule": "qualifying future trade price OR opposite L1 best quote",
        "summary": summary,
        "daily": daily,
        "validation": {
            "old_results_reproduced": True,
            "net_delta_equals_gross_delta_minus_additional_fees": all(
                np.isclose(
                    row["net_pnl_delta_hkd"],
                    row["gross_pnl_delta_hkd"] - row["additional_fees_hkd"],
                )
                for row in summary
            ),
            "new_fill_evidence_counts_sum_to_new_fills": all(
                row["new_fills"]
                == row["new_trade_price_fills"] + row["new_best_quote_fills"]
                and row["new_best_quote_fills"]
                == row["new_best_ask_fills"] + row["new_best_bid_fills"]
                for row in daily
            ),
        },
    }


def main() -> None:
    result = build_diagnostic()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("WROTE", OUTPUT)


if __name__ == "__main__":
    main()
