#!/usr/bin/env python3
"""Run the July walk-forward experiment with the 0906 L1-book fill model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LEGACY_DIR = HERE.parent / "0824"
if str(LEGACY_DIR) not in sys.path:
    sys.path.append(str(LEGACY_DIR))

# Because the script directory (0906) precedes 0824 on sys.path, imports made
# by the legacy orchestration resolve directional_market_maker to the new model.
import run_july_month_strategy as experiment  # noqa: E402


def _selected_policy(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reconstruction-policy", choices=("strict", "tolerant"), default="strict"
    )
    arguments, _ = parser.parse_known_args(argv)
    return str(arguments.reconstruction_policy)


def _order_lifecycle_valid(output: dict[str, Any]) -> bool:
    return all(
        int(row["quoted_bid"])
        == int(row["buy_fills"]) + int(row["bid_cancels"])
        and int(row["quoted_ask"])
        == int(row["sell_fills"]) + int(row["ask_cancels"])
        for test in output["tests"]
        for strategy_rows in test["strategies"].values()
        for row in strategy_rows.values()
    )


def _fill_evidence_valid(output: dict[str, Any]) -> bool:
    return all(
        int(row["maker_fills"])
        == int(row["trade_price_fills"]) + int(row["best_quote_fills"])
        and int(row["best_quote_fills"])
        == int(row["best_ask_fills"]) + int(row["best_bid_fills"])
        for test in output["tests"]
        for strategy_rows in test["strategies"].values()
        for row in strategy_rows.values()
    )


def _add_lifecycle_aggregates(output: dict[str, Any]) -> None:
    for fill_mode in experiment.FILL_MODES:
        for strategy in experiment.STRATEGIES:
            rows = [
                test["strategies"][strategy][fill_mode] for test in output["tests"]
            ]
            aggregate = output["aggregate"][fill_mode][strategy]
            aggregate["quoted_bid"] = sum(int(row["quoted_bid"]) for row in rows)
            aggregate["quoted_ask"] = sum(int(row["quoted_ask"]) for row in rows)
            aggregate["bid_cancels"] = sum(int(row["bid_cancels"]) for row in rows)
            aggregate["ask_cancels"] = sum(int(row["ask_cancels"]) for row in rows)
            aggregate["expired_orders"] = sum(
                int(row["expired_orders"]) for row in rows
            )
            for key in (
                "trade_price_fills",
                "best_quote_fills",
                "best_ask_fills",
                "best_bid_fills",
            ):
                aggregate[key] = sum(int(row[key]) for row in rows)
            maker_fills = int(aggregate["maker_fills"])
            aggregate["trade_price_fill_share"] = float(
                aggregate["trade_price_fills"] / max(maker_fills, 1)
            )
            aggregate["best_quote_fill_share"] = float(
                aggregate["best_quote_fills"] / max(maker_fills, 1)
            )


def _write_daily_csv(output: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for test in output["tests"]:
        for fill_mode in experiment.FILL_MODES:
            row: dict[str, Any] = {
                "date": test["test_date"],
                "fill_mode": fill_mode,
                "direction_accuracy": test["classification"]["accuracy"],
                "direction_macro_f1": test["classification"]["macro_f1"],
                "drift_slope_ticks": test["calibrator"]["slope_ticks"],
            }
            for strategy in experiment.STRATEGIES:
                result = test["strategies"][strategy][fill_mode]
                row[f"{strategy}_net_pnl_hkd"] = result["net_pnl_hkd"]
                row[f"{strategy}_gross_pnl_hkd"] = result["gross_pnl_hkd"]
                row[f"{strategy}_maker_fills"] = result["maker_fills"]
                row[f"{strategy}_trade_price_fills"] = result[
                    "trade_price_fills"
                ]
                row[f"{strategy}_best_quote_fills"] = result["best_quote_fills"]
                row[f"{strategy}_best_ask_fills"] = result["best_ask_fills"]
                row[f"{strategy}_best_bid_fills"] = result["best_bid_fills"]
                row[f"{strategy}_trade_price_fill_share"] = result[
                    "trade_price_fill_share"
                ]
                row[f"{strategy}_best_quote_fill_share"] = result[
                    "best_quote_fill_share"
                ]
                row[f"{strategy}_bid_cancels"] = result["bid_cancels"]
                row[f"{strategy}_ask_cancels"] = result["ask_cancels"]
                row[f"{strategy}_expired_orders"] = result["expired_orders"]
            rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_metadata(policy: str) -> Path:
    stem = experiment.monthly_output_stem(policy, "as_gp_drift")
    result_path = HERE / "output" / f"{stem}_results.json"
    output = json.loads(result_path.read_text(encoding="utf-8"))
    output["as_of"] = "2026-09-06"
    execution = output["design"]["execution"]
    execution.update(
        {
            "fill_evidence": "earliest qualifying future trade price OR opposite L1 best quote",
            "active_window": "snapshots t+1 through t+20 whose timestamp is later than t+5ms",
            "through": "buy: bid quote >= future best ask/trade price + 1 tick; sell: ask quote <= future best bid/trade price - 1 tick; either evidence is sufficient",
            "touch": "buy: bid quote >= future best ask OR future trade price; sell: ask quote <= future best bid OR future trade price",
            "unfilled_at_horizon": "cancel after evaluating snapshot t+20",
        }
    )
    output["limitations"] = [
        item
        for item in output["limitations"]
        if "Touch/through" not in item and "Market impact" not in item
    ]
    output["limitations"].extend(
        [
            "Trade-touch or best-quote crossing is an execution proxy: queue-ahead volume, hidden liquidity, order acknowledgements, and cancel acknowledgements are unavailable.",
            "Market impact, short-borrow costs, and broker minimum fees are omitted.",
        ]
    )
    _add_lifecycle_aggregates(output)
    lifecycle_valid = _order_lifecycle_valid(output)
    output["validation"]["order_lifecycle_fill_or_cancel"] = lifecycle_valid
    output["validation"]["fill_evidence_counts_match_maker_fills"] = (
        _fill_evidence_valid(output)
    )
    output["validation"]["all_passed"] = bool(
        all(
            value
            for key, value in output["validation"].items()
            if key != "all_passed"
        )
    )
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_path = HERE / "output" / f"{stem}_validation.json"
    validation_path.write_text(
        json.dumps(output["validation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_daily_csv(output, HERE / "output" / f"{stem}_daily.csv")
    if not output["validation"]["all_passed"]:
        raise AssertionError(output["validation"])
    return result_path


def main() -> None:
    policy = _selected_policy(sys.argv[1:])
    experiment.OUTPUT_DIR = HERE / "output"
    experiment.main()
    result_path = _rewrite_metadata(policy)
    print("UPDATED", result_path, flush=True)


if __name__ == "__main__":
    main()
