#!/usr/bin/env python3
"""Causal July-2026 replay of the paper-faithful GP/QVI policy on HK 07709."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date as Date
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEGACY_DIR = ROOT / "0824"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(LEGACY_DIR) not in sys.path:
    sys.path.append(str(LEGACY_DIR))

from directional_market_maker import load_or_extract_trades  # noqa: E402
from gp_qvi_model import (  # noqa: E402
    FILL_MODES,
    SESSION_BUCKETS,
    GPConfig,
    aggregate_calibration_stats,
    collect_daily_calibration_stats,
    config_asdict,
    policy_audit,
    simulate_gp_day,
    solve_gp_policy,
)
from run_multiday_strategy import load_day  # noqa: E402


SOURCE_RESULT = HERE / "output/july_2026_tolerant_as_gp_drift_results.json"
OUTPUT_DIR = HERE / "output"
RESULT_PATH = OUTPUT_DIR / "july_2026_gp_qvi_new_fill_results.json"
DAILY_PATH = OUTPUT_DIR / "july_2026_gp_qvi_new_fill_daily.csv"
VALIDATION_PATH = OUTPUT_DIR / "july_2026_gp_qvi_new_fill_validation.json"
ROLLING_DAYS = 5
STRATEGIES = ("gp_qvi", "gp_qvi_womo", "constant_best")


def _portable_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep provenance while avoiding machine-specific absolute paths."""

    portable: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for field in ("source", "cache"):
            if item.get(field):
                item[field] = Path(str(item[field])).name
        portable.append(item)
    return portable


def _load_inputs() -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, Any]]:
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    usable_dates = list(source_result["source_audit"]["clean_trading_dates"])
    source_dir = Path(source_result["source_directory"])
    days: dict[str, Any] = {}
    trades: dict[str, Any] = {}
    for date in usable_dates:
        book_cache = LEGACY_DIR / (
            f"data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz"
        )
        trade_cache = LEGACY_DIR / f"data/july_2026/hk07709_{date}_trades.npz"
        days[date] = load_day(date, "tolerant_mbo", book_cache)
        trades[date] = load_or_extract_trades(
            source_dir / f"hk07709_{date}.csv", trade_cache
        )
    return source_result, usable_dates, days, trades


def _aggregate(tests: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for fill_mode in FILL_MODES:
        output[fill_mode] = {}
        for strategy in STRATEGIES:
            rows = [test["strategies"][strategy][fill_mode] for test in tests]
            daily_net = np.asarray([row["net_pnl_hkd"] for row in rows], dtype=np.float64)
            daily_gross = np.asarray(
                [row["gross_pnl_hkd"] for row in rows], dtype=np.float64
            )
            cumulative = np.cumsum(daily_net)
            cumulative_with_zero = np.concatenate([[0.0], cumulative])
            cumulative_drawdown = np.maximum.accumulate(cumulative_with_zero) - cumulative_with_zero
            positive = float(daily_net[daily_net > 0].sum())
            negative = float(-daily_net[daily_net < 0].sum())
            sums = {
                key: sum(float(row[key]) for row in rows)
                for key in (
                    "gross_pnl_hkd",
                    "official_only_fees_hkd",
                    "all_in_fees_hkd",
                    "net_pnl_official_only_hkd",
                    "net_pnl_hkd",
                    "maker_turnover_hkd",
                    "market_turnover_hkd",
                    "feeable_turnover_hkd",
                )
            }
            counts = {
                key: sum(int(row[key]) for row in rows)
                for key in (
                    "quoted_bid",
                    "quoted_ask",
                    "maker_fills",
                    "buy_fills",
                    "sell_fills",
                    "trade_price_fills",
                    "best_quote_fills",
                    "best_ask_fills",
                    "best_bid_fills",
                    "bid_cancels",
                    "ask_cancels",
                    "expired_orders",
                    "market_orders",
                    "market_order_lots",
                    "terminal_flatten_lots",
                )
            }
            feeable = sums["feeable_turnover_hkd"]
            output[fill_mode][strategy] = {
                "test_days": len(rows),
                "profitable_days": int(np.sum(daily_net > 0)),
                "losing_days": int(np.sum(daily_net < 0)),
                **sums,
                **counts,
                "mean_daily_net_pnl_hkd": float(np.mean(daily_net)),
                "median_daily_net_pnl_hkd": float(np.median(daily_net)),
                "daily_net_pnl_std_hkd": float(np.std(daily_net, ddof=1)),
                "daily_profit_factor": positive / negative if negative else None,
                "day_close_max_drawdown_hkd": float(np.max(cumulative_drawdown)),
                "worst_intraday_drawdown_hkd": max(
                    float(row["max_drawdown_hkd"]) for row in rows
                ),
                "gross_bps_of_feeable_turnover": (
                    sums["gross_pnl_hkd"] / feeable * 10_000 if feeable else None
                ),
                "net_bps_of_feeable_turnover": (
                    sums["net_pnl_hkd"] / feeable * 10_000 if feeable else None
                ),
                "maker_fill_rate": (
                    counts["maker_fills"]
                    / max(counts["quoted_bid"] + counts["quoted_ask"], 1)
                ),
                "trade_price_fill_share": (
                    counts["trade_price_fills"] / max(counts["maker_fills"], 1)
                ),
                "best_quote_fill_share": (
                    counts["best_quote_fills"] / max(counts["maker_fills"], 1)
                ),
            }
    return output


def _validation(output: dict[str, Any]) -> dict[str, Any]:
    tests = output["tests"]
    chronological = all(
        all(
            Date.fromisoformat(date) < Date.fromisoformat(test["test_date"])
            for date in test["calibration_dates"]
        )
        for test in tests
    )
    rows = [
        test["strategies"][strategy][fill_mode]
        for test in tests
        for strategy in STRATEGIES
        for fill_mode in FILL_MODES
    ]
    lifecycle = all(
        int(row["quoted_bid"]) == int(row["buy_fills"]) + int(row["bid_cancels"])
        and int(row["quoted_ask"])
        == int(row["sell_fills"]) + int(row["ask_cancels"])
        for row in rows
    )
    evidence = all(
        int(row["maker_fills"])
        == int(row["trade_price_fills"]) + int(row["best_quote_fills"])
        and int(row["best_quote_fills"])
        == int(row["best_ask_fills"]) + int(row["best_bid_fills"])
        for row in rows
    )
    flat = all(int(row["end_inventory_lots"]) == 0 for row in rows)
    fee_identity = all(
        np.isclose(
            float(row["net_pnl_hkd"]),
            float(row["gross_pnl_hkd"]) - float(row["all_in_fees_hkd"]),
            atol=1e-6,
        )
        for row in rows
    )
    womo_has_no_policy_market_orders = all(
        int(test["strategies"]["gp_qvi_womo"][fill_mode]["policy_market_orders"])
        == 0
        for test in tests
        for fill_mode in FILL_MODES
    )
    aggregate_reconciles = all(
        np.isclose(
            output["aggregate"][fill_mode][strategy]["net_pnl_hkd"],
            sum(
                float(test["strategies"][strategy][fill_mode]["net_pnl_hkd"])
                for test in tests
            ),
            atol=1e-6,
        )
        for strategy in STRATEGIES
        for fill_mode in FILL_MODES
    )
    checks = {
        "strictly_prior_rolling_calibration": chronological,
        "order_lifecycle_reconciles": lifecycle,
        "fill_evidence_reconciles": evidence,
        "all_days_end_flat": flat,
        "net_equals_gross_minus_all_in_fees": fee_identity,
        "womo_has_no_policy_market_orders": womo_has_no_policy_market_orders,
        "monthly_aggregate_reconciles_to_daily": aggregate_reconciles,
        "test_day_count_is_19": len(tests) == 19,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def _write_daily(tests: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for test in tests:
        for fill_mode in FILL_MODES:
            row: dict[str, Any] = {
                "date": test["test_date"],
                "fill_mode": fill_mode,
                "tick_hkd": test["tick_hkd"],
                "calibration_dates": "|".join(test["calibration_dates"]),
            }
            for strategy in STRATEGIES:
                result = test["strategies"][strategy][fill_mode]
                for metric in (
                    "gross_pnl_hkd",
                    "all_in_fees_hkd",
                    "net_pnl_hkd",
                    "maker_fills",
                    "market_order_lots",
                    "max_abs_inventory_lots",
                    "max_drawdown_hkd",
                ):
                    row[f"{strategy}_{metric}"] = result[metric]
            rows.append(row)
    with DAILY_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-days", type=int, default=ROLLING_DAYS)
    parser.add_argument("--limit-test-days", type=int)
    args = parser.parse_args()
    config = GPConfig()
    source_result, usable_dates, days, trades = _load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_stats: dict[str, Any] = {}
    for date in usable_dates:
        print("CALIBRATION-STATS", date, flush=True)
        daily_stats[date] = collect_daily_calibration_stats(
            days[date], trades[date], config
        )

    test_dates = usable_dates[1:]
    if args.limit_test_days is not None:
        test_dates = test_dates[: args.limit_test_days]
    tests: list[dict[str, Any]] = []
    for position, test_date in enumerate(test_dates, start=1):
        calibration_dates = usable_dates[max(0, position - args.rolling_days):position]
        inputs = aggregate_calibration_stats(
            [daily_stats[date] for date in calibration_dates], config
        )
        tick = float(daily_stats[test_date].tick_hkd)
        strategy_results = {strategy: {} for strategy in STRATEGIES}
        audit: dict[str, Any] = {}
        for fill_mode in FILL_MODES:
            policies = {
                bucket: solve_gp_policy(
                    inputs, tick, fill_mode, bucket, config, market_orders_enabled=True
                )
                for bucket in range(len(SESSION_BUCKETS))
            }
            womo_policies = {
                bucket: solve_gp_policy(
                    inputs, tick, fill_mode, bucket, config, market_orders_enabled=False
                )
                for bucket in range(len(SESSION_BUCKETS))
            }
            strategy_results["gp_qvi"][fill_mode] = simulate_gp_day(
                days[test_date], trades[test_date], policies, config, fill_mode, "gp_qvi"
            )
            strategy_results["gp_qvi_womo"][fill_mode] = simulate_gp_day(
                days[test_date], trades[test_date], womo_policies, config,
                fill_mode, "gp_qvi_womo"
            )
            strategy_results["constant_best"][fill_mode] = simulate_gp_day(
                days[test_date], trades[test_date], None, config,
                fill_mode, "constant_best"
            )
            audit[fill_mode] = {
                "gp_qvi_opening_bucket": policy_audit(policies[0]),
                "gp_qvi_womo_opening_bucket": policy_audit(womo_policies[0]),
            }
        test = {
            "test_date": test_date,
            "calibration_dates": calibration_dates,
            "tick_hkd": tick,
            "calibration": inputs.audit,
            "policy_audit": audit,
            "strategies": strategy_results,
        }
        tests.append(test)
        through = strategy_results["gp_qvi"]["through"]
        touch = strategy_results["gp_qvi"]["touch"]
        print(
            "TEST", test_date,
            f"tick={tick:.3f}",
            f"through={through['net_pnl_hkd']:.2f}",
            f"touch={touch['net_pnl_hkd']:.2f}",
            flush=True,
        )

    output: dict[str, Any] = {
        "as_of": "2026-09-10",
        "instrument": "HK 07709",
        "period": "2026-07",
        "design": {
            "model": "Guilbaud-Pham mean criterion with running quadratic inventory penalty",
            "value_reduction": "v_i(t,x,y,p)=x+yp+phi_i(t,y)",
            "controls": {
                "limit_quotes": "none, current best, or improve current best by one legal tick on each side",
                "limit_size": "one board lot (100 shares)",
                "market_order": "inventory-reducing impulse, enabled only for gp_qvi",
                "inventory_grid_lots": [
                    -config.max_inventory_lots, config.max_inventory_lots
                ],
            },
            "paper_parameter_mapping": {
                "horizon_seconds": config.horizon_seconds,
                "time_steps": config.time_steps,
                "inventory_penalty_gamma": config.inventory_penalty_gamma,
                "gamma_scaling": "inventory measured in 100-share lots; coefficient is HKD-equivalent per lot^2-second",
                "spread_states": config.spread_states,
            },
            "calibration": {
                "rolling_complete_prior_days": args.rolling_days,
                "rho": "spread-change transition counts, row normalised",
                "lambda_t": "spread changes / observed seconds in six intraday buckets",
                "execution_intensity": "fills / time at risk by side, quote level, spread state and fill mode",
                "test_day_excluded": True,
            },
            "execution": {
                "new_fill_logic": "earliest qualifying future trade price OR opposite L1 best quote",
                "latency_ms": config.latency_ms,
                "lifetime_snapshots": config.quote_horizon_snapshots,
                "through": "requires one actual daily tick beyond the order price",
                "touch": "price equality is sufficient",
                "unfilled": "cancel at t+20 inclusive",
                "daily_terminal_flattening": True,
            },
            "cost_bps_per_execution_side": {
                "official_fixed": 1.27,
                "brokerage_assumption": 1.50,
                "all_in": 2.77,
            },
            "strategies": {
                "gp_qvi": "paper-form GP policy with limit and inventory-reducing market orders",
                "gp_qvi_womo": "same solved GP policy with market-order impulse disabled",
                "constant_best": "paper benchmark: one lot at current best bid and ask",
            },
        },
        "source_audit": {
            "source_directory": "DEEPLOB_7709_DATA_DIR (local, not committed)",
            "calendar_files": source_result["source_audit"]["calendar_files"],
            "reconstruction": _portable_audit_rows(
                source_result["source_audit"]["reconstruction"]
            ),
            "clean_trading_dates": usable_dates,
            "initial_calibration_date": usable_dates[0],
            "full_day_out_of_sample_tests": [test["test_date"] for test in tests],
            "excluded_from_clean_days": [
                row
                for row in _portable_audit_rows(
                    source_result["source_audit"]["reconstruction"]
                )
                if row.get("date") not in usable_dates
            ],
        },
        "config": config_asdict(config),
        "tests": tests,
    }
    output["aggregate"] = _aggregate(tests)
    output["validation"] = _validation(output)
    RESULT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_daily(tests)
    VALIDATION_PATH.write_text(
        json.dumps(output["validation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("WROTE", RESULT_PATH, DAILY_PATH, VALIDATION_PATH, flush=True)
    if args.limit_test_days is None and not output["validation"]["all_passed"]:
        raise AssertionError(output["validation"])


if __name__ == "__main__":
    main()
