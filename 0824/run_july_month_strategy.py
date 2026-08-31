#!/usr/bin/env python3
"""July 2026 walk-forward AS/GP/drift market-making evaluation.

The first valid day contributes a strictly chronological 70/30 fit/calibration
split.  Every later day is a full-day out-of-sample test.  The direction model
uses at most the five preceding complete days, while quote parameters and the
probability-to-drift calibration use at most the five preceding out-of-sample
periods.  No test-day labels, fills, or PnL enter that day's decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date as Date
from pathlib import Path
from typing import Any

import numpy as np

from directional_market_maker import (
    load_or_extract_trades,
    fit_direction_classifier,
    future_mid_move_ticks,
    predict_direction,
    select_quote_indices,
    simulate_market_maker,
)
from lob_reconstruction import reconstruct_mbo_csv, save_reconstruction
from run_directional_markout import pooled_calibrator, tune_model_parameters
from run_multiday_strategy import HORIZON, load_day, per_day_split


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path(
    os.environ.get("DEEPLOB_7709_DATA_DIR", Path.home() / "Downloads" / "7709_202607")
).expanduser()
MONTH_CACHE = HERE / "data" / "july_2026"
OUTPUT_DIR = HERE / "output"
MODEL_TRAIN_DAYS = 5
PARAMETER_VALIDATION_PERIODS = 5
FILE_PATTERN = re.compile(r"hk07709_(2026-07-\d{2})\.csv$")
STRATEGIES = ("directional", "classic_as", "gp_style_discrete", "paired_no_signal")
FILL_MODES = ("through", "touch")


def _book_cache(date: str, policy: str = "strict") -> Path:
    suffix = "mbo_tolerant_5m_s10" if policy == "tolerant" else "mbo_strict_s10"
    return MONTH_CACHE / f"hk07709_{date}_{suffix}.npz"


def monthly_output_stem(policy: str, experiment: str) -> str:
    prefix = "july_2026_tolerant" if policy == "tolerant" else "july_2026"
    return f"{prefix}_{experiment}"


def _trade_cache(date: str) -> Path:
    return MONTH_CACHE / f"hk07709_{date}_trades.npz"


def discover_sources(raw_dir: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for path in sorted(raw_dir.glob("hk07709_2026-07-*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            sources[match.group(1)] = path
    if not sources:
        raise FileNotFoundError(f"no July 2026 HK 07709 CSV files under {raw_dir}")
    return sources


def reconstruct_one(payload: tuple[str, str, bool, str]) -> dict[str, Any]:
    date, source_text, force, policy = payload
    source = Path(source_text)
    destination = _book_cache(date, policy)
    try:
        if force or not destination.exists():
            result = reconstruct_mbo_csv(
                source, snapshot_every=10, policy=policy
            )
            save_reconstruction(destination, result)
            metadata = result.metadata
            status = "reconstructed"
        else:
            with np.load(destination, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            status = "cached"
        first_send_time = int(metadata["first_send_time"])
        last_send_time = int(metadata["last_send_time"])
        first_time_of_day = first_send_time % 1_000_000_000
        last_time_of_day = last_send_time % 1_000_000_000
        full_session_coverage = bool(
            first_time_of_day <= 93_500_000
            and last_time_of_day >= 155_500_000
        )
        return {
            "date": date,
            "status": status,
            "cache": str(destination.resolve()),
            "snapshots": int(metadata["snapshot_count"]),
            "tainted": bool(metadata.get("final_state_tainted", False)),
            "order_errors": metadata.get("order_errors", {}),
            "reconstruction_policy": metadata.get(
                "reconstruction_policy", "strict"
            ),
            "gap_events": metadata.get("gap_events", []),
            "recovery_skipped_changed_groups": int(
                metadata.get("quality_counts", {}).get(
                    "skipped_recovery_changed_groups", 0
                )
            ),
            "first_send_time": first_send_time,
            "last_send_time": last_send_time,
            "full_session_coverage": full_session_coverage,
        }
    except Exception as error:  # Keep non-trading/corrupt files in the audit.
        return {
            "date": date,
            "status": "excluded",
            "source": str(source.resolve()),
            "reason": f"{type(error).__name__}: {error}",
        }


def extract_trades_one(payload: tuple[str, str, bool]) -> dict[str, Any]:
    date, source_text, force = payload
    source = Path(source_text)
    destination = _trade_cache(date)
    if force and destination.exists():
        destination.unlink()
    trades = load_or_extract_trades(source, destination)
    return {
        "date": date,
        "trade_prints": int(len(trades.price_hkd)),
        "cache": str(destination.resolve()),
    }


def parallel_map(
    function: Any,
    payloads: list[tuple[Any, ...]],
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(label, record["date"], record.get("status", "ok"), flush=True)
    return sorted(records, key=lambda item: item["date"])


def subset_values(
    eligible: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(eligible, selected)
    if np.any(positions >= len(eligible)) or not np.array_equal(
        eligible[positions], selected
    ):
        raise ValueError("selected quote indices are not eligible")
    return values[positions]


def validation_record(
    day: Any,
    trades: Any,
    eligible: np.ndarray,
    score: np.ndarray,
) -> dict[str, Any]:
    return {
        "day": day,
        "trades": trades,
        "eligible": eligible,
        "score": score,
        "future_ticks": future_mid_move_ticks(day, eligible, HORIZON),
    }


def compact_tuning(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": record["model"],
        "objective": record["objective"],
        "candidate_count": record["candidate_count"],
        "validation_dates": record["validation_dates"],
        "calibrator": record["calibrator"],
        "as_market_inputs": record["as_market_inputs"],
        "best_parameters": record["best_parameters"],
        "best": record["best"],
    }


def evaluate_day(
    day: Any,
    trades: Any,
    score: np.ndarray,
    classification: dict[str, Any],
    calibrator: Any,
    directional_parameters: Any,
    gp_parameters: Any,
    as_parameters: Any,
    tuning: dict[str, dict[str, Any]],
    train_dates: list[str],
) -> dict[str, Any]:
    quote_indices = select_quote_indices(day, day.eligible, HORIZON)
    quote_alpha = subset_values(
        day.eligible, calibrator.transform(score), quote_indices
    )
    parameters = {
        "directional": directional_parameters,
        "classic_as": as_parameters,
        "gp_style_discrete": gp_parameters,
        "paired_no_signal": replace(directional_parameters, signal_strength=0.0),
    }
    results: dict[str, dict[str, Any]] = {}
    for strategy, strategy_parameters in parameters.items():
        results[strategy] = {}
        for fill_mode in FILL_MODES:
            result, _ = simulate_market_maker(
                day,
                trades,
                quote_indices,
                quote_alpha,
                strategy_parameters,
                fill_mode,
                strategy,
                record_events=False,
            )
            results[strategy][fill_mode] = result

    comparisons: dict[str, dict[str, float]] = {}
    for fill_mode in FILL_MODES:
        directional = results["directional"][fill_mode]
        comparisons[fill_mode] = {}
        for baseline in ("classic_as", "gp_style_discrete", "paired_no_signal"):
            other = results[baseline][fill_mode]
            comparisons[fill_mode][f"directional_minus_{baseline}_hkd"] = float(
                directional["net_pnl_hkd"]
            ) - float(other["net_pnl_hkd"])

    return {
        "test_date": day.date,
        "model_train_dates": train_dates,
        "parameter_validation_dates": tuning["directional"]["validation_dates"],
        "classification": classification,
        "calibrator": asdict(calibrator),
        "quote_opportunities": int(len(quote_indices)),
        "parameters": {name: asdict(value) for name, value in parameters.items()},
        "tuning": {name: compact_tuning(value) for name, value in tuning.items()},
        "strategies": results,
        "comparisons": comparisons,
    }


def daily_max_drawdown(values: list[float]) -> float:
    equity = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=np.float64))))
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def aggregate_tests(tests: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for fill_mode in FILL_MODES:
        aggregate[fill_mode] = {}
        for strategy in STRATEGIES:
            rows = [test["strategies"][strategy][fill_mode] for test in tests]
            daily_net = [float(row["net_pnl_hkd"]) for row in rows]
            gross = sum(float(row["gross_pnl_hkd"]) for row in rows)
            official = sum(float(row["official_only_fees_hkd"]) for row in rows)
            fees = sum(float(row["fees_hkd"]) for row in rows)
            maker_turnover = sum(float(row["maker_turnover_hkd"]) for row in rows)
            feeable_turnover = sum(float(row["feeable_turnover_hkd"]) for row in rows)
            positive = sum(value for value in daily_net if value > 0)
            negative = -sum(value for value in daily_net if value < 0)
            aggregate[fill_mode][strategy] = {
                "test_days": len(rows),
                "profitable_days": sum(value > 0 for value in daily_net),
                "gross_pnl_hkd": gross,
                "official_only_fees_hkd": official,
                "all_in_fees_hkd": fees,
                "net_pnl_official_only_hkd": gross - official,
                "net_pnl_hkd": gross - fees,
                "mean_daily_net_pnl_hkd": float(np.mean(daily_net)),
                "median_daily_net_pnl_hkd": float(np.median(daily_net)),
                "daily_profit_factor": positive / negative if negative else None,
                "day_close_max_drawdown_hkd": daily_max_drawdown(daily_net),
                "worst_intraday_drawdown_hkd": max(
                    float(row["max_drawdown_hkd"]) for row in rows
                ),
                "maker_fills": sum(int(row["maker_fills"]) for row in rows),
                "buy_fills": sum(int(row["buy_fills"]) for row in rows),
                "sell_fills": sum(int(row["sell_fills"]) for row in rows),
                "maker_turnover_hkd": maker_turnover,
                "feeable_turnover_hkd": feeable_turnover,
                "gross_bps_of_feeable_turnover": (
                    gross / feeable_turnover * 10_000 if feeable_turnover else None
                ),
                "net_bps_of_feeable_turnover": (
                    (gross - fees) / feeable_turnover * 10_000
                    if feeable_turnover
                    else None
                ),
            }
        directional = aggregate[fill_mode]["directional"]
        directional["incremental_net_pnl_hkd"] = {
            baseline: directional["net_pnl_hkd"]
            - aggregate[fill_mode][baseline]["net_pnl_hkd"]
            for baseline in ("classic_as", "gp_style_discrete", "paired_no_signal")
        }
        directional["beats_baseline_days"] = {
            baseline: sum(
                float(test["strategies"]["directional"][fill_mode]["net_pnl_hkd"])
                > float(test["strategies"][baseline][fill_mode]["net_pnl_hkd"])
                for test in tests
            )
            for baseline in ("classic_as", "gp_style_discrete", "paired_no_signal")
        }
    samples = sum(int(test["classification"]["samples"]) for test in tests)
    aggregate["direction_prediction"] = {
        "test_days": len(tests),
        "samples": samples,
        "weighted_accuracy": sum(
            float(test["classification"]["accuracy"])
            * int(test["classification"]["samples"])
            for test in tests
        )
        / samples,
        "weighted_macro_f1": sum(
            float(test["classification"]["macro_f1"])
            * int(test["classification"]["samples"])
            for test in tests
        )
        / samples,
        "mean_probability_edge": float(
            np.mean(
                [test["classification"]["mean_probability_edge"] for test in tests]
            )
        ),
    }
    return aggregate


def validation_checks(output: dict[str, Any]) -> dict[str, Any]:
    tests = output["tests"]
    chronological = all(
        all(train_date < test["test_date"] for train_date in test["model_train_dates"])
        and all(
            validation_date.split()[0] < test["test_date"]
            for validation_date in test["parameter_validation_dates"]
        )
        for test in tests
    )
    accounting = True
    for test in tests:
        for strategy in STRATEGIES:
            for fill_mode in FILL_MODES:
                row = test["strategies"][strategy][fill_mode]
                accounting &= bool(
                    np.isclose(
                        row["gross_pnl_hkd"] - row["fees_hkd"],
                        row["net_pnl_hkd"],
                    )
                )
                accounting &= int(row["maker_fills"]) == int(row["buy_fills"]) + int(
                    row["sell_fills"]
                )
                accounting &= int(row["max_abs_inventory_lots"]) <= int(
                    row["parameters"]["max_inventory_lots"]
                )
    aggregate_match = True
    for fill_mode in FILL_MODES:
        for strategy in STRATEGIES:
            expected = sum(
                float(test["strategies"][strategy][fill_mode]["net_pnl_hkd"])
                for test in tests
            )
            actual = output["aggregate"][fill_mode][strategy]["net_pnl_hkd"]
            aggregate_match &= bool(np.isclose(expected, actual))
    checks = {
        "chronological_no_future_training_or_tuning": chronological,
        "fee_fill_and_inventory_accounting": accounting,
        "monthly_aggregate_recomputed": aggregate_match,
        "all_test_days_end_flat": all(
            test["strategies"][strategy][fill_mode]["end_flatten_side"]
            in {"none", "buy", "sell"}
            for test in tests
            for strategy in STRATEGIES
            for fill_mode in FILL_MODES
        ),
        "selected_dates_have_open_to_close_snapshot_coverage": all(
            record.get("full_session_coverage", False)
            for record in output["source_audit"]["reconstruction"]
            if record.get("date")
            in output["source_audit"]["clean_trading_dates"]
        ),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def write_daily_csv(tests: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for test in tests:
        for fill_mode in FILL_MODES:
            row: dict[str, Any] = {
                "date": test["test_date"],
                "fill_mode": fill_mode,
                "direction_accuracy": test["classification"]["accuracy"],
                "direction_macro_f1": test["classification"]["macro_f1"],
                "drift_slope_ticks": test["calibrator"]["slope_ticks"],
            }
            for strategy in STRATEGIES:
                result = test["strategies"][strategy][fill_mode]
                row[f"{strategy}_net_pnl_hkd"] = result["net_pnl_hkd"]
                row[f"{strategy}_gross_pnl_hkd"] = result["gross_pnl_hkd"]
                row[f"{strategy}_maker_fills"] = result["maker_fills"]
            rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force-reconstruct", action="store_true")
    parser.add_argument(
        "--reconstruction-policy",
        choices=("strict", "tolerant"),
        default="strict",
    )
    args = parser.parse_args()
    policy = args.reconstruction_policy
    raw_dir = args.raw_dir.expanduser().resolve()
    sources = discover_sources(raw_dir)
    MONTH_CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [
        (date, str(path.resolve()), bool(args.force_reconstruct), policy)
        for date, path in sources.items()
    ]

    reconstruction = parallel_map(
        reconstruct_one, payloads, args.workers, "BOOK"
    )
    usable_dates = [
        record["date"]
        for record in reconstruction
        if record["status"] != "excluded"
        and not record["tainted"]
        and record["full_session_coverage"]
    ]
    if len(usable_dates) < 2:
        raise RuntimeError("fewer than two clean reconstructed trading days")
    trade_payloads = [
        (date, str(sources[date].resolve()), bool(args.force_reconstruct))
        for date in usable_dates
    ]
    trade_audit = parallel_map(
        extract_trades_one, trade_payloads, args.workers, "TRADES"
    )

    days = {
        date: load_day(
            date,
            "tolerant_mbo" if policy == "tolerant" else "strict_mbo",
            _book_cache(date, policy),
        )
        for date in usable_dates
    }
    trades = {
        date: load_or_extract_trades(sources[date], _trade_cache(date))
        for date in usable_dates
    }
    first_day = days[usable_dates[0]]
    first_fit, first_validation = per_day_split(first_day)
    initial_estimator, initial_alpha = fit_direction_classifier(
        [first_day], [first_fit]
    )
    initial_score, initial_metrics = predict_direction(
        initial_estimator, initial_alpha, first_day, first_validation
    )
    validation_records = [
        validation_record(
            first_day,
            trades[first_day.date],
            first_validation,
            initial_score,
        )
    ]
    tests: list[dict[str, Any]] = []

    for position, test_date in enumerate(usable_dates[1:], start=1):
        test_day = days[test_date]
        training_dates = usable_dates[max(0, position - MODEL_TRAIN_DAYS) : position]
        training_days = [days[date] for date in training_dates]
        estimator, alpha = fit_direction_classifier(
            training_days, [day.eligible for day in training_days]
        )
        score, classification = predict_direction(
            estimator, alpha, test_day, test_day.eligible
        )
        recent_records = validation_records[-PARAMETER_VALIDATION_PERIODS:]
        calibrator = pooled_calibrator(recent_records)
        directional, directional_tuning = tune_model_parameters(
            recent_records, calibrator, "directional"
        )
        gp, gp_tuning = tune_model_parameters(
            recent_records, calibrator, "gp_style_discrete"
        )
        classic_as, as_tuning = tune_model_parameters(
            recent_records, calibrator, "classic_as"
        )
        tuning = {
            "directional": directional_tuning,
            "gp_style_discrete": gp_tuning,
            "classic_as": as_tuning,
        }
        result = evaluate_day(
            test_day,
            trades[test_date],
            score,
            classification,
            calibrator,
            directional,
            gp,
            classic_as,
            tuning,
            training_dates,
        )
        tests.append(result)
        validation_records.append(
            validation_record(
                test_day, trades[test_date], test_day.eligible, score
            )
        )
        through = result["strategies"]
        print(
            "TEST",
            test_date,
            f"acc={classification['accuracy']:.4f}",
            f"AS={through['classic_as']['through']['net_pnl_hkd']:.2f}",
            f"GP={through['gp_style_discrete']['through']['net_pnl_hkd']:.2f}",
            f"DRIFT={through['directional']['through']['net_pnl_hkd']:.2f}",
            flush=True,
        )

    output: dict[str, Any] = {
        "as_of": "2026-08-31",
        "instrument": "HK 07709",
        "source_directory": str(raw_dir),
        "design": {
            "data_family": (
                "tolerant MBO reconstruction: ignore unknown modify/delete, quarantine five minutes after each >30s continuous-session gap, and split segments"
                if policy == "tolerant"
                else "strict MBO reconstruction, snapshot every 10 changed SendTime groups"
            ),
            "reconstruction_policy": policy,
            "initial_period": f"{first_day.date} first 70% model fit / last 30% calibration only",
            "full_day_out_of_sample_tests": [test["test_date"] for test in tests],
            "direction_model": "class-balanced multinomial logistic regression on 25 causal LOB features",
            "direction_model_rolling_train_days": MODEL_TRAIN_DAYS,
            "parameter_validation_rolling_periods": PARAMETER_VALIDATION_PERIODS,
            "drift": "p(up)-p(down), linearly calibrated on prior OOS periods to expected 20-snapshot mid move in ticks",
            "strategies": {
                "classic_as": "finite-horizon AS closed-form quotes with prior-period variance and fill-decay estimates",
                "gp_style_discrete": "validation-tuned discrete tick/inventory approximation; not the original GP QVI solution",
                "directional": "GP-style discrete quote center plus calibrated directional drift",
                "paired_no_signal": "same directional parameters with signal strength forced to zero",
            },
            "execution": {
                "lot_size": 100,
                "latency_ms": 5,
                "quote_horizon_snapshots": HORIZON,
                "through": "conservative proxy: later trade at least one tick beyond quote",
                "touch": "optimistic proxy: later trade reaches quote",
                "daily_terminal_flattening": True,
            },
            "cost_bps_per_execution_side": {
                "official_fixed": 1.27,
                "brokerage_assumption": 1.50,
                "all_in": 2.77,
            },
        },
        "source_audit": {
            "files_found": len(sources),
            "calendar_files": [
                {
                    "date": date,
                    "weekday": Date.fromisoformat(date).strftime("%A"),
                    "bytes": path.stat().st_size,
                }
                for date, path in sources.items()
            ],
            "reconstruction": reconstruction,
            "trades": trade_audit,
            "clean_trading_dates": usable_dates,
        },
        "initial_calibration": {
            "date": first_day.date,
            "classification": initial_metrics,
            "samples": int(len(first_validation)),
        },
        "tests": tests,
        "aggregate": aggregate_tests(tests),
        "limitations": [
            "The first day's last 30% is calibration, so the primary PnL covers every later usable trading day rather than an in-sample first-day PnL.",
            (
                "Tolerant MBO assumes missing modify/delete events can be ignored after a five-minute post-gap quarantine; without an absolute snapshot, exact aggregate depth recovery cannot be proven."
                if policy == "tolerant"
                else "Strict MBO excludes the remainder of a day after the first impossible order transition."
            ),
            "A turnover yield is reported; an account return is undefined without user-specified capital, margin, short-borrow, and minimum-commission rules.",
            "Touch/through are queue sensitivity assumptions because exact hypothetical queue position is unavailable.",
            "The GP baseline is a practical discrete inventory approximation, not a numerical solution of the original Guilbaud-Pham QVI.",
            "Market impact, queue-ahead volume, order acknowledgement/cancel latency beyond the fixed 5ms assumption, short-borrow costs, and broker minimum fees are omitted.",
        ],
    }
    output["validation"] = validation_checks(output)
    stem = monthly_output_stem(policy, "as_gp_drift")
    result_path = OUTPUT_DIR / f"{stem}_results.json"
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = OUTPUT_DIR / f"{stem}_daily.csv"
    write_daily_csv(tests, csv_path)
    validation_path = OUTPUT_DIR / f"{stem}_validation.json"
    validation_path.write_text(
        json.dumps(output["validation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("WROTE", result_path, csv_path, validation_path, flush=True)
    if not output["validation"]["all_passed"]:
        raise AssertionError(output["validation"])


if __name__ == "__main__":
    main()
