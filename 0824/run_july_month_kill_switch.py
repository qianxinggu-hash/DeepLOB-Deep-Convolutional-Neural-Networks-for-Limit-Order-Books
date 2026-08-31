#!/usr/bin/env python3
"""Extend the original direction+inventory alpha-kill-switch experiment to July.

This intentionally follows ``run_directional_market_maker.py`` rather than the
later forced-drift model-comparison experiment.  Each test day uses only the
immediately preceding out-of-sample period for drift calibration and quote
parameter selection.  Non-zero drift is deployed only when its conservative
validation objective beats the independently tuned no-signal baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from directional_market_maker import (
    fit_alpha_calibrator,
    fit_direction_classifier,
    predict_direction,
    select_quote_indices,
    simulate_market_maker,
)
from run_directional_market_maker import activate_signal, tune_parameters
from run_july_month_strategy import (
    FILL_MODES,
    MODEL_TRAIN_DAYS,
    OUTPUT_DIR,
    _book_cache,
    _trade_cache,
    daily_max_drawdown,
    discover_sources,
    monthly_output_stem,
    subset_values,
    validation_record,
)
from run_multiday_strategy import HORIZON, load_day, per_day_split
from directional_market_maker import load_or_extract_trades


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path.home() / "Downloads" / "7709_202607"
STRATEGIES = (
    "traditional_baseline",
    "directional_kill_switch",
    "paired_no_signal_ablation",
)


def load_clean_dates(result_path: Path) -> list[str]:
    source = json.loads(result_path.read_text(encoding="utf-8"))
    return list(source["source_audit"]["clean_trading_dates"])


def evaluate_day(
    day: Any,
    trades: Any,
    eligible: np.ndarray,
    score: np.ndarray,
    calibrator: Any,
    baseline_parameters: Any,
    candidate_parameters: Any,
    deployed_parameters: Any,
    baseline_tuning: dict[str, Any],
    directional_tuning: dict[str, Any],
    activation: dict[str, Any],
    classification: dict[str, Any],
    training_dates: list[str],
    validation_date: str,
) -> dict[str, Any]:
    quote_indices = select_quote_indices(day, eligible, HORIZON)
    quote_alpha = subset_values(
        eligible, calibrator.transform(score), quote_indices
    )
    parameters = {
        "traditional_baseline": baseline_parameters,
        "directional_kill_switch": deployed_parameters,
        "paired_no_signal_ablation": replace(
            deployed_parameters, signal_strength=0.0
        ),
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

    comparisons: dict[str, dict[str, float | bool]] = {}
    for fill_mode in FILL_MODES:
        switched = results["directional_kill_switch"][fill_mode]
        comparisons[fill_mode] = {}
        for baseline in ("traditional_baseline", "paired_no_signal_ablation"):
            other = results[baseline][fill_mode]
            comparisons[fill_mode][f"minus_{baseline}_hkd"] = float(
                switched["net_pnl_hkd"]
            ) - float(other["net_pnl_hkd"])
        comparisons[fill_mode]["profitable"] = bool(
            switched["net_pnl_hkd"] > 0
        )
    return {
        "test_date": day.date,
        "model_train_dates": training_dates,
        "parameter_validation_date": validation_date,
        "classification": classification,
        "calibrator": asdict(calibrator),
        "quote_opportunities": int(len(quote_indices)),
        "baseline_parameters": asdict(baseline_parameters),
        "directional_candidate_parameters": asdict(candidate_parameters),
        "deployed_parameters": asdict(deployed_parameters),
        "activation": activation,
        "tuning": {
            "traditional_baseline": baseline_tuning,
            "directional_candidate": directional_tuning,
        },
        "strategies": results,
        "comparisons": comparisons,
    }


def aggregate(tests: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for fill_mode in FILL_MODES:
        output[fill_mode] = {}
        for strategy in STRATEGIES:
            rows = [test["strategies"][strategy][fill_mode] for test in tests]
            daily_net = [float(row["net_pnl_hkd"]) for row in rows]
            gross = sum(float(row["gross_pnl_hkd"]) for row in rows)
            official_fees = sum(
                float(row["official_only_fees_hkd"]) for row in rows
            )
            all_in_fees = sum(float(row["fees_hkd"]) for row in rows)
            turnover = sum(float(row["feeable_turnover_hkd"]) for row in rows)
            positive = sum(value for value in daily_net if value > 0)
            negative = -sum(value for value in daily_net if value < 0)
            output[fill_mode][strategy] = {
                "test_days": len(rows),
                "profitable_days": sum(value > 0 for value in daily_net),
                "gross_pnl_hkd": gross,
                "official_only_fees_hkd": official_fees,
                "all_in_fees_hkd": all_in_fees,
                "net_pnl_official_only_hkd": gross - official_fees,
                "net_pnl_hkd": gross - all_in_fees,
                "net_bps_of_feeable_turnover": (
                    (gross - all_in_fees) / turnover * 10_000 if turnover else None
                ),
                "maker_fills": sum(int(row["maker_fills"]) for row in rows),
                "feeable_turnover_hkd": turnover,
                "mean_daily_net_pnl_hkd": float(np.mean(daily_net)),
                "median_daily_net_pnl_hkd": float(np.median(daily_net)),
                "daily_profit_factor": positive / negative if negative else None,
                "day_close_max_drawdown_hkd": daily_max_drawdown(daily_net),
                "worst_intraday_drawdown_hkd": max(
                    float(row["max_drawdown_hkd"]) for row in rows
                ),
            }
        switched = output[fill_mode]["directional_kill_switch"]
        switched["incremental_net_pnl_hkd"] = {
            baseline: switched["net_pnl_hkd"]
            - output[fill_mode][baseline]["net_pnl_hkd"]
            for baseline in (
                "traditional_baseline",
                "paired_no_signal_ablation",
            )
        }
        switched["beats_baseline_days"] = {
            baseline: sum(
                test["comparisons"][fill_mode][f"minus_{baseline}_hkd"] > 0
                for test in tests
            )
            for baseline in (
                "traditional_baseline",
                "paired_no_signal_ablation",
            )
        }
    samples = sum(int(test["classification"]["samples"]) for test in tests)
    output["direction_prediction"] = {
        "samples": samples,
        "weighted_accuracy": sum(
            test["classification"]["accuracy"]
            * int(test["classification"]["samples"])
            for test in tests
        )
        / samples,
        "weighted_macro_f1": sum(
            test["classification"]["macro_f1"]
            * int(test["classification"]["samples"])
            for test in tests
        )
        / samples,
    }
    output["activation"] = {
        "enabled_days": [
            test["test_date"] for test in tests if test["activation"]["enabled"]
        ],
        "disabled_days": [
            test["test_date"] for test in tests if not test["activation"]["enabled"]
        ],
    }
    return output


def compare_forced_drift(
    aggregate_result: dict[str, Any], forced_path: Path
) -> dict[str, Any] | None:
    if not forced_path.exists():
        return None
    forced = json.loads(forced_path.read_text(encoding="utf-8"))
    comparison: dict[str, Any] = {}
    for fill_mode in FILL_MODES:
        switched = aggregate_result[fill_mode]["directional_kill_switch"]
        forced_drift = forced["aggregate"][fill_mode]["directional"]
        classic_as = forced["aggregate"][fill_mode]["classic_as"]
        comparison[fill_mode] = {
            "kill_switch_net_pnl_hkd": switched["net_pnl_hkd"],
            "forced_drift_net_pnl_hkd": forced_drift["net_pnl_hkd"],
            "kill_switch_minus_forced_drift_hkd": switched["net_pnl_hkd"]
            - forced_drift["net_pnl_hkd"],
            "classic_as_net_pnl_hkd_from_model_comparison": classic_as[
                "net_pnl_hkd"
            ],
            "kill_switch_minus_classic_as_hkd": switched["net_pnl_hkd"]
            - classic_as["net_pnl_hkd"],
        }
    return comparison


def validate(output: dict[str, Any]) -> dict[str, bool]:
    chronological = all(
        all(date < test["test_date"] for date in test["model_train_dates"])
        and test["parameter_validation_date"] < test["test_date"]
        for test in output["tests"]
    )
    accounting = True
    for test in output["tests"]:
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
    activation = all(
        bool(test["activation"]["enabled"])
        == (
            float(test["directional_candidate_parameters"]["signal_strength"])
            == float(test["deployed_parameters"]["signal_strength"])
        )
        for test in output["tests"]
    )
    aggregation = all(
        np.isclose(
            output["aggregate"][fill_mode][strategy]["net_pnl_hkd"],
            sum(
                test["strategies"][strategy][fill_mode]["net_pnl_hkd"]
                for test in output["tests"]
            ),
        )
        for fill_mode in FILL_MODES
        for strategy in STRATEGIES
    )
    checks = {
        "chronological_no_future_information": chronological,
        "fee_and_fill_accounting": accounting,
        "kill_switch_deployment_matches_activation": activation,
        "monthly_aggregate_recomputed": aggregation,
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
                "alpha_enabled": test["activation"]["enabled"],
                "validation_score_improvement": test["activation"][
                    "validation_score_improvement_hkd_equivalent"
                ],
                "classification_accuracy": test["classification"]["accuracy"],
                "calibrator_slope_ticks": test["calibrator"]["slope_ticks"],
            }
            for strategy in STRATEGIES:
                result = test["strategies"][strategy][fill_mode]
                row[f"{strategy}_net_pnl_hkd"] = result["net_pnl_hkd"]
                row[f"{strategy}_maker_fills"] = result["maker_fills"]
            rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reconstruction-policy",
        choices=("strict", "tolerant"),
        default="strict",
    )
    args = parser.parse_args()
    policy = args.reconstruction_policy
    raw_dir = DEFAULT_RAW.resolve()
    sources = discover_sources(raw_dir)
    prior_result = OUTPUT_DIR / f"{monthly_output_stem(policy, 'as_gp_drift')}_results.json"
    clean_dates = load_clean_dates(prior_result)
    days = {
        date: load_day(
            date,
            "tolerant_mbo" if policy == "tolerant" else "strict_mbo",
            _book_cache(date, policy),
        )
        for date in clean_dates
    }
    trades = {
        date: load_or_extract_trades(sources[date], _trade_cache(date))
        for date in clean_dates
    }

    first_day = days[clean_dates[0]]
    first_fit, first_validation = per_day_split(first_day)
    initial_estimator, initial_alpha = fit_direction_classifier(
        [first_day], [first_fit]
    )
    initial_score, initial_metrics = predict_direction(
        initial_estimator, initial_alpha, first_day, first_validation
    )
    previous_record = validation_record(
        first_day, trades[first_day.date], first_validation, initial_score
    )
    tests: list[dict[str, Any]] = []

    for position, test_date in enumerate(clean_dates[1:], start=1):
        calibrator = fit_alpha_calibrator(
            previous_record["score"], previous_record["future_ticks"]
        )
        validation_alpha = calibrator.transform(previous_record["score"])
        baseline_parameters, baseline_tuning = tune_parameters(
            previous_record["day"],
            previous_record["trades"],
            previous_record["eligible"],
            validation_alpha,
            directional=False,
        )
        candidate_parameters, directional_tuning = tune_parameters(
            previous_record["day"],
            previous_record["trades"],
            previous_record["eligible"],
            validation_alpha,
            directional=True,
        )
        deployed_parameters, activation = activate_signal(
            baseline_parameters,
            candidate_parameters,
            baseline_tuning,
            directional_tuning,
        )

        training_dates = clean_dates[max(0, position - MODEL_TRAIN_DAYS) : position]
        training_days = [days[date] for date in training_dates]
        estimator, alpha = fit_direction_classifier(
            training_days, [day.eligible for day in training_days]
        )
        test_day = days[test_date]
        score, classification = predict_direction(
            estimator, alpha, test_day, test_day.eligible
        )
        result = evaluate_day(
            test_day,
            trades[test_date],
            test_day.eligible,
            score,
            calibrator,
            baseline_parameters,
            candidate_parameters,
            deployed_parameters,
            baseline_tuning,
            directional_tuning,
            activation,
            classification,
            training_dates,
            previous_record["day"].date,
        )
        tests.append(result)
        previous_record = validation_record(
            test_day, trades[test_date], test_day.eligible, score
        )
        through = result["strategies"]
        print(
            "TEST",
            test_date,
            f"alpha={'ON' if activation['enabled'] else 'OFF'}",
            f"baseline={through['traditional_baseline']['through']['net_pnl_hkd']:.2f}",
            f"switched={through['directional_kill_switch']['through']['net_pnl_hkd']:.2f}",
            flush=True,
        )

    aggregate_result = aggregate(tests)
    output: dict[str, Any] = {
        "as_of": "2026-08-31",
        "instrument": "HK 07709",
        "question": "Does the original validation-gated drift+inventory strategy retain its improvement over July?",
        "design": {
            "source_strategy": "0824/run_directional_market_maker.py",
            "reconstruction_policy": policy,
            "initial_period": f"{first_day.date} first 70% fit / last 30% validation",
            "test_dates": [test["test_date"] for test in tests],
            "direction_model_rolling_train_days": MODEL_TRAIN_DAYS,
            "parameter_validation": "immediately preceding completed out-of-sample period, matching the original experiment",
            "alpha_kill_switch": "deploy non-zero drift only when its prior-period through-fill net-PnL-minus-0.25-drawdown score beats the independently tuned no-signal baseline",
            "parameter_grid": {
                "base_distance_ticks": [0, 1, 2],
                "signal_strength": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                "inventory_skew_ticks": [0.5, 1.0, 2.0],
                "max_inventory_lots": [3, 5],
            },
            "cost_bps_per_execution_side": {
                "official_fixed": 1.27,
                "brokerage_assumption": 1.50,
                "all_in": 2.77,
            },
            "execution": {
                "lot_size": 100,
                "latency_ms": 5,
                "quote_horizon_snapshots": HORIZON,
                "daily_terminal_flattening": True,
                "through": "conservative: trade at least one tick beyond quote",
                "touch": "optimistic: trade reaches quote",
            },
        },
        "initial_calibration": {
            "date": first_day.date,
            "classification": initial_metrics,
        },
        "tests": tests,
        "aggregate": aggregate_result,
        "comparison_to_forced_drift": compare_forced_drift(
            aggregate_result, prior_result
        ),
        "limitations": [
            (
                "Tolerant MBO ignores unknown modify/delete events after a fixed five-minute post-gap quarantine; exact depth recovery cannot be proven without an absolute snapshot."
                if policy == "tolerant"
                else "Only ten full-day out-of-sample tests pass strict MBO quality checks."
            ),
            "The kill switch and quote grid were already selected in prior research; this is an extension, not a pristine strategy-discovery test.",
            "Touch/through remain queue-position sensitivities, not exact fills.",
            "Account return is undefined without capital, margin, borrow, and minimum-commission assumptions.",
        ],
    }
    output["validation"] = validate(output)
    stem = monthly_output_stem(policy, "directional_kill_switch")
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
