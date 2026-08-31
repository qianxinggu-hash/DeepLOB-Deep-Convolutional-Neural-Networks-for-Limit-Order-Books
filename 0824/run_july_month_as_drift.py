#!/usr/bin/env python3
"""July walk-forward comparison of classic AS and AS with directional drift."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from directional_market_maker import (
    ASDriftParameters,
    ASParameters,
    fit_direction_classifier,
    load_or_extract_trades,
    predict_direction,
    select_quote_indices,
    simulate_market_maker,
)
from run_directional_markout import (
    estimate_as_market_inputs,
    pooled_calibrator,
    tune_model_parameters,
)
from run_july_month_strategy import (
    FILL_MODES,
    MODEL_TRAIN_DAYS,
    OUTPUT_DIR,
    PARAMETER_VALIDATION_PERIODS,
    _book_cache,
    _trade_cache,
    daily_max_drawdown,
    discover_sources,
    monthly_output_stem,
    subset_values,
    validation_record,
)
from run_multiday_strategy import HORIZON, load_day, per_day_split


DEFAULT_RAW = Path.home() / "Downloads" / "7709_202607"
STRATEGIES = (
    "classic_as",
    "as_drift_unit",
    "as_drift",
    "paired_as_no_drift",
    "as_drift_kill_switch",
)
GAMMAS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
SIGNAL_STRENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
INVENTORY_CAPS = (3, 5)


def load_clean_dates(result_path: Path) -> list[str]:
    source = json.loads(result_path.read_text(encoding="utf-8"))
    return list(source["source_audit"]["clean_trading_dates"])


def tune_as_drift(
    records: list[dict[str, Any]], calibrator: Any
) -> tuple[ASDriftParameters, dict[str, Any]]:
    variance, decay, market_input_audit = estimate_as_market_inputs(records)
    candidates: list[tuple[float, ASDriftParameters, dict[str, Any]]] = []
    for gamma in GAMMAS:
        for strength in SIGNAL_STRENGTHS:
            for cap in INVENTORY_CAPS:
                parameters = ASDriftParameters(
                    risk_aversion_per_tick=gamma,
                    order_decay_per_tick=decay,
                    horizon_variance_ticks2=variance,
                    signal_strength=strength,
                    max_inventory_lots=cap,
                )
                total_net = 0.0
                total_drawdown = 0.0
                total_fills = 0
                periods: list[dict[str, Any]] = []
                for record in records:
                    day = record["day"]
                    quote_indices = select_quote_indices(
                        day, record["eligible"], HORIZON
                    )
                    quote_alpha = subset_values(
                        record["eligible"],
                        calibrator.transform(record["score"]),
                        quote_indices,
                    )
                    result, _ = simulate_market_maker(
                        day,
                        record["trades"],
                        quote_indices,
                        quote_alpha,
                        parameters,
                        "through",
                        "as_drift_tuning",
                    )
                    total_net += float(result["net_pnl_hkd"])
                    total_drawdown += float(result["max_drawdown_hkd"])
                    total_fills += int(result["maker_fills"])
                    periods.append(
                        {
                            "date": day.date,
                            "net_pnl_hkd": result["net_pnl_hkd"],
                            "max_drawdown_hkd": result["max_drawdown_hkd"],
                            "maker_fills": result["maker_fills"],
                        }
                    )
                objective = total_net - 0.25 * total_drawdown
                if total_fills < 20:
                    objective = -1e18
                candidates.append(
                    (
                        objective,
                        parameters,
                        {
                            "score": objective,
                            "net_pnl_hkd": total_net,
                            "max_drawdown_hkd": total_drawdown,
                            "maker_fills": total_fills,
                            "validation_periods": periods,
                        },
                    )
                )
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, parameters, best = candidates[0]
    return parameters, {
        "model": "as_drift",
        "objective": "all-in net PnL - 0.25 * max drawdown, using conservative through fills",
        "candidate_count": len(candidates),
        "validation_dates": [record["day"].date for record in records],
        "calibrator": asdict(calibrator),
        "as_market_inputs": market_input_audit,
        "best_parameters": asdict(parameters),
        "best": best,
        "top_five": [
            {"parameters": asdict(item[1]), **item[2]} for item in candidates[:5]
        ],
    }


def compact_tuning(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": record["model"],
        "objective": record["objective"],
        "candidate_count": record["candidate_count"],
        "validation_dates": record["validation_dates"],
        "calibrator": record.get("calibrator"),
        "as_market_inputs": record.get("as_market_inputs"),
        "best_parameters": record["best_parameters"],
        "best": record["best"],
        "top_five": record["top_five"],
    }


def evaluate_day(
    day: Any,
    trades: Any,
    score: np.ndarray,
    classification: dict[str, Any],
    calibrator: Any,
    classic_parameters: ASParameters,
    drift_parameters: ASDriftParameters,
    drift_enabled: bool,
    classic_tuning: dict[str, Any],
    drift_tuning: dict[str, Any],
    training_dates: list[str],
) -> dict[str, Any]:
    quote_indices = select_quote_indices(day, day.eligible, HORIZON)
    quote_alpha = subset_values(
        day.eligible, calibrator.transform(score), quote_indices
    )
    paired_parameters = replace(drift_parameters, signal_strength=0.0)
    unit_parameters = ASDriftParameters(
        risk_aversion_per_tick=classic_parameters.risk_aversion_per_tick,
        order_decay_per_tick=classic_parameters.order_decay_per_tick,
        horizon_variance_ticks2=classic_parameters.horizon_variance_ticks2,
        signal_strength=1.0,
        max_inventory_lots=classic_parameters.max_inventory_lots,
        quote_horizon_snapshots=classic_parameters.quote_horizon_snapshots,
    )
    deployed_parameters: ASParameters | ASDriftParameters = (
        drift_parameters if drift_enabled else classic_parameters
    )
    parameters = {
        "classic_as": classic_parameters,
        "as_drift_unit": unit_parameters,
        "as_drift": drift_parameters,
        "paired_as_no_drift": paired_parameters,
        "as_drift_kill_switch": deployed_parameters,
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
        comparisons[fill_mode] = {}
        for strategy in ("as_drift_unit", "as_drift", "as_drift_kill_switch"):
            for baseline in ("classic_as", "paired_as_no_drift"):
                comparisons[fill_mode][f"{strategy}_minus_{baseline}_hkd"] = float(
                    results[strategy][fill_mode]["net_pnl_hkd"]
                ) - float(results[baseline][fill_mode]["net_pnl_hkd"])
    validation_improvement = float(drift_tuning["best"]["score"]) - float(
        classic_tuning["best"]["score"]
    )
    return {
        "test_date": day.date,
        "model_train_dates": training_dates,
        "parameter_validation_dates": drift_tuning["validation_dates"],
        "classification": classification,
        "calibrator": asdict(calibrator),
        "quote_opportunities": int(len(quote_indices)),
        "parameters": {name: asdict(value) for name, value in parameters.items()},
        "activation": {
            "enabled": drift_enabled,
            "validation_score_improvement_hkd_equivalent": validation_improvement,
            "rule": "deploy AS+drift only when its prior-period conservative objective beats independently tuned classic AS",
        },
        "tuning": {
            "classic_as": compact_tuning(classic_tuning),
            "as_drift": compact_tuning(drift_tuning),
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
            gains = sum(value for value in daily_net if value > 0)
            losses = -sum(value for value in daily_net if value < 0)
            output[fill_mode][strategy] = {
                "test_days": len(rows),
                "profitable_days": sum(value > 0 for value in daily_net),
                "gross_pnl_hkd": gross,
                "net_pnl_official_only_hkd": gross - official_fees,
                "net_pnl_hkd": gross - all_in_fees,
                "all_in_fees_hkd": all_in_fees,
                "feeable_turnover_hkd": turnover,
                "net_bps_of_feeable_turnover": (
                    (gross - all_in_fees) / turnover * 10_000 if turnover else None
                ),
                "maker_fills": sum(int(row["maker_fills"]) for row in rows),
                "mean_daily_net_pnl_hkd": float(np.mean(daily_net)),
                "median_daily_net_pnl_hkd": float(np.median(daily_net)),
                "daily_profit_factor": gains / losses if losses else None,
                "day_close_max_drawdown_hkd": daily_max_drawdown(daily_net),
                "worst_intraday_drawdown_hkd": max(
                    float(row["max_drawdown_hkd"]) for row in rows
                ),
            }
        for strategy in ("as_drift_unit", "as_drift", "as_drift_kill_switch"):
            output[fill_mode][strategy]["incremental_net_pnl_hkd"] = {
                baseline: output[fill_mode][strategy]["net_pnl_hkd"]
                - output[fill_mode][baseline]["net_pnl_hkd"]
                for baseline in ("classic_as", "paired_as_no_drift")
            }
            output[fill_mode][strategy]["beats_baseline_days"] = {
                baseline: sum(
                    test["comparisons"][fill_mode][
                        f"{strategy}_minus_{baseline}_hkd"
                    ]
                    > 0
                    for test in tests
                )
                for baseline in ("classic_as", "paired_as_no_drift")
            }
    output["activation"] = {
        "enabled_days": [
            test["test_date"] for test in tests if test["activation"]["enabled"]
        ],
        "disabled_days": [
            test["test_date"] for test in tests if not test["activation"]["enabled"]
        ],
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
    return output


def validate(
    output: dict[str, Any], prior_result: dict[str, Any]
) -> dict[str, bool]:
    chronological = all(
        all(date < test["test_date"] for date in test["model_train_dates"])
        and all(
            date < test["test_date"]
            for date in test["parameter_validation_dates"]
        )
        for test in output["tests"]
    )
    accounting = all(
        np.isclose(
            row["gross_pnl_hkd"] - row["fees_hkd"], row["net_pnl_hkd"]
        )
        and int(row["maker_fills"]) == int(row["buy_fills"]) + int(row["sell_fills"])
        for test in output["tests"]
        for strategy in STRATEGIES
        for fill_mode in FILL_MODES
        for row in [test["strategies"][strategy][fill_mode]]
    )
    zero_drift_parameters = all(
        test["parameters"]["paired_as_no_drift"]["signal_strength"] == 0.0
        and all(
            test["parameters"]["paired_as_no_drift"][key]
            == test["parameters"]["as_drift"][key]
            for key in (
                "risk_aversion_per_tick",
                "order_decay_per_tick",
                "horizon_variance_ticks2",
                "max_inventory_lots",
                "quote_horizon_snapshots",
            )
        )
        for test in output["tests"]
    )
    unit_drift_parameters = all(
        test["parameters"]["as_drift_unit"]["signal_strength"] == 1.0
        and all(
            test["parameters"]["as_drift_unit"][key]
            == test["parameters"]["classic_as"][key]
            for key in (
                "risk_aversion_per_tick",
                "order_decay_per_tick",
                "horizon_variance_ticks2",
                "max_inventory_lots",
                "quote_horizon_snapshots",
            )
        )
        for test in output["tests"]
    )
    activation = all(
        test["activation"]["enabled"]
        == (test["activation"]["validation_score_improvement_hkd_equivalent"] > 0)
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
    classic_reproduced = all(
        np.isclose(
            output["aggregate"][fill_mode]["classic_as"]["net_pnl_hkd"],
            prior_result["aggregate"][fill_mode]["classic_as"]["net_pnl_hkd"],
        )
        for fill_mode in FILL_MODES
    )
    checks = {
        "chronological_no_future_information": chronological,
        "fee_and_fill_accounting": accounting,
        "paired_zero_drift_parameters_match": zero_drift_parameters,
        "unit_drift_uses_classic_as_parameters": unit_drift_parameters,
        "kill_switch_matches_validation_objective": activation,
        "monthly_aggregate_recomputed": aggregation,
        "classic_as_reproduces_prior_monthly_result": classic_reproduced,
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
    prior_path = OUTPUT_DIR / f"{monthly_output_stem(policy, 'as_gp_drift')}_results.json"
    prior_result = json.loads(prior_path.read_text(encoding="utf-8"))
    clean_dates = load_clean_dates(prior_path)
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
    validation_records = [
        validation_record(
            first_day,
            trades[first_day.date],
            first_validation,
            initial_score,
        )
    ]
    tests: list[dict[str, Any]] = []

    for position, test_date in enumerate(clean_dates[1:], start=1):
        training_dates = clean_dates[max(0, position - MODEL_TRAIN_DAYS) : position]
        training_days = [days[date] for date in training_dates]
        estimator, alpha = fit_direction_classifier(
            training_days, [day.eligible for day in training_days]
        )
        test_day = days[test_date]
        score, classification = predict_direction(
            estimator, alpha, test_day, test_day.eligible
        )
        recent_records = validation_records[-PARAMETER_VALIDATION_PERIODS:]
        calibrator = pooled_calibrator(recent_records)
        classic_parameters, classic_tuning = tune_model_parameters(
            recent_records, calibrator, "classic_as"
        )
        drift_parameters, drift_tuning = tune_as_drift(
            recent_records, calibrator
        )
        drift_enabled = float(drift_tuning["best"]["score"]) > float(
            classic_tuning["best"]["score"]
        )
        result = evaluate_day(
            test_day,
            trades[test_date],
            score,
            classification,
            calibrator,
            classic_parameters,
            drift_parameters,
            drift_enabled,
            classic_tuning,
            drift_tuning,
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
            f"alpha={'ON' if drift_enabled else 'OFF'}",
            f"AS={through['classic_as']['through']['net_pnl_hkd']:.2f}",
            f"AS_DRIFT={through['as_drift']['through']['net_pnl_hkd']:.2f}",
            f"SWITCH={through['as_drift_kill_switch']['through']['net_pnl_hkd']:.2f}",
            flush=True,
        )

    aggregate_result = aggregate(tests)
    output: dict[str, Any] = {
        "as_of": "2026-08-31",
        "instrument": "HK 07709",
        "question": "Does calibrated direction drift improve the classic Avellaneda-Stoikov market maker?",
        "design": {
            "reconstruction_policy": policy,
            "formula_ticks": {
                "reservation_price": "mid + eta * calibrated_alpha - inventory_lots * gamma * horizon_variance",
                "total_spread": "gamma * horizon_variance + 2/gamma * log(1 + gamma/order_decay)",
            },
            "direction_model_rolling_train_days": MODEL_TRAIN_DAYS,
            "parameter_validation_rolling_periods": PARAMETER_VALIDATION_PERIODS,
            "joint_tuning": {
                "risk_aversion_per_tick": list(GAMMAS),
                "signal_strength": list(SIGNAL_STRENGTHS),
                "max_inventory_lots": list(INVENTORY_CAPS),
                "objective": "through-fill all-in net PnL - 0.25 * max drawdown",
            },
            "models": {
                "classic_as": "independently tuned classic AS",
                "as_drift_unit": "classic AS parameters with the calibrated expected move added once (eta=1)",
                "as_drift": "jointly tuned AS risk aversion, inventory cap, and drift strength",
                "paired_as_no_drift": "same AS+drift parameters with eta forced to zero",
                "as_drift_kill_switch": "deploy AS+drift only when its prior validation objective beats classic AS",
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
                "through": "conservative: later trade at least one tick beyond quote",
                "touch": "optimistic: later trade reaches quote",
            },
        },
        "initial_calibration": {
            "date": first_day.date,
            "classification": initial_metrics,
        },
        "test_dates": [test["test_date"] for test in tests],
        "tests": tests,
        "aggregate": aggregate_result,
        "limitations": [
            (
                "Tolerant MBO ignores unknown modify/delete events after a fixed five-minute post-gap quarantine; exact depth recovery cannot be proven without an absolute snapshot."
                if policy == "tolerant"
                else "Only ten full-day out-of-sample tests pass strict MBO quality checks."
            ),
            "Touch/through are queue-position sensitivities, not exact fills.",
            "The alpha model and candidate grids were developed before this extension, so this is not a pristine strategy-discovery test.",
            "Account return is undefined without capital, margin, borrow, and minimum-commission assumptions.",
        ],
    }
    output["validation"] = validate(output, prior_result)
    stem = monthly_output_stem(policy, "as_drift")
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
