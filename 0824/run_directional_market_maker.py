#!/usr/bin/env python3
"""Walk-forward evaluation of a direction-enhanced AS/GP market maker."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from directional_market_maker import (
    AlphaCalibrator,
    QuoteParameters,
    fit_alpha_calibrator,
    fit_direction_classifier,
    future_mid_move_ticks,
    load_or_extract_trades,
    predict_direction,
    select_quote_indices,
    simulate_market_maker,
    tuning_score,
)
from run_multiday_strategy import HORIZON, RAW, load_day, per_day_split


HERE = Path(__file__).resolve().parent
DATES = ("2026-07-09", "2026-07-21", "2026-08-07")


def subset_signal(
    eligible: np.ndarray,
    values: np.ndarray,
    quote_indices: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(eligible, quote_indices)
    if np.any(positions >= len(eligible)) or not np.array_equal(eligible[positions], quote_indices):
        raise ValueError("quote indices are not a subset of eligible indices")
    return values[positions]


def tune_parameters(
    day: object,
    trades: object,
    eligible: np.ndarray,
    calibrated_alpha: np.ndarray,
    directional: bool,
) -> tuple[QuoteParameters, dict[str, object]]:
    quote_indices = select_quote_indices(day, eligible, HORIZON)
    quote_alpha = subset_signal(eligible, calibrated_alpha, quote_indices)
    strengths = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) if directional else (0.0,)
    candidates: list[tuple[float, QuoteParameters, dict[str, object]]] = []
    for base_distance in (0, 1, 2):
        for strength in strengths:
            for inventory_skew in (0.5, 1.0, 2.0):
                for max_inventory in (3, 5):
                    parameters = QuoteParameters(
                        base_distance_ticks=base_distance,
                        signal_strength=strength,
                        inventory_skew_ticks=inventory_skew,
                        max_inventory_lots=max_inventory,
                    )
                    result, _ = simulate_market_maker(
                        day,
                        trades,
                        quote_indices,
                        quote_alpha,
                        parameters,
                        fill_mode="through",
                        strategy_name="directional" if directional else "traditional_baseline",
                    )
                    candidates.append((tuning_score(result), parameters, result))
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, parameters, result = candidates[0]
    audit = {
        "objective": "net_pnl_hkd - 0.25 * max_drawdown_hkd; require at least 20 conservative fills",
        "fill_assumption": "through: subsequent trade must be at least one tick beyond quote",
        "candidate_count": len(candidates),
        "best_score": score,
        "best_validation_result": result,
        "top_five": [
            {
                "score": item[0],
                "parameters": asdict(item[1]),
                "net_pnl_hkd": item[2]["net_pnl_hkd"],
                "maker_fills": item[2]["maker_fills"],
                "max_drawdown_hkd": item[2]["max_drawdown_hkd"],
            }
            for item in candidates[:5]
        ],
    }
    return parameters, audit


def evaluate_strategies(
    day: object,
    trades: object,
    eligible: np.ndarray,
    calibrated_alpha: np.ndarray,
    baseline_parameters: QuoteParameters,
    directional_parameters: QuoteParameters,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    quote_indices = select_quote_indices(day, eligible, HORIZON)
    quote_alpha = subset_signal(eligible, calibrated_alpha, quote_indices)
    strategies = {
        "traditional_baseline": baseline_parameters,
        "directional": directional_parameters,
        "paired_no_signal_ablation": replace(directional_parameters, signal_strength=0.0),
    }
    results: dict[str, object] = {}
    events: list[dict[str, object]] = []
    for name, parameters in strategies.items():
        results[name] = {}
        for fill_mode in ("through", "touch"):
            result, rows = simulate_market_maker(
                day,
                trades,
                quote_indices,
                quote_alpha,
                parameters,
                fill_mode=fill_mode,
                strategy_name=name,
                record_events=True,
            )
            results[name][fill_mode] = result
            events.extend(rows)
    comparisons: dict[str, object] = {}
    for fill_mode in ("through", "touch"):
        directional_net = float(results["directional"][fill_mode]["net_pnl_hkd"])
        comparisons[fill_mode] = {
            "directional_minus_traditional_hkd": directional_net
            - float(results["traditional_baseline"][fill_mode]["net_pnl_hkd"]),
            "directional_minus_paired_ablation_hkd": directional_net
            - float(results["paired_no_signal_ablation"][fill_mode]["net_pnl_hkd"]),
            "directional_profitable": directional_net > 0,
            "directional_beats_both_no_signal_controls": directional_net
            > max(
                float(results["traditional_baseline"][fill_mode]["net_pnl_hkd"]),
                float(results["paired_no_signal_ablation"][fill_mode]["net_pnl_hkd"]),
            ),
        }
    return {
        "date": day.date,
        "quote_opportunities": int(len(quote_indices)),
        "strategies": results,
        "comparisons": comparisons,
    }, events


def activate_signal(
    baseline_parameters: QuoteParameters,
    directional_parameters: QuoteParameters,
    baseline_tuning: dict[str, object],
    directional_tuning: dict[str, object],
) -> tuple[QuoteParameters, dict[str, object]]:
    """Use alpha only when it beats the no-signal control on prior data."""

    improvement = float(directional_tuning["best_score"]) - float(
        baseline_tuning["best_score"]
    )
    enabled = improvement > 0.0
    deployed = directional_parameters if enabled else baseline_parameters
    return deployed, {
        "enabled": enabled,
        "validation_score_improvement_hkd_equivalent": improvement,
        "rule": "enable only when the best non-zero-alpha candidate beats the independently tuned no-signal baseline on the prior validation period",
        "rejected_candidate_parameters": None if enabled else asdict(directional_parameters),
    }


def calibrator_record(calibrator: AlphaCalibrator) -> dict[str, object]:
    return asdict(calibrator)


def main() -> None:
    days = {
        date: load_day(
            date,
            "strict_mbo",
            HERE / "data" / f"hk07709_{date}_mbo_strict_s10.npz",
        )
        for date in DATES
    }
    for day in days.values():
        if day.metadata.get("final_state_tainted"):
            raise RuntimeError(f"tainted MBO state cannot be used: {day.date}")

    trade_data = {}
    for date in DATES:
        print("TRADES", date, flush=True)
        trade_data[date] = load_or_extract_trades(
            RAW / f"hk07709_{date}.csv",
            HERE / "data" / f"hk07709_{date}_trades.npz",
        )

    day_0709 = days["2026-07-09"]
    fit_0709, validation_0709 = per_day_split(day_0709)
    first_estimator, first_alpha = fit_direction_classifier([day_0709], [fit_0709])
    validation_score, validation_metrics = predict_direction(
        first_estimator, first_alpha, day_0709, validation_0709
    )
    first_calibrator = fit_alpha_calibrator(
        validation_score, future_mid_move_ticks(day_0709, validation_0709, HORIZON)
    )
    validation_alpha = first_calibrator.transform(validation_score)
    baseline_parameters_1, baseline_tuning_1 = tune_parameters(
        day_0709,
        trade_data["2026-07-09"],
        validation_0709,
        validation_alpha,
        directional=False,
    )
    directional_parameters_1, directional_tuning_1 = tune_parameters(
        day_0709,
        trade_data["2026-07-09"],
        validation_0709,
        validation_alpha,
        directional=True,
    )
    deployed_parameters_1, activation_1 = activate_signal(
        baseline_parameters_1,
        directional_parameters_1,
        baseline_tuning_1,
        directional_tuning_1,
    )

    full_0709_estimator, full_0709_alpha = fit_direction_classifier(
        [day_0709], [day_0709.eligible]
    )
    day_0721 = days["2026-07-21"]
    score_0721, metrics_0721 = predict_direction(
        full_0709_estimator, full_0709_alpha, day_0721, day_0721.eligible
    )
    calibrated_0721 = first_calibrator.transform(score_0721)
    result_0721, events_0721 = evaluate_strategies(
        day_0721,
        trade_data["2026-07-21"],
        day_0721.eligible,
        calibrated_0721,
        baseline_parameters_1,
        deployed_parameters_1,
    )
    experiment_1 = {
        "name": "0709_validation_to_0721_test",
        "model_train_dates": ["2026-07-09"],
        "parameter_validation_date": "2026-07-09 (last 30%)",
        "test_date": "2026-07-21",
        "validation_classification": validation_metrics,
        "test_classification": metrics_0721,
        "calibrator": calibrator_record(first_calibrator),
        "baseline_tuning": baseline_tuning_1,
        "directional_tuning": directional_tuning_1,
        "signal_activation": activation_1,
        "test": result_0721,
    }
    print(
        "TEST 2026-07-21",
        f"accuracy={metrics_0721['accuracy']:.4f}",
        f"directional_through={result_0721['strategies']['directional']['through']['net_pnl_hkd']:.2f}",
        flush=True,
    )

    # The first truly out-of-sample day becomes the validation day for the next step.
    second_calibrator = fit_alpha_calibrator(
        score_0721, future_mid_move_ticks(day_0721, day_0721.eligible, HORIZON)
    )
    alpha_0721_for_tuning = second_calibrator.transform(score_0721)
    baseline_parameters_2, baseline_tuning_2 = tune_parameters(
        day_0721,
        trade_data["2026-07-21"],
        day_0721.eligible,
        alpha_0721_for_tuning,
        directional=False,
    )
    directional_parameters_2, directional_tuning_2 = tune_parameters(
        day_0721,
        trade_data["2026-07-21"],
        day_0721.eligible,
        alpha_0721_for_tuning,
        directional=True,
    )
    deployed_parameters_2, activation_2 = activate_signal(
        baseline_parameters_2,
        directional_parameters_2,
        baseline_tuning_2,
        directional_tuning_2,
    )
    full_estimator, full_alpha = fit_direction_classifier(
        [day_0709, day_0721], [day_0709.eligible, day_0721.eligible]
    )
    day_0807 = days["2026-08-07"]
    score_0807, metrics_0807 = predict_direction(
        full_estimator, full_alpha, day_0807, day_0807.eligible
    )
    calibrated_0807 = second_calibrator.transform(score_0807)
    result_0807, events_0807 = evaluate_strategies(
        day_0807,
        trade_data["2026-08-07"],
        day_0807.eligible,
        calibrated_0807,
        baseline_parameters_2,
        deployed_parameters_2,
    )
    experiment_2 = {
        "name": "0721_validation_to_0807_test",
        "model_train_dates": ["2026-07-09", "2026-07-21"],
        "parameter_validation_date": "2026-07-21 (out-of-sample predictions from 07-09 model)",
        "test_date": "2026-08-07",
        "validation_classification": metrics_0721,
        "test_classification": metrics_0807,
        "calibrator": calibrator_record(second_calibrator),
        "baseline_tuning": baseline_tuning_2,
        "directional_tuning": directional_tuning_2,
        "signal_activation": activation_2,
        "test": result_0807,
    }
    print(
        "TEST 2026-08-07",
        f"accuracy={metrics_0807['accuracy']:.4f}",
        f"directional_through={result_0807['strategies']['directional']['through']['net_pnl_hkd']:.2f}",
        flush=True,
    )

    experiments = [experiment_1, experiment_2]
    aggregate: dict[str, object] = {}
    for fill_mode in ("through", "touch"):
        aggregate[fill_mode] = {}
        for strategy in ("traditional_baseline", "directional", "paired_no_signal_ablation"):
            values = [
                float(item["test"]["strategies"][strategy][fill_mode]["net_pnl_hkd"])
                for item in experiments
            ]
            aggregate[fill_mode][strategy] = {
                "test_days": len(values),
                "profitable_days": sum(value > 0 for value in values),
                "total_net_pnl_hkd": sum(values),
                "mean_net_pnl_hkd": float(np.mean(values)),
                "total_gross_pnl_hkd": sum(
                    float(item["test"]["strategies"][strategy][fill_mode]["gross_pnl_hkd"])
                    for item in experiments
                ),
                "total_net_pnl_official_only_hkd": sum(
                    float(item["test"]["strategies"][strategy][fill_mode]["net_pnl_official_only_hkd"])
                    for item in experiments
                ),
            }
        directional = aggregate[fill_mode]["directional"]
        directional["beats_traditional_days"] = sum(
            item["test"]["comparisons"][fill_mode]["directional_minus_traditional_hkd"] > 0
            for item in experiments
        )
        directional["beats_paired_ablation_days"] = sum(
            item["test"]["comparisons"][fill_mode]["directional_minus_paired_ablation_hkd"] > 0
            for item in experiments
        )

    output = {
        "as_of": "2026-08-24",
        "instrument": "HK 07709",
        "design": {
            "model": "direction-enhanced discrete Avellaneda-Stoikov / Guilbaud-Pham-style market maker",
            "reservation_price": "mid + calibrated_expected_move - inventory_penalty",
            "signal": "multinomial logistic p(up)-p(down), calibrated on a prior validation period to future mid-price ticks",
            "signal_is_not_a_taker_rule": True,
            "quote_action": "move both bid and ask around the signal/inventory-shifted center; remain passive",
            "inventory_action": "stop adding to inventory at hard limit; flatten remaining inventory at end of test session",
            "alpha_kill_switch": "deploy non-zero signal strength only if its conservative validation objective beats the no-signal baseline",
            "quote_horizon_snapshots": HORIZON,
            "tick_hkd": 0.02,
            "lot_size": 100,
            "latency_ms": 5,
            "fill_bounds": {
                "through": "conservative queue proxy: a later trade is at least one tick beyond the quote",
                "touch": "optimistic queue proxy: a later trade reaches the quote",
            },
            "fee_assumption_bps_per_execution_side": {
                "official_fixed": 1.27,
                "brokerage_assumption": 1.50,
                "total": 2.77,
            },
        },
        "data_quality": {
            date: {
                "book_snapshots": int(len(days[date].book)),
                "mbo_state_tainted": bool(days[date].metadata.get("final_state_tainted")),
                "order_errors": days[date].metadata.get("order_errors", {}),
                "trade_prints": int(len(trade_data[date].price_hkd)),
            }
            for date in DATES
        },
        "experiments": experiments,
        "aggregate_test_results": aggregate,
        "limitations": [
            "OrderID-level reconstruction is strict, but the feed does not identify our hypothetical queue position.",
            "Touch and through are sensitivity bounds, not exact fill replay; cancellations ahead and latency races are unobserved.",
            "Only two walk-forward test days are available, so profitability cannot be inferred statistically.",
            "Broker minimum commission, exchange rebates, borrow constraints, market impact, and order acknowledgement latency are unavailable.",
            "The reported direction classifier is logistic, not the DeepLOB checkpoint, because cross-day DeepLOB probabilities were not available without retraining.",
        ],
        "primary_sources": [
            "https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf",
            "https://arxiv.org/abs/1106.5040",
            "https://arxiv.org/abs/1205.3051",
        ],
    }
    output_path = HERE / "output" / "directional_market_maker_results.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    events = events_0721 + events_0807
    event_path = HERE / "output" / "directional_market_maker_fills.csv"
    if events:
        with event_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(events[0]))
            writer.writeheader()
            writer.writerows(events)
    print("WROTE", output_path, event_path, flush=True)


if __name__ == "__main__":
    main()
