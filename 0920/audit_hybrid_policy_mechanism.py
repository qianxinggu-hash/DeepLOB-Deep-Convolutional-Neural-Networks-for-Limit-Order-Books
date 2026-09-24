#!/usr/bin/env python3
"""Re-solve selected July GP policy grids to audit the drift mechanism.

This is a policy-only diagnostic.  The gamma=0.1 surface is a controlled
counterfactual to show that the solver responds to drift; it is not a parameter
selection or a monthly trading result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0906"))
sys.path.insert(0, str(HERE))

from gp_qvi_model import (  # noqa: E402
    FILL_MODES,
    SESSION_BUCKETS,
    GPConfig,
    aggregate_calibration_stats,
    collect_daily_calibration_stats,
    infer_tick_hkd,
    make_gp_drift_regime,
    solve_gp_drift_policy,
    solve_gp_policy,
)
from run_gp_qvi_experiment import _load_inputs  # noqa: E402


def compare_grid(inputs, tick, mode, config, regime, buckets):
    counts = {"bid": 0, "ask": 0, "impulse": 0}
    cells = 0
    baseline_nonzero_q_cells = 0
    drift_nonzero_q_cells = 0
    baseline_not_flattened = 0
    drift_not_flattened = 0
    max_value_difference_hkd = 0.0
    controls_enabled = True
    inventories = np.arange(-config.max_inventory_lots, config.max_inventory_lots + 1)
    nonzero = inventories != 0
    for bucket in buckets:
        baseline = solve_gp_policy(inputs, tick, mode, bucket, config)
        drift = solve_gp_drift_policy(inputs, tick, mode, bucket, config, regime)
        controls_enabled &= baseline.market_orders_enabled and drift.market_orders_enabled
        for label, attr in (
            ("bid", "bid_action"),
            ("ask", "ask_action"),
            ("impulse", "impulse_target"),
        ):
            baseline_grid = getattr(baseline, attr)[:, None, :, :]
            drift_grid = getattr(drift, attr)
            counts[label] += int(np.count_nonzero(drift_grid != baseline_grid))
        cells += int(drift.bid_action.size)
        baseline_targets = baseline.impulse_target[1:, :, nonzero] - config.max_inventory_lots
        drift_targets = drift.impulse_target[1:, :, :, nonzero] - config.max_inventory_lots
        baseline_nonzero_q_cells += int(baseline_targets.size)
        drift_nonzero_q_cells += int(drift_targets.size)
        baseline_not_flattened += int(np.count_nonzero(baseline_targets))
        drift_not_flattened += int(np.count_nonzero(drift_targets))
        max_value_difference_hkd = max(
            max_value_difference_hkd,
            float(np.max(np.abs(drift.values - baseline.values[:, None, :, :]))),
        )
    return {
        "bucket_indices": list(buckets),
        "market_orders_enabled_in_both": bool(controls_enabled),
        "full_grid_state_cells": cells,
        "different_bid_actions": counts["bid"],
        "different_ask_actions": counts["ask"],
        "different_impulse_targets": counts["impulse"],
        "max_abs_value_difference_hkd": max_value_difference_hkd,
        "baseline_nonzero_inventory_cells_excluding_terminal": baseline_nonzero_q_cells,
        "baseline_not_immediately_flattened_to_zero": baseline_not_flattened,
        "drift_nonzero_inventory_cells_excluding_terminal": drift_nonzero_q_cells,
        "drift_not_immediately_flattened_to_zero": drift_not_flattened,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--month-output", type=Path,
        default=HERE / "output/hybrid_month",
    )
    args = parser.parse_args()
    source, dates, days, trades = _load_inputs()
    base_config = GPConfig(max_inventory_lots=10, inventory_penalty_gamma=5.0)
    output = {
        "description": "Full GP policy-grid audit at fixed gamma=5; gamma=0.1 is a mechanism-only counterfactual, not parameter selection or trading PnL",
        "source_month_result": str(args.month_output / "results.json"),
        "grid_axes": "time step 0..100, drift state 0..4, spread state 0..5, inventory -10..10; each baseline cell is broadcast across all five drift states",
        "nonzero_inventory_flatten_audit_excludes_terminal_step_zero": True,
        "dates": {},
        "gamma_0_1_policy_only_counterfactual": {},
    }
    for date in ("2026-07-03", "2026-07-17"):
        saved = json.loads((args.month_output / f"day_{date}.json").read_text())
        prior = saved["calibration_dates"]
        inputs = aggregate_calibration_stats(
            [collect_daily_calibration_stats(days[d], trades[d], base_config) for d in prior],
            base_config,
        )
        drift_json = saved["drift_regime"]
        regime = make_gp_drift_regime(
            drift_json["drift_hkd_per_second"],
            np.asarray(drift_json["transition_probabilities"]),
            drift_json["labels"],
        )
        tick = infer_tick_hkd(days[date])
        day_result = {
            "calibration_dates": prior,
            "gamma": 5.0,
            "max_inventory_lots": base_config.max_inventory_lots,
            "tick_hkd": tick,
            "modes": {},
        }
        for mode in FILL_MODES:
            diagnostic = compare_grid(
                inputs, tick, mode, base_config, regime,
                range(len(SESSION_BUCKETS)),
            )
            visited = saved["counterfactual_action_audit_at_hybrid_visited_states"][mode]
            diagnostic["actual_hybrid_visited_quotes"] = visited["visited_quotes"]
            diagnostic["different_actions_at_hybrid_visited_states"] = visited[
                "any_changed_count"
            ]
            day_result["modes"][mode] = diagnostic
        output["dates"][date] = day_result
        if date == "2026-07-17":
            low_config = GPConfig(max_inventory_lots=10, inventory_penalty_gamma=0.1)
            output["gamma_0_1_policy_only_counterfactual"] = {
                "date": date,
                "gamma": 0.1,
                "calibration_dates": prior,
                "bucket_indices": [0],
                "modes": {
                    mode: compare_grid(
                        inputs, tick, mode, low_config, regime, [0]
                    )
                    for mode in FILL_MODES
                },
            }
        print("AUDITED", date, flush=True)
    destination = args.month_output / "policy_mechanism_audit.json"
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print("SAVED", destination.resolve(), flush=True)


if __name__ == "__main__":
    main()
