#!/usr/bin/env python3
"""Independent integrity checks for the multi-day strategy artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return bool(abs(left - right) <= tolerance)


def main() -> None:
    result_path = HERE / "output" / "multiday_strategy_results.json"
    trade_path = HERE / "output" / "multiday_strategy_trades.csv"
    notebook_path = HERE / "multiday_strategy_analysis.ipynb"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(trade_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    components = results["method"]["official_fixed_cost_components_per_side_percent"]
    calculated_round_trip_bps = 2 * sum(components.values()) * 100
    checks["official_fixed_cost_math"] = close(calculated_round_trip_bps, 2.54)

    classification_ok = True
    strategy_math_ok = True
    strategy_count_ok = True
    for experiments in results["experiments"].values():
        for experiment in experiments:
            metric = experiment["classification"]
            confusion = np.asarray(metric["confusion_matrix"], dtype=np.int64)
            classification_ok &= int(confusion.sum()) == int(metric["samples"])
            classification_ok &= close(
                float(np.trace(confusion) / confusion.sum()), float(metric["accuracy"])
            )
            horizon = int(experiment.get("holding_horizon", 20))
            for key, strategy in experiment["strategies"].items():
                mode, threshold_text = key.rsplit("_p", 1)
                threshold = float(threshold_text)
                selected = trades[
                    (trades["split"] == experiment["name"])
                    & (trades["mode"] == mode)
                    & np.isclose(trades["confidence_threshold"], threshold)
                    & (trades["holding_horizon"] == horizon)
                ]
                strategy_count_ok &= len(selected) == int(strategy["trades"])
                gross_mean = strategy["gross_mean_bps_after_spread"]
                if gross_mean is not None:
                    strategy_math_ok &= close(
                        float(selected["gross_bps_after_spread"].mean()),
                        float(gross_mean),
                        1e-7,
                    )
                    for cost_text, scenario in strategy["cost_scenarios"].items():
                        expected = float(gross_mean) - float(cost_text)
                        strategy_math_ok &= close(
                            expected, float(scenario["mean_net_bps"]), 1e-7
                        )
    checks["classification_recomputed"] = classification_ok
    checks["strategy_counts_match_trade_log"] = strategy_count_ok
    checks["strategy_cost_math"] = strategy_math_ok

    long_rows = trades["side"] == "long"
    recomputed = np.empty(len(trades), dtype=np.float64)
    recomputed[long_rows] = (
        trades.loc[long_rows, "exit_price_hkd"].to_numpy()
        / trades.loc[long_rows, "entry_price_hkd"].to_numpy()
        - 1.0
    ) * 10_000
    recomputed[~long_rows] = (
        trades.loc[~long_rows, "entry_price_hkd"].to_numpy()
        / trades.loc[~long_rows, "exit_price_hkd"].to_numpy()
        - 1.0
    ) * 10_000
    checks["trade_returns_recomputed"] = bool(
        np.allclose(recomputed, trades["gross_bps_after_spread"], atol=1e-7)
    )
    checks["execution_times_increase"] = bool(
        (
            (trades["signal_time"] < trades["entry_time"])
            & (trades["entry_time"] < trades["exit_time"])
        ).all()
    )

    non_overlap = True
    grouping = [
        "split",
        "date",
        "mode",
        "confidence_threshold",
        "holding_horizon",
    ]
    for _, frame in trades.sort_values("signal_time").groupby(grouping, dropna=False):
        if len(frame) > 1:
            non_overlap &= bool(
                (frame["signal_time"].to_numpy()[1:] > frame["exit_time"].to_numpy()[:-1]).all()
            )
    checks["positions_do_not_overlap"] = non_overlap

    quality = {
        (row["family"], row["date"]): row for row in results["data_quality"]
    }
    checks["mbo_complete_days_untainted"] = all(
        not quality[("strict_mbo", date)]["final_state_tainted"]
        and not quality[("strict_mbo", date)]["order_errors"]
        for date in ("2026-07-09", "2026-07-21", "2026-08-07")
    )
    checks["mbo_0804_correctly_quarantined"] = bool(
        quality[("strict_mbo", "2026-08-04")]["final_state_tainted"]
        and quality[("strict_mbo", "2026-08-04")]["order_errors"]
    )

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    checks["notebook_executed_without_cell_errors"] = bool(
        code_cells
        and all(cell["execution_count"] is not None for cell in code_cells)
        and not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )
    )

    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "trade_rows_checked": int(len(trades)),
        "classification_experiments_checked": int(
            sum(len(group) for group in results["experiments"].values())
        ),
    }
    destination = HERE / "output" / "multiday_strategy_validation.json"
    destination.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
