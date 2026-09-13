#!/usr/bin/env python3
"""Reprice the July 2026 market-making results across fee assumptions.

The fill path, quotes, inventory, and gross PnL are held fixed.  Only the fee
rate applied to feeable turnover changes, so this is a fee sensitivity
experiment rather than a new execution simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OFFICIAL_FIXED_BPS_PER_SIDE = 1.27
CURRENT_BROKERAGE_BPS_PER_SIDE = 1.50
CURRENT_ALL_IN_BPS_PER_SIDE = 2.77
FILL_MODES = ("through", "touch")
FEE_SCENARIOS_BPS_PER_SIDE = (
    0.00,
    1.27,
    1.50,
    1.75,
    1.90,
    2.00,
    2.25,
    2.50,
    2.77,
    3.00,
    3.25,
)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    label_zh: str
    source: Path
    source_strategy: str


MODEL_SPECS = (
    ModelSpec(
        "classic_as",
        "经典 AS",
        ROOT / "0824/output/july_2026_tolerant_as_drift_results.json",
        "classic_as",
    ),
    ModelSpec(
        "as_drift_unit",
        "AS + unit drift",
        ROOT / "0824/output/july_2026_tolerant_as_drift_results.json",
        "as_drift_unit",
    ),
    ModelSpec(
        "as_drift_tuned",
        "AS + tuned drift",
        ROOT / "0824/output/july_2026_tolerant_as_drift_results.json",
        "as_drift",
    ),
    ModelSpec(
        "as_drift_kill_switch",
        "AS + drift kill switch",
        ROOT / "0824/output/july_2026_tolerant_as_drift_results.json",
        "as_drift_kill_switch",
    ),
    ModelSpec(
        "gp_style_discrete",
        "GP-style",
        ROOT / "0824/output/july_2026_tolerant_as_gp_drift_results.json",
        "gp_style_discrete",
    ),
    ModelSpec(
        "forced_drift_inventory",
        "强制 drift + 库存",
        ROOT / "0824/output/july_2026_tolerant_as_gp_drift_results.json",
        "directional",
    ),
    ModelSpec(
        "original_drift_inventory_kill_switch",
        "原始 drift + 库存 + kill switch",
        ROOT
        / "0824/output/july_2026_tolerant_directional_kill_switch_results.json",
        "directional_kill_switch",
    ),
)


def net_pnl_at_fee(
    gross_pnl_hkd: float, feeable_turnover_hkd: float, fee_bps_per_side: float
) -> float:
    """Return net PnL when the same per-side rate applies to all turnover."""
    return gross_pnl_hkd - feeable_turnover_hkd * fee_bps_per_side / 10_000.0


def break_even_fee_bps(
    gross_pnl_hkd: float, feeable_turnover_hkd: float
) -> float:
    """Return the per-execution-side all-in rate at which net PnL is zero."""
    if feeable_turnover_hkd <= 0:
        raise ValueError("feeable turnover must be positive")
    return gross_pnl_hkd / feeable_turnover_hkd * 10_000.0


def profitable_days_at_fee(
    daily_rows: list[dict[str, Any]], fee_bps_per_side: float
) -> int:
    return sum(
        net_pnl_at_fee(
            float(row["gross_pnl_hkd"]),
            float(row["feeable_turnover_hkd"]),
            fee_bps_per_side,
        )
        > 0.0
        for row in daily_rows
    )


def load_sources() -> dict[Path, dict[str, Any]]:
    sources: dict[Path, dict[str, Any]] = {}
    for spec in MODEL_SPECS:
        if spec.source not in sources:
            sources[spec.source] = json.loads(spec.source.read_text(encoding="utf-8"))
    return sources


def build_experiment() -> dict[str, Any]:
    sources = load_sources()
    summaries: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        source = sources[spec.source]
        for fill_mode in FILL_MODES:
            aggregate = source["aggregate"][fill_mode][spec.source_strategy]
            daily_rows = [
                test["strategies"][spec.source_strategy][fill_mode]
                for test in source["tests"]
            ]
            gross = float(aggregate["gross_pnl_hkd"])
            turnover = float(aggregate["feeable_turnover_hkd"])
            break_even = break_even_fee_bps(gross, turnover)
            max_brokerage = break_even - OFFICIAL_FIXED_BPS_PER_SIDE
            current_net = net_pnl_at_fee(gross, turnover, CURRENT_ALL_IN_BPS_PER_SIDE)

            summaries.append(
                {
                    "model": spec.model,
                    "model_label_zh": spec.label_zh,
                    "fill_mode": fill_mode,
                    "test_days": len(daily_rows),
                    "gross_pnl_hkd": gross,
                    "feeable_turnover_hkd": turnover,
                    "current_all_in_fee_bps_per_side": CURRENT_ALL_IN_BPS_PER_SIDE,
                    "current_net_pnl_hkd": current_net,
                    "break_even_all_in_fee_bps_per_side": break_even,
                    "break_even_all_in_fee_percent_per_side": break_even / 100.0,
                    "break_even_round_trip_bps": break_even * 2.0,
                    "max_brokerage_bps_per_side_after_official_fees": max_brokerage,
                    "max_brokerage_percent_per_side_after_official_fees": max_brokerage
                    / 100.0,
                    "profitable_at_current_fee": current_net > 0.0,
                    "profitable_with_nonnegative_all_in_fee": break_even > 0.0,
                    "profitable_with_official_fees_only": break_even
                    > OFFICIAL_FIXED_BPS_PER_SIDE,
                    "required_all_in_fee_reduction_bps_per_side": max(
                        0.0, CURRENT_ALL_IN_BPS_PER_SIDE - break_even
                    ),
                    "required_brokerage_reduction_percent": (
                        max(0.0, CURRENT_BROKERAGE_BPS_PER_SIDE - max_brokerage)
                        / CURRENT_BROKERAGE_BPS_PER_SIDE
                        * 100.0
                    ),
                    "profit_condition": "all_in_fee_bps_per_side < break_even_all_in_fee_bps_per_side",
                    "source": str(spec.source.relative_to(ROOT)),
                }
            )

            for fee_bps in FEE_SCENARIOS_BPS_PER_SIDE:
                scenarios.append(
                    {
                        "model": spec.model,
                        "model_label_zh": spec.label_zh,
                        "fill_mode": fill_mode,
                        "all_in_fee_bps_per_side": fee_bps,
                        "brokerage_bps_per_side_after_official_fees": fee_bps
                        - OFFICIAL_FIXED_BPS_PER_SIDE,
                        "net_pnl_hkd": net_pnl_at_fee(gross, turnover, fee_bps),
                        "profitable_days": profitable_days_at_fee(daily_rows, fee_bps),
                        "test_days": len(daily_rows),
                    }
                )

    validations: dict[str, bool] = {}
    validations["source_fee_assumptions_match"] = all(
        abs(
            float(source["design"]["cost_bps_per_execution_side"]["official_fixed"])
            - OFFICIAL_FIXED_BPS_PER_SIDE
        )
        < 1e-12
        and abs(
            float(source["design"]["cost_bps_per_execution_side"]["all_in"])
            - CURRENT_ALL_IN_BPS_PER_SIDE
        )
        < 1e-12
        for source in sources.values()
    )
    validations["current_net_pnl_reproduced"] = all(
        abs(
            row["current_net_pnl_hkd"]
            - float(
                sources[ROOT / row["source"]]["aggregate"][row["fill_mode"]][
                    next(
                        spec.source_strategy
                        for spec in MODEL_SPECS
                        if spec.model == row["model"]
                    )
                ]["net_pnl_hkd"]
            )
        )
        < 1e-6
        for row in summaries
    )
    validations["break_even_net_pnl_is_zero"] = all(
        abs(
            net_pnl_at_fee(
                row["gross_pnl_hkd"],
                row["feeable_turnover_hkd"],
                row["break_even_all_in_fee_bps_per_side"],
            )
        )
        < 1e-8
        for row in summaries
    )
    validations["scenario_current_fee_matches_summary"] = all(
        abs(
            next(
                scenario["net_pnl_hkd"]
                for scenario in scenarios
                if scenario["model"] == row["model"]
                and scenario["fill_mode"] == row["fill_mode"]
                and scenario["all_in_fee_bps_per_side"]
                == CURRENT_ALL_IN_BPS_PER_SIDE
            )
            - row["current_net_pnl_hkd"]
        )
        < 1e-8
        for row in summaries
    )
    validations["all_passed"] = all(validations.values())

    return {
        "as_of": "2026-08-31",
        "instrument": "HK 07709",
        "question": "How low must transaction fees be for the July 2026 strategies to be profitable?",
        "method": {
            "held_constant": [
                "quotes",
                "fills",
                "inventory path",
                "terminal flattening",
                "gross PnL",
                "feeable turnover",
            ],
            "varied": "all-in fee in bps per execution side",
            "net_pnl_formula": "gross_pnl_hkd - feeable_turnover_hkd * fee_bps_per_side / 10000",
            "break_even_formula": "gross_pnl_hkd / feeable_turnover_hkd * 10000",
            "profit_condition": "fee must be strictly below the break-even rate",
        },
        "fee_assumptions_bps_per_execution_side": {
            "official_fixed": OFFICIAL_FIXED_BPS_PER_SIDE,
            "brokerage": CURRENT_BROKERAGE_BPS_PER_SIDE,
            "all_in": CURRENT_ALL_IN_BPS_PER_SIDE,
        },
        "summary": summaries,
        "scenarios": scenarios,
        "validation": validations,
        "limitations": [
            "This reprices fixed historical fills; it does not model changes in quoting or fills caused by a different fee schedule.",
            "A zero net PnL threshold is not a safety margin and does not imply statistical robustness.",
            "The original touch/through queue-position assumptions and all source-data limitations remain unchanged.",
            "Broker minimum commissions, rebates, market impact, borrow costs, and account-specific fee tiers remain unavailable.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = build_experiment()
    result_path = args.output_dir / "7709_202607_fee_break_even_results.json"
    summary_path = args.output_dir / "7709_202607_fee_break_even_summary.csv"
    scenario_path = args.output_dir / "7709_202607_fee_sensitivity.csv"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(summary_path, result["summary"])
    write_csv(scenario_path, result["scenarios"])

    classic = {
        row["fill_mode"]: row
        for row in result["summary"]
        if row["model"] == "classic_as"
    }
    print(
        "Classic AS break-even all-in fee (bps/execution side): "
        f"through={classic['through']['break_even_all_in_fee_bps_per_side']:.4f}, "
        f"touch={classic['touch']['break_even_all_in_fee_bps_per_side']:.4f}"
    )
    print(
        "Max brokerage after 1.27 bps official fees: "
        f"through={classic['through']['max_brokerage_bps_per_side_after_official_fees']:.4f}, "
        f"touch={classic['touch']['max_brokerage_bps_per_side_after_official_fees']:.4f}"
    )
    print("WROTE", result_path, summary_path, scenario_path)
    if not result["validation"]["all_passed"]:
        raise AssertionError(result["validation"])


if __name__ == "__main__":
    main()
