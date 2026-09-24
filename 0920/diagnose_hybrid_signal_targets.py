#!/usr/bin/env python3
"""Audit the Hybrid classification target against the GP drift markout.

This is a read-only diagnostic on the same July 2026 quote decisions as the
monthly Hybrid replay. It does not fit a model or change the trading policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0824"))
sys.path.insert(0, str(ROOT / "0906"))

from directional_market_maker import select_quote_indices  # noqa: E402
from gp_qvi_model import infer_tick_hkd  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def describe(a: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "quantiles_0_25_50_75_95_99_100": np.quantile(
            a, [0, .25, .5, .75, .95, .99, 1]
        ).tolist(),
    }


def main() -> None:
    monthly = json.loads((HERE / "output/hybrid_month/results.json").read_text())
    alpha = json.loads(
        (HERE / "output/hybrid_month/pre_july_training.json").read_text()
    )["labeling"]["alpha"]
    pieces: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "endpoint", "future_mean_from_now", "past_anchor", "equation4_ticks",
            "equation4_return", "model_history",
        )
    }
    daily = []
    for result in monthly["tests"]:
        date = result["date"]
        path = ROOT / f"0824/data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz"
        day = load_day(date, "tolerant_mbo", path)
        indices = select_quote_indices(day, day.eligible, 20)
        if len(indices) != result["signal"]["quotes"]:
            raise AssertionError(f"{date}: quote decisions differ from monthly replay")
        tick = infer_tick_hkd(day)
        mid = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2
        cumulative = np.r_[0.0, np.cumsum(mid)]
        k = 20
        past = (cumulative[indices + 1] - cumulative[indices - k + 1]) / k
        future = (cumulative[indices + k + 1] - cumulative[indices + 1]) / k
        endpoint = (mid[indices + k] - mid[indices]) / tick
        forward_mean = (future - mid[indices]) / tick
        past_anchor = (mid[indices] - past) / tick
        equation4 = forward_mean + past_anchor
        equation4_return = future / past - 1.0
        np.testing.assert_allclose(equation4_return, day.returns[indices], atol=1e-10)
        if not np.isclose(
            np.mean(endpoint),
            result["signal"]["actual_20_snapshot_mid_move_ticks"]["mean"],
            rtol=0, atol=2e-4,
        ):
            raise AssertionError(f"{date}: endpoint target differs from monthly replay")
        # The first quote of some sessions lacks the 100-lag auxiliary feature.
        segment_start = np.maximum.accumulate(
            np.where(
                np.r_[True, day.segments[1:] != day.segments[:-1]],
                np.arange(len(day.segments)), 0,
            )
        )
        model_history = indices - segment_start[indices] >= 100
        for key, values in (
            ("endpoint", endpoint),
            ("future_mean_from_now", forward_mean),
            ("past_anchor", past_anchor),
            ("equation4_ticks", equation4),
            ("equation4_return", equation4_return),
            ("model_history", model_history),
        ):
            pieces[key].append(values)
        daily.append({
            "date": date,
            "quotes": len(indices),
            "endpoint_zero_share": float(np.mean(np.abs(endpoint) < 1e-8)),
            "endpoint_absolute_move_ge_1_tick_share": float(
                np.mean(np.abs(endpoint) >= 1 - 1e-4)
            ),
            "equation4_endpoint_correlation": correlation(equation4, endpoint),
        })
        print(date, len(indices), flush=True)

    values = {key: np.concatenate(chunks) for key, chunks in pieces.items()}
    endpoint = values["endpoint"]
    forward_mean = values["future_mean_from_now"]
    past_anchor = values["past_anchor"]
    equation4 = values["equation4_ticks"]
    returns = values["equation4_return"]
    history = values["model_history"].astype(bool)
    labels = np.where(returns < -alpha, 0, np.where(returns > alpha, 2, 1))
    active_label = labels != 1
    endpoint_sign = np.sign(np.where(np.abs(endpoint) < 1e-8, 0.0, endpoint))
    label_sign = labels - 1
    model_mae_sum = sum(
        day["signal"]["prediction_actual_mae_ticks_model_quotes"]
        * (day["signal"]["quotes"] - day["signal"]["unscored_quote_count"])
        for day in monthly["tests"]
    )
    model_count = sum(
        day["signal"]["quotes"] - day["signal"]["unscored_quote_count"]
        for day in monthly["tests"]
    )
    if model_count != int(history.sum()):
        raise AssertionError("model-history population differs from monthly replay")
    result = {
        "population": {
            "dates": len(daily),
            "quote_decisions": len(endpoint),
            "model_scored_quotes": int(history.sum()),
            "unscored_quotes": int((~history).sum()),
            "exact_same_quote_population_as_monthly_replay": True,
        },
        "target_definitions": {
            "equation4": "(mean(mid[t+1:t+20]) - mean(mid[t-19:t])) / tick, with class thresholds on relative return",
            "gp_markout": "(mid[t+20] - mid[t]) / tick",
            "decomposition": "equation4_ticks = future_mean_from_now + past_anchor",
            "alpha_relative_return": alpha,
        },
        "distribution": {
            "endpoint_20_snapshot_ticks": describe(endpoint),
            "future_mean_from_now_ticks": describe(forward_mean),
            "past_anchor_ticks": describe(past_anchor),
            "equation4_ticks": describe(equation4),
            "endpoint_zero_share": float(np.mean(endpoint_sign == 0)),
            "endpoint_abs_ge_half_tick_share": float(np.mean(np.abs(endpoint) >= .5 - 1e-4)),
            "endpoint_abs_ge_one_tick_share": float(np.mean(np.abs(endpoint) >= 1 - 1e-4)),
            "endpoint_abs_ge_two_ticks_share": float(np.mean(np.abs(endpoint) >= 2 - 1e-4)),
            "equation4_label_counts_down_stationary_up": np.bincount(labels, minlength=3).tolist(),
            "equation4_label_shares_down_stationary_up": (
                np.bincount(labels, minlength=3) / len(labels)
            ).tolist(),
            "among_directional_equation4_labels_endpoint_zero_share": float(
                np.mean(endpoint_sign[active_label] == 0)
            ),
            "among_directional_equation4_labels_endpoint_opposite_share": float(
                np.mean(endpoint_sign[active_label] == -label_sign[active_label])
            ),
            "among_directional_equation4_labels_endpoint_same_share": float(
                np.mean(endpoint_sign[active_label] == label_sign[active_label])
            ),
        },
        "relationships": {
            "equation4_vs_endpoint_correlation": correlation(equation4, endpoint),
            "future_mean_vs_endpoint_correlation": correlation(forward_mean, endpoint),
            "past_anchor_vs_endpoint_correlation": correlation(past_anchor, endpoint),
            "past_anchor_vs_future_mean_correlation": correlation(past_anchor, forward_mean),
            "equation4_variance_from_past_anchor_fraction": float(
                np.var(past_anchor) / np.var(equation4)
            ),
            "equation4_variance_from_forward_mean_fraction": float(
                np.var(forward_mean) / np.var(equation4)
            ),
        },
        "prediction_vs_zero_baseline": {
            "hybrid_calibrated_mae_ticks": float(model_mae_sum / model_count),
            "zero_move_mae_ticks": float(np.mean(np.abs(endpoint[history]))),
            "hybrid_calibrated_minus_zero_mae_ticks": float(
                model_mae_sum / model_count - np.mean(np.abs(endpoint[history]))
            ),
            "note": "MAE comparison uses monthly JSON for Hybrid and raw quote replay for zero; same scored quotes.",
        },
        "daily": daily,
    }
    out = HERE / "output/hybrid_month/target_diagnostic.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "daily"}, indent=2))
    print("SAVED", out)


if __name__ == "__main__":
    main()
