#!/usr/bin/env python3
"""Causal July 2026 drift forecasts for the GP/QVI quote horizon.

Fit a fixed regularized linear model on the preceding five completed dates
accepted by the source audit.  The target is (mid[t+20] - mid[t]) / that day's exchange tick.  The
features and the seconds used to turn ticks into HKD/second are known at t.
Daily test forecasts are strictly out of sample.  The separately cached
prior-fit forecasts are for fitting GP drift states, not forecast scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "0824"))
sys.path.insert(0, str(ROOT / "0906"))

from gp_qvi_model import compact_time_to_day_ms, infer_tick_hkd  # noqa: E402
from run_multiday_strategy import load_day  # noqa: E402
from run_improved_experiment import causal_features  # noqa: E402
from directional_market_maker import select_quote_indices  # noqa: E402
from hybrid_drift_adapter import (  # noqa: E402
    causal_prediction_horizon_seconds,
    prediction_horizon_seconds,
)
from gap_window_filter import enforce_safe_day_eligibility  # noqa: E402


HORIZON = 20
RIDGE_ALPHA = 1000.0  # Fixed before looking at July test PnL.
PREDICTION_CLIP_TICKS = 3.0
DEFAULT_OUTPUT = HERE / "output/direct_drift_signal"
SOURCE_RESULT = ROOT / "0906/output/july_2026_tolerant_as_gp_drift_results.json"


@dataclass
class PreparedDay:
    date: str
    tick_hkd: float
    indices: np.ndarray
    segment_ids: np.ndarray
    has_100_snapshot_history: np.ndarray
    features: np.ndarray
    actual_move_ticks: np.ndarray
    microprice_move_ticks: np.ndarray
    causal_horizon_seconds: np.ndarray
    realized_horizon_seconds: np.ndarray
    quote_intervals_seconds: np.ndarray
    recorded_gap_events: int
    recorded_order_errors: int
    zero_recorded_errors: bool


def prepare_day(date: str, path: Path | None = None,
                *, safe_eligibility: bool = False) -> PreparedDay:
    if path is None:
        path = ROOT / f"0824/data/july_2026/hk07709_{date}_mbo_tolerant_5m_s10.npz"
    day = load_day(date, "tolerant_mbo", path)
    if safe_eligibility or day.metadata.get("gap_blackout_filter"):
        day, _ = enforce_safe_day_eligibility(day)
    if day.metadata.get("final_state_tainted"):
        raise ValueError(f"{date}: tainted reconstruction")
    indices = select_quote_indices(day, day.eligible, HORIZON)
    if not len(indices):
        raise ValueError(f"{date}: no GP quote decisions")
    if np.any(day.segments[indices] != day.segments[indices + HORIZON]):
        raise ValueError(f"{date}: markout crosses a session boundary")

    tick = infer_tick_hkd(day)
    mid = (day.book[:, 0].astype(np.float64) + day.book[:, 2]) / 2.0
    target = (mid[indices + HORIZON] - mid[indices]) / tick
    ask = day.book[indices, 0].astype(np.float64)
    ask_size = day.book[indices, 1].astype(np.float64)
    bid = day.book[indices, 2].astype(np.float64)
    bid_size = day.book[indices, 3].astype(np.float64)
    if np.any((ask_size <= 0) | (bid_size <= 0)):
        raise ValueError(f"{date}: nonpositive top-level depth")
    microprice = (ask * bid_size + bid * ask_size) / (ask_size + bid_size)
    microprice_move = (microprice - mid[indices]) / tick

    segment_starts = np.maximum.accumulate(
        np.where(
            np.r_[True, day.segments[1:] != day.segments[:-1]],
            np.arange(len(day.segments)), 0,
        )
    )
    # 0824's first eligible quote has only 99 preceding snapshots.  Its
    # 100-lag feature would otherwise index into the preceding session/day.
    valid = indices - segment_starts[indices] >= 100
    causal_seconds = causal_prediction_horizon_seconds(
        day.send_times, day.segments, indices, HORIZON
    )
    realized_seconds = prediction_horizon_seconds(
        day.send_times, day.segments, indices, HORIZON
    )
    quote_times_ms = compact_time_to_day_ms(day.send_times)[indices]
    quote_gaps = np.diff(quote_times_ms) / 1000.0
    same_segment = day.segments[indices[1:]] == day.segments[indices[:-1]]
    quote_intervals = quote_gaps[same_segment & (quote_gaps > 0)]
    if not len(quote_intervals):
        raise ValueError(f"{date}: no positive within-session quote intervals")
    spread_ticks = (ask - bid) / tick
    base = causal_features(day.book, indices[valid])
    features = np.column_stack(
        [
            base,
            microprice_move[valid],
            spread_ticks[valid],
            np.log1p(causal_seconds[valid]),
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError(f"{date}: nonfinite signal data")
    gap_count = len(day.metadata.get("gap_events") or [])
    order_error_count = sum(int(v) for v in (day.metadata.get("order_errors") or {}).values())
    return PreparedDay(
        date=date,
        tick_hkd=float(tick),
        indices=indices,
        segment_ids=day.segments[indices].astype(np.int16),
        has_100_snapshot_history=valid,
        features=features,
        actual_move_ticks=target,
        microprice_move_ticks=microprice_move,
        causal_horizon_seconds=causal_seconds,
        realized_horizon_seconds=realized_seconds,
        quote_intervals_seconds=quote_intervals,
        recorded_gap_events=gap_count,
        recorded_order_errors=order_error_count,
        zero_recorded_errors=gap_count == 0 and order_error_count == 0,
    )


def fit_model(prior: list[PreparedDay]) -> Any:
    x = np.concatenate([day.features for day in prior], axis=0)
    y = np.concatenate(
        [day.actual_move_ticks[day.has_100_snapshot_history] for day in prior]
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(x, y)
    return model


def predict_day(model: Any, day: PreparedDay) -> np.ndarray:
    predicted = np.zeros(len(day.indices), dtype=np.float64)
    predicted[day.has_100_snapshot_history] = np.clip(
        model.predict(day.features), -PREDICTION_CLIP_TICKS, PREDICTION_CLIP_TICKS
    )
    return predicted


def scalar_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    if not len(y):
        return {"mae_ticks": None, "rmse_ticks": None, "correlation": None,
                "mean_predicted_ticks": None, "prediction_std_ticks": None}
    corr = (
        float(np.corrcoef(y, p)[0, 1])
        if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0
        else None
    )
    return {
        "mae_ticks": float(np.mean(np.abs(y - p))),
        "rmse_ticks": float(np.sqrt(np.mean((y - p) ** 2))),
        "correlation": corr,
        "mean_predicted_ticks": float(np.mean(p)),
        "prediction_std_ticks": float(np.std(p)),
    }


def decile_spread(y: np.ndarray, p: np.ndarray) -> dict[str, float | int | None]:
    if len(y) < 20 or np.std(p) == 0:
        return {"bottom_mean_actual_ticks": None, "top_mean_actual_ticks": None,
                "top_minus_bottom_actual_ticks": None, "tail_count_each": 0}
    order = np.argsort(p, kind="stable")
    n = max(1, len(y) // 10)
    bottom = float(np.mean(y[order[:n]]))
    top = float(np.mean(y[order[-n:]]))
    return {
        "bottom_mean_actual_ticks": bottom,
        "top_mean_actual_ticks": top,
        "top_minus_bottom_actual_ticks": top - bottom,
        "tail_count_each": n,
    }


def prior_fit_cache(prior: list[PreparedDay], model: Any) -> dict[str, np.ndarray]:
    """Current fitted model on completed prior days, for GP state estimation."""
    indices = []
    prediction = []
    seconds = []
    tick = []
    session = []
    day_number = []
    day_lengths = []
    intervals = []
    for ordinal, day in enumerate(prior):
        p = predict_day(model, day)
        indices.append(day.indices)
        prediction.append(p)
        seconds.append(day.causal_horizon_seconds)
        tick.append(np.full(len(p), day.tick_hkd, dtype=np.float64))
        session.append(day.segment_ids)
        day_number.append(np.full(len(p), ordinal, dtype=np.int16))
        day_lengths.append(len(p))
        intervals.append(day.quote_intervals_seconds)
    return {
        "prior_quote_indices": np.concatenate(indices),
        "prior_expected_move_ticks": np.concatenate(prediction),
        "prior_causal_horizon_seconds": np.concatenate(seconds),
        "prior_tick_hkd": np.concatenate(tick),
        "prior_segment_ids": np.concatenate(session),
        "prior_day_ordinals": np.concatenate(day_number),
        "prior_day_quote_counts": np.asarray(day_lengths, dtype=np.int32),
        "prior_predicted_drift_hkd_per_second": (
            np.concatenate(prediction) * np.concatenate(tick) / np.concatenate(seconds)
        ),
        "prior_quote_intervals_seconds": np.concatenate(intervals),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def evaluate_august_holdout(
    july_dates: list[str], prepared: dict[str, PreparedDay],
    prior_days: int, output_dir: Path,
) -> dict[str, Any]:
    """Extra time holdout trained only on July dates with no recorded errors."""
    prior_dates = [date for date in july_dates if prepared[date].zero_recorded_errors][-prior_days:]
    if not prior_dates:
        raise ValueError("no anomaly-free July dates for August holdout")
    prior = [prepared[date] for date in prior_dates]
    model = fit_model(prior)
    prior_cache = prior_fit_cache(prior, model)
    rows = []
    for date in ("2026-08-04", "2026-08-07"):
        path = ROOT / f"0920/output/august_holdout/hk07709_{date}_mbo_tolerant_5m_s10.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        day = prepare_day(date, path)
        prediction = predict_day(model, day)
        valid = day.has_100_snapshot_history
        y = day.actual_move_ticks[valid]
        p = prediction[valid]
        micro = day.microprice_move_ticks[valid]
        cache_path = output_dir / f"august_day_{date}.npz"
        np.savez_compressed(
            cache_path,
            quote_indices=day.indices,
            segment_ids=day.segment_ids,
            has_100_snapshot_history=valid,
            expected_move_ticks=prediction,
            predicted_drift_hkd_per_second=(
                prediction * day.tick_hkd / day.causal_horizon_seconds
            ),
            causal_horizon_seconds=day.causal_horizon_seconds,
            actual_move_ticks_diagnostic=day.actual_move_ticks,
            realized_horizon_seconds_diagnostic=day.realized_horizon_seconds,
            raw_microprice_move_ticks=day.microprice_move_ticks,
            tick_hkd=np.asarray(day.tick_hkd),
            prior_dates=np.asarray(prior_dates),
            **prior_cache,
        )
        row = {
            "date": date,
            "training_dates": prior_dates,
            "quotes": int(len(day.indices)),
            "scored_quotes": int(valid.sum()),
            "recorded_gap_events": day.recorded_gap_events,
            "recorded_order_errors": day.recorded_order_errors,
            "zero_recorded_errors": day.zero_recorded_errors,
            "tick_hkd": day.tick_hkd,
            "causal_horizon_seconds_median": float(np.median(day.causal_horizon_seconds)),
            "realized_horizon_seconds_median_diagnostic": float(np.median(day.realized_horizon_seconds)),
            "ridge": scalar_metrics(y, p),
            "zero": scalar_metrics(y, np.zeros(len(y))),
            "raw_microprice": scalar_metrics(y, micro),
            "ridge_deciles": decile_spread(y, p),
            "microprice_deciles": decile_spread(y, micro),
            "cache": cache_path.name,
        }
        rows.append(row)
        print("AUGUST OOS", date, "corr", row["ridge"]["correlation"],
              "RMSE", row["ridge"]["rmse_ticks"],
              "zero", row["zero"]["rmse_ticks"], flush=True)
    result = {
        "design": {
            "training_dates": prior_dates,
            "training_date_rule": "most recent July source-audit eligible dates with zero recorded order errors and zero recorded session gaps",
            "model": "the same fixed Ridge(alpha=1000), scaler and 3-tick cap as July",
            "both_august_dates_scored_with_identical_july_fitted_model": True,
            "august_labels_used_for_training": False,
            "august_04_quality_note": "one source session gap and missing-order records; the tolerant reconstruction quarantines recovery spans",
        },
        "holdout": rows,
    }
    write_json(output_dir / "august_holdout_signal.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--book-manifest", type=Path, default=None,
        help="JSON book_cache_by_date mapping; permits a separate gap-filtered reconstruction",
    )
    parser.add_argument("--prior-days", type=int, default=5)
    parser.add_argument("--limit-test-days", type=int, default=None)
    parser.add_argument("--training-quality", choices=("source-audit", "zero-recorded-errors"),
                        default="source-audit")
    parser.add_argument("--august-holdout", action="store_true")
    args = parser.parse_args()
    if args.prior_days < 1 or (args.limit_test_days is not None and args.limit_test_days < 1):
        parser.error("prior-days and limit-test-days must be positive")
    if args.book_manifest is not None and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("gap-filtered books require a separate --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_RESULT.read_text())
    dates = list(source["source_audit"]["clean_trading_dates"])
    if len(dates) != 20 or not all(date.startswith("2026-07") for date in dates):
        raise ValueError("expected 20 clean July 2026 dates")
    book_paths: dict[str, Path] = {}
    if args.book_manifest is not None:
        manifest = json.loads(args.book_manifest.read_text())
        mapping = manifest["book_cache_by_date"]
        if set(mapping) != set(dates):
            raise ValueError("book manifest dates must match the source-audit dates")
        for date, value in mapping.items():
            path = Path(value).expanduser()
            book_paths[date] = path if path.is_absolute() else ROOT / path
            if not book_paths[date].is_file():
                raise FileNotFoundError(book_paths[date])
    needed = dates[: 1 + (args.limit_test_days or len(dates) - 1)]
    if args.august_holdout and len(needed) != len(dates):
        parser.error("August holdout requires all July dates prepared")
    prepared = {}
    for date in needed:
        prepared[date] = prepare_day(
            date, book_paths.get(date),
            safe_eligibility=args.book_manifest is not None,
        )
        print("PREPARED", date, len(prepared[date].indices), flush=True)

    daily = []
    pooled = {"actual": [], "ridge": [], "zero": [], "microprice": []}
    for date_ordinal, date in enumerate(needed[1:], start=1):
        available = dates[:date_ordinal]
        if args.training_quality == "zero-recorded-errors":
            available = [d for d in available if prepared[d].zero_recorded_errors]
        prior_dates = available[-args.prior_days:]
        if not prior_dates:
            raise ValueError(f"{date}: no eligible historical training dates")
        prior = [prepared[p] for p in prior_dates]
        model = fit_model(prior)
        test = prepared[date]
        prediction = predict_day(model, test)
        valid = test.has_100_snapshot_history
        y = test.actual_move_ticks[valid]
        p = prediction[valid]
        micro = test.microprice_move_ticks[valid]
        zero = np.zeros(len(y), dtype=np.float64)
        for name, values in (("actual", y), ("ridge", p),
                             ("zero", zero), ("microprice", micro)):
            pooled[name].append(values)
        prior_cache = prior_fit_cache(prior, model)
        cache_path = args.output_dir / f"day_{date}.npz"
        np.savez_compressed(
            cache_path,
            quote_indices=test.indices,
            segment_ids=test.segment_ids,
            has_100_snapshot_history=valid,
            expected_move_ticks=prediction,
            predicted_drift_hkd_per_second=(
                prediction * test.tick_hkd / test.causal_horizon_seconds
            ),
            causal_horizon_seconds=test.causal_horizon_seconds,
            actual_move_ticks_diagnostic=test.actual_move_ticks,
            realized_horizon_seconds_diagnostic=test.realized_horizon_seconds,
            raw_microprice_move_ticks=test.microprice_move_ticks,
            tick_hkd=np.asarray(test.tick_hkd),
            prior_dates=np.asarray(prior_dates),
            **prior_cache,
        )
        row = {
            "date": date,
            "prior_dates": prior_dates,
            "quote_decisions": int(len(test.indices)),
            "scored_quotes": int(valid.sum()),
            "unscored_quotes": int((~valid).sum()),
            "tick_hkd": test.tick_hkd,
            "recorded_gap_events": test.recorded_gap_events,
            "recorded_order_errors": test.recorded_order_errors,
            "zero_recorded_errors": test.zero_recorded_errors,
            "realized_horizon_seconds_median_diagnostic": float(
                np.median(test.realized_horizon_seconds)
            ),
            "causal_horizon_seconds_median": float(
                np.median(test.causal_horizon_seconds)
            ),
            "actual_markout_mean_ticks": float(np.mean(y)),
            "actual_markout_std_ticks": float(np.std(y)),
            "actual_markout_exact_zero_share": float(np.mean(np.abs(y) < 1e-5)),
            "ridge": scalar_metrics(y, p),
            "zero": scalar_metrics(y, zero),
            "raw_microprice": scalar_metrics(y, micro),
            "ridge_deciles": decile_spread(y, p),
            "microprice_deciles": decile_spread(y, micro),
            "cache": cache_path.name,
        }
        daily.append(row)
        write_json(args.output_dir / f"day_{date}.json", row)
        write_json(args.output_dir / "progress.json", {
            "completed_dates": [r["date"] for r in daily],
            "planned_dates": needed[1:],
        })
        print(
            "OOS", date,
            "corr", row["ridge"]["correlation"],
            "MAE", row["ridge"]["mae_ticks"],
            "zero", row["zero"]["mae_ticks"],
            flush=True,
        )

    joined = {key: np.concatenate(values) for key, values in pooled.items()}
    y = joined["actual"]
    result = {
        "experiment": "Fixed Ridge direct drift signal for July 2026 GP/QVI horizon",
        "period": "2026-07",
        "instrument": "HK 07709",
        "design": {
            "source_audit_eligible_dates": dates,
            "book_manifest": str(args.book_manifest) if args.book_manifest else None,
            "initial_training_date": dates[0],
            "test_dates": needed[1:],
            "rolling_prior_days": args.prior_days,
            "training_quality": args.training_quality,
            "target": "(mid[t+20]-mid[t])/current-day tick, within one continuous session",
            "feature_source": "0824 causal_features (25) plus current microprice offset, spread in ticks, log1p(observed past-20-snapshot seconds)",
            "model": "StandardScaler + Ridge(alpha=1000), refit on preceding completed dates selected by training_quality",
            "prediction_clip_ticks": PREDICTION_CLIP_TICKS,
            "gp_drift_conversion": "expected_move_ticks * current-day tick_hkd / observed t-20..t seconds",
            "unscored_session_boundary": "first quote in each segment lacks the causal 100-snapshot feature; expected move set to 0",
            "prior_fit_cache_use": "predictions from the current model on completed prior days, for fitting GP states only; these are in sample and excluded from OOS signal metrics",
            "raw_microprice_baseline": "(best_ask*bid_size + best_bid*ask_size)/(ask_size+bid_size) - current mid, in ticks",
            "no_test_day_labels_used_for_training": True,
            "model_parameters_selected_before_test_pnl": True,
            "source_quality_limitation": "Some source-audit eligible dates have recorded missing-order errors or session gaps; recovered spans are quarantined by the tolerant reconstruction, but zero recorded errors is a stricter subset.",
        },
        "summary": {
            "test_days": len(daily),
            "quote_decisions": int(sum(r["quote_decisions"] for r in daily)),
            "scored_quotes": int(len(y)),
            "ridge": scalar_metrics(y, joined["ridge"]),
            "zero": scalar_metrics(y, joined["zero"]),
            "raw_microprice": scalar_metrics(y, joined["microprice"]),
            "ridge_deciles": decile_spread(y, joined["ridge"]),
            "microprice_deciles": decile_spread(y, joined["microprice"]),
            "days_ridge_lower_mae_than_zero": sum(
                r["ridge"]["mae_ticks"] < r["zero"]["mae_ticks"] for r in daily
            ),
            "days_ridge_lower_rmse_than_zero": sum(
                r["ridge"]["rmse_ticks"] < r["zero"]["rmse_ticks"] for r in daily
            ),
            "test_days_with_recorded_gap_or_order_error": sum(
                not r["zero_recorded_errors"] for r in daily
            ),
        },
        "daily": daily,
    }
    write_json(args.output_dir / "results.json", result)
    with (args.output_dir / "daily.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "date", "quote_decisions", "scored_quotes", "tick_hkd",
            "recorded_gap_events", "recorded_order_errors", "zero_recorded_errors",
            "actual_markout_mean_ticks", "actual_markout_std_ticks",
            "realized_horizon_seconds_median_diagnostic",
            "causal_horizon_seconds_median", "ridge_mae_ticks",
            "zero_mae_ticks", "raw_microprice_mae_ticks",
            "ridge_rmse_ticks", "zero_rmse_ticks", "raw_microprice_rmse_ticks",
            "ridge_correlation", "raw_microprice_correlation",
            "ridge_top_minus_bottom_actual_ticks", "raw_microprice_top_minus_bottom_actual_ticks",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in daily:
            writer.writerow({
                **{key: row[key] for key in (
                    "date", "quote_decisions", "scored_quotes", "tick_hkd",
                    "recorded_gap_events", "recorded_order_errors", "zero_recorded_errors",
                    "actual_markout_mean_ticks", "actual_markout_std_ticks",
                    "realized_horizon_seconds_median_diagnostic",
                    "causal_horizon_seconds_median",
                )},
                **{
                    f"{name}_{metric}_ticks" if metric != "correlation" else f"{name}_correlation": row[name][metric + "_ticks" if metric in ("mae", "rmse") else metric]
                    for name in ("ridge", "zero", "raw_microprice")
                    for metric in ("mae", "rmse", "correlation")
                    if f"{name}_{metric}_ticks" in fields or f"{name}_correlation" in fields
                },
                "ridge_top_minus_bottom_actual_ticks": row["ridge_deciles"]["top_minus_bottom_actual_ticks"],
                "raw_microprice_top_minus_bottom_actual_ticks": row["microprice_deciles"]["top_minus_bottom_actual_ticks"],
            })
    print("SAVED", args.output_dir / "results.json", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    if args.august_holdout:
        evaluate_august_holdout(dates, prepared, args.prior_days, args.output_dir)


if __name__ == "__main__":
    main()
