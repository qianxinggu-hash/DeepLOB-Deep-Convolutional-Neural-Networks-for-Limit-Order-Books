#!/usr/bin/env python3
"""Walk-forward comparison of top-ten L2 and identity-aware L3 drift signals.

The six L3 features and Ridge penalty are fixed in code.  Each prediction uses
at most five prior complete, zero-error trading days; no current-day labels are
used.  Dates with recorded gaps are omitted because missing MBO messages can
leave the surviving OrderID population wrong after a two-minute blackout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from l3_order_signal import FEATURE_NAMES, extract_l3_features
from run_direct_drift_signal import (
    RIDGE_ALPHA,
    ROOT,
    decile_spread,
    prepare_day,
    scalar_metrics,
)
from run_multiday_strategy import load_day


MANIFEST = ROOT / "0920/output/gap_2m_books/filtered_manifest.json"
DEFAULT_OUTPUT = ROOT / "0920/output/l3_order_signal"


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), -3.0, 3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=Path.home() / "Downloads/7709_202607")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    days = {}
    for date in manifest["date_order"]:
        cache_path = ROOT / manifest["book_cache_by_date"][date]
        prepared = prepare_day(date, cache_path)
        if not prepared.zero_recorded_errors:
            continue
        feature_cache = args.output / f"l3_features_{date}.npz"
        if feature_cache.exists() and not args.rebuild:
            with np.load(feature_cache) as archive:
                if not np.array_equal(archive["quote_indices"], prepared.indices):
                    raise ValueError(f"stale L3 cache for {date}")
                if not np.array_equal(archive["feature_names"], np.asarray(FEATURE_NAMES)):
                    raise ValueError(f"stale L3 feature names for {date}; use --rebuild")
                l3 = archive["l3_features"].copy()
                diagnostics = json.loads(str(archive["diagnostics"]))
        else:
            book_day = load_day(date, "tolerant_mbo", cache_path)
            raw_csv = args.raw_dir / f"hk07709_{date}.csv"
            l3, diagnostics = extract_l3_features(
                raw_csv, book_day.send_times[prepared.indices], book_day.book[prepared.indices]
            )
            np.savez_compressed(
                feature_cache,
                quote_indices=prepared.indices,
                l3_features=l3,
                feature_names=np.asarray(FEATURE_NAMES),
                diagnostics=json.dumps(diagnostics),
            )
        if l3.shape != (len(prepared.indices), len(FEATURE_NAMES)):
            raise ValueError(f"bad L3 feature shape for {date}")
        valid = prepared.has_100_snapshot_history
        days[date] = {
            "prepared": prepared,
            "l2": prepared.features.astype(np.float64),
            "l3": l3[valid].astype(np.float64),
            "y": prepared.actual_move_ticks[valid],
            "diagnostics": diagnostics,
        }
        print(f"{date}: {len(prepared.indices)} quotes, {diagnostics['quotes_checked']} matched", flush=True)

    dates = list(days)
    if len(dates) < 2:
        raise ValueError("need at least two clean dates")
    rows = []
    model_names = (
        "l2", "l3_lifecycle", "l3_identity", "l3_removal", "l3",
        "l2_plus_l3_lifecycle", "l2_plus_l3_identity",
        "l2_plus_l3_removal", "l2_plus_l3",
    )
    pooled = {key: {"actual": [], "predicted": []} for key in model_names}
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for position, date in enumerate(dates[1:], start=1):
        prior_dates = dates[max(0, position - 5):position]
        current = days[date]
        y = current["y"]
        predictions = {}
        for name in pooled:
            if name == "l2":
                matrix = lambda day: day["l2"]
            elif name == "l3_lifecycle":
                matrix = lambda day: day["l3"][:, 1:4]
            elif name == "l3_identity":
                matrix = lambda day: day["l3"][:, :4]
            elif name == "l3_removal":
                matrix = lambda day: day["l3"][:, 4:]
            elif name == "l3":
                matrix = lambda day: day["l3"]
            elif name == "l2_plus_l3_lifecycle":
                matrix = lambda day: np.column_stack((day["l2"], day["l3"][:, 1:4]))
            elif name == "l2_plus_l3_identity":
                matrix = lambda day: np.column_stack((day["l2"], day["l3"][:, :4]))
            elif name == "l2_plus_l3_removal":
                matrix = lambda day: np.column_stack((day["l2"], day["l3"][:, 4:]))
            else:
                matrix = lambda day: np.column_stack((day["l2"], day["l3"]))
            train_x = np.concatenate([matrix(days[d]) for d in prior_dates])
            train_y = np.concatenate([days[d]["y"] for d in prior_dates])
            p = fit_predict(train_x, train_y, matrix(current))
            predictions[name] = p
            pooled[name]["actual"].append(y)
            pooled[name]["predicted"].append(p)
        np.savez_compressed(
            prediction_dir / f"{date}.npz",
            quote_indices=current["prepared"].indices[current["prepared"].has_100_snapshot_history],
            causal_horizon_seconds=current["prepared"].causal_horizon_seconds[
                current["prepared"].has_100_snapshot_history
            ],
            tick_hkd=np.asarray(current["prepared"].tick_hkd),
            actual_move_ticks=y,
            **{f"prediction_{name}": p for name, p in predictions.items()},
        )
        row = {
            "date": date,
            "training_dates": prior_dates,
            "quotes": int(len(y)),
            "models": {
                name: {**scalar_metrics(y, p), **decile_spread(y, p)}
                for name, p in predictions.items()
            },
        }
        baseline_mse = float(np.mean((y - predictions["l2"]) ** 2))
        combined_mse = float(np.mean((y - predictions["l2_plus_l3"]) ** 2))
        row["combined_mse_reduction_vs_l2_pct"] = 100 * (baseline_mse - combined_mse) / baseline_mse
        rows.append(row)
        print(f"{date}: L2 corr={row['models']['l2']['correlation']:.4f}, "
              f"L2+L3 corr={row['models']['l2_plus_l3']['correlation']:.4f}, "
              f"MSE lift={row['combined_mse_reduction_vs_l2_pct']:+.3f}%", flush=True)

    overall = {}
    for name, values in pooled.items():
        y = np.concatenate(values["actual"])
        p = np.concatenate(values["predicted"])
        overall[name] = {**scalar_metrics(y, p), **decile_spread(y, p)}
    l2_mse = overall["l2"]["rmse_ticks"] ** 2
    combined_mse = overall["l2_plus_l3"]["rmse_ticks"] ** 2
    improvement = np.asarray([r["combined_mse_reduction_vs_l2_pct"] for r in rows])
    # Resample whole test days, preserving all within-day serial dependence.
    rng = np.random.default_rng(20260923)
    draws = rng.integers(0, len(rows), size=(20_000, len(rows)))
    day_block_uncertainty = {}
    for name in ("l3_lifecycle", "l2_plus_l3_lifecycle", "l3_identity",
                 "l2_plus_l3_identity", "l3", "l2_plus_l3"):
        baseline_sse = np.asarray([
            r["models"]["l2"]["rmse_ticks"] ** 2 * r["quotes"] for r in rows
        ])
        signal_sse = np.asarray([
            r["models"][name]["rmse_ticks"] ** 2 * r["quotes"] for r in rows
        ])
        sampled_baseline = baseline_sse[draws].sum(axis=1)
        sampled_signal = signal_sse[draws].sum(axis=1)
        sampled_lift = 100 * (sampled_baseline - sampled_signal) / sampled_baseline
        day_block_uncertainty[name] = {
            "mse_reduction_pct_ci95": np.quantile(sampled_lift, (0.025, 0.975)).tolist(),
            "bootstrap_fraction_positive": float(np.mean(sampled_lift > 0)),
            "days_mse_better_than_l2": int(sum(
                r["models"][name]["rmse_ticks"] < r["models"]["l2"]["rmse_ticks"]
                for r in rows
            )),
        }
    result = {
        "protocol": {
            "source": "HK 07709 MBO CSV joined to independently reconstructed ten-level sampled book",
            "dates": dates,
            "train_days_max": 5,
            "horizon_snapshots": 20,
            "ridge_alpha": RIDGE_ALPHA,
            "features": list(FEATURE_NAMES),
            "price_unit": "exchange ticks",
            "sample": "non-overlapping GP quote starts, full 100-snapshot history, no recorded MBO gaps or errors",
            "delete_semantics": "MsgType=32 indicates order removal; cannot distinguish cancellation from execution from this field alone",
        },
        "extractor_diagnostics": {d: days[d]["diagnostics"] for d in dates},
        "days": rows,
        "pooled": overall,
        "day_block_uncertainty": day_block_uncertainty,
        "pooled_combined_mse_reduction_vs_l2_pct": 100 * (l2_mse - combined_mse) / l2_mse,
        "daily_mse_reduction_median_pct": float(np.median(improvement)),
        "days_combined_mse_better": int(np.sum(improvement > 0)),
        "test_days": len(rows),
        "test_quotes": int(sum(row["quotes"] for row in rows)),
    }
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "test_days", "test_quotes", "pooled_combined_mse_reduction_vs_l2_pct",
        "daily_mse_reduction_median_pct", "days_combined_mse_better")}, indent=2))


if __name__ == "__main__":
    main()
