#!/usr/bin/env python3
"""Multi-day causal-feature logistic regression and executable L1 backtest.

The script keeps MBO and MBP reconstruction families separate.  Every model is
fit on chronologically earlier observations, labels use a train-only threshold,
and every trade enters on the next sampled snapshot at the displayed ask/bid.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lob_reconstruction import reconstruct_l2_npz, save_reconstruction
from run_experiment import constant_baseline, symmetric_returns
from run_improved_experiment import (
    SEQUENCE_LENGTH,
    causal_features,
    interval_indices,
    make_labels,
    metrics_from_prediction,
)


HERE = Path(__file__).resolve().parent
RAW = Path(
    os.environ.get("DEEPLOB_7709_DATA_DIR", HERE.parent / "7709_tickdata")
).expanduser()
HORIZON = 20
STATIONARY_SHARE = 1.0 / 3.0
MODEL_C = 1.0
FEE_SCENARIOS_BPS = (0.0, 2.54, 5.54, 10.0, 20.0)
CONFIDENCE_THRESHOLDS = (0.0, 0.5, 0.6)


@dataclass
class DayData:
    date: str
    family: str
    path: Path
    book: np.ndarray
    send_times: np.ndarray
    segments: np.ndarray
    returns: np.ndarray
    eligible: np.ndarray
    features: np.ndarray
    metadata: dict[str, object]


def load_day(date: str, family: str, path: Path) -> DayData:
    with np.load(path, allow_pickle=False) as archive:
        book = archive["features"].astype(np.float32, copy=False)
        send_times = archive["send_times"].astype(np.int64, copy=False)
        segments = archive["segments"].astype(np.int16, copy=False)
        metadata = json.loads(str(archive["metadata"]))
    mids = (book[:, 0].astype(np.float64) + book[:, 2]) / 2
    returns = symmetric_returns(mids, segments, HORIZON)
    eligible = interval_indices(
        returns, segments, 0, len(book), SEQUENCE_LENGTH, HORIZON
    )
    features = causal_features(book, eligible)
    return DayData(
        date=date,
        family=family,
        path=path,
        book=book,
        send_times=send_times,
        segments=segments,
        returns=returns,
        eligible=eligible,
        features=features,
        metadata=metadata,
    )


def ensure_mbp_cache(date: str) -> Path:
    destination = HERE / "data" / f"hk07709_{date}_l2_s10.npz"
    if not destination.exists():
        result = reconstruct_l2_npz(RAW / f"hk07709_{date}.npz", snapshot_every=10)
        save_reconstruction(destination, result)
    return destination


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=MODEL_C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=20260824,
        ),
    )


def pnl_model(class_weight: str | None) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=MODEL_C,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=20260824,
        ),
    )


def label_summary(labels: np.ndarray) -> dict[str, object]:
    counts = np.bincount(labels, minlength=3)
    return {"counts": counts.tolist(), "shares": (counts / counts.sum()).tolist()}


def max_drawdown(returns: np.ndarray) -> float:
    if not len(returns):
        return 0.0
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))


def summarize_trade_returns(gross_bps: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "trades": int(len(gross_bps)),
        "gross_mean_bps_after_spread": float(np.mean(gross_bps)) if len(gross_bps) else None,
        "gross_median_bps_after_spread": float(np.median(gross_bps)) if len(gross_bps) else None,
        "break_even_extra_cost_bps": float(np.mean(gross_bps)) if len(gross_bps) else None,
        "cost_scenarios": {},
    }
    for cost in FEE_SCENARIOS_BPS:
        net = (gross_bps - cost) / 10_000.0
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()
        result["cost_scenarios"][f"{cost:.2f}"] = {
            "round_trip_extra_cost_bps": cost,
            "mean_net_bps": float(np.mean(net) * 10_000) if len(net) else None,
            "median_net_bps": float(np.median(net) * 10_000) if len(net) else None,
            "win_rate": float(np.mean(net > 0)) if len(net) else None,
            "compounded_return": float(np.prod(1.0 + net) - 1.0) if len(net) else None,
            "max_drawdown": max_drawdown(net),
            "profit_factor": float(gains / losses) if losses > 0 else None,
        }
    return result


def backtest(
    day: DayData,
    indices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    split_name: str,
    threshold: float,
    mode: str,
    holding_horizon: int = HORIZON,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if mode not in {"long_short", "long_only"}:
        raise ValueError(mode)
    last_exit = -1
    rows: list[dict[str, object]] = []
    for target, prediction, probability in zip(indices, predictions, probabilities):
        target = int(target)
        prediction = int(prediction)
        if target <= last_exit or prediction == 1:
            continue
        if mode == "long_only" and prediction != 2:
            continue
        confidence = float(probability[prediction])
        if confidence < threshold:
            continue
        entry = target + 1
        exit_index = entry + holding_horizon
        if exit_index >= len(day.book):
            continue
        if not (
            day.segments[target]
            == day.segments[entry]
            == day.segments[exit_index]
        ):
            continue
        if prediction == 2:
            side = "long"
            entry_price = float(day.book[entry, 0])
            exit_price = float(day.book[exit_index, 2])
            gross_return = exit_price / entry_price - 1.0
        else:
            side = "short"
            entry_price = float(day.book[entry, 2])
            exit_price = float(day.book[exit_index, 0])
            gross_return = entry_price / exit_price - 1.0
        gross_bps = gross_return * 10_000.0
        rows.append(
            {
                "split": split_name,
                "family": day.family,
                "date": day.date,
                "mode": mode,
                "confidence_threshold": threshold,
                "holding_horizon": holding_horizon,
                "signal_time": int(day.send_times[target]),
                "entry_time": int(day.send_times[entry]),
                "exit_time": int(day.send_times[exit_index]),
                "side": side,
                "confidence": confidence,
                "entry_price_hkd": entry_price,
                "exit_price_hkd": exit_price,
                "gross_bps_after_spread": gross_bps,
            }
        )
        last_exit = exit_index
    gross = np.asarray([row["gross_bps_after_spread"] for row in rows], dtype=np.float64)
    summary = summarize_trade_returns(gross)
    summary["long_trades"] = sum(row["side"] == "long" for row in rows)
    summary["short_trades"] = sum(row["side"] == "short" for row in rows)
    summary["confidence_threshold"] = threshold
    summary["mode"] = mode
    return summary, rows


def executable_indices(
    day: DayData, indices: np.ndarray, holding_horizon: int = HORIZON
) -> np.ndarray:
    selected = indices[indices + holding_horizon + 1 < len(day.book)]
    return selected[
        (day.segments[selected] == day.segments[selected + 1])
        & (
            day.segments[selected]
            == day.segments[selected + holding_horizon + 1]
        )
    ]


def pnl_labels(
    day: DayData,
    indices: np.ndarray,
    training_cost_bps: float = 5.54,
    holding_horizon: int = HORIZON,
) -> np.ndarray:
    """Best executable action after spread and a fixed round-trip cost.

    0=short, 1=flat, 2=long.  Future prices are used only to form training and
    evaluation outcomes; model inputs remain strictly known at signal time.
    """

    entry = indices + 1
    exit_index = entry + holding_horizon
    long_bps = (day.book[exit_index, 2] / day.book[entry, 0] - 1.0) * 10_000
    short_bps = (day.book[entry, 2] / day.book[exit_index, 0] - 1.0) * 10_000
    long_net = long_bps - training_cost_bps
    short_net = short_bps - training_cost_bps
    return np.where(
        (long_net > 0) & (long_net >= short_net),
        2,
        np.where(short_net > 0, 0, 1),
    ).astype(np.int64)


def fit_evaluate_pnl(
    train_days: list[DayData],
    test_day: DayData,
    split_name: str,
    class_weight: str | None,
    holding_horizon: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_indices = [
        executable_indices(day, day.eligible, holding_horizon) for day in train_days
    ]
    test_indices = executable_indices(test_day, test_day.eligible, holding_horizon)
    train_features = np.concatenate(
        [causal_features(day.book, idx) for day, idx in zip(train_days, train_indices)]
    )
    train_labels = np.concatenate(
        [
            pnl_labels(day, idx, holding_horizon=holding_horizon)
            for day, idx in zip(train_days, train_indices)
        ]
    )
    test_features = causal_features(test_day.book, test_indices)
    test_labels = pnl_labels(
        test_day, test_indices, holding_horizon=holding_horizon
    )
    estimator = pnl_model(class_weight)
    estimator.fit(train_features, train_labels)
    predictions = estimator.predict(test_features)
    probabilities = estimator.predict_proba(test_features)
    classification = metrics_from_prediction(test_labels, predictions)
    classification["train_samples"] = int(len(train_labels))
    classification["train_labels"] = label_summary(train_labels)
    classification["test_labels"] = label_summary(test_labels)
    classification["training_round_trip_cost_bps"] = 5.54
    classification["holding_horizon"] = holding_horizon
    strategies: dict[str, object] = {}
    trade_rows: list[dict[str, object]] = []
    for mode in ("long_short", "long_only"):
        for threshold in CONFIDENCE_THRESHOLDS:
            summary, rows = backtest(
                test_day,
                test_indices,
                predictions,
                probabilities,
                split_name,
                threshold,
                mode,
                holding_horizon,
            )
            strategies[f"{mode}_p{threshold:.1f}"] = summary
            trade_rows.extend(rows)
    return (
        {
            "name": split_name,
            "family": test_day.family,
            "train_dates": [day.date for day in train_days],
            "test_date": test_day.date,
            "class_weight": class_weight,
            "holding_horizon": holding_horizon,
            "classification": classification,
            "strategies": strategies,
        },
        trade_rows,
    )


def fit_evaluate(
    train_days: list[DayData],
    test_day: DayData,
    split_name: str,
    test_indices: np.ndarray | None = None,
    train_slices: list[np.ndarray] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    if train_slices is None:
        train_slices = [day.eligible for day in train_days]
    if test_indices is None:
        test_indices = test_day.eligible
    train_returns = np.concatenate(
        [day.returns[idx] for day, idx in zip(train_days, train_slices)]
    )
    alpha = float(np.quantile(np.abs(train_returns), STATIONARY_SHARE))
    train_labels = np.concatenate(
        [make_labels(day.returns, idx, alpha) for day, idx in zip(train_days, train_slices)]
    )
    train_features = np.concatenate(
        [causal_features(day.book, idx) for day, idx in zip(train_days, train_slices)]
    )
    test_labels = make_labels(test_day.returns, test_indices, alpha)
    estimator = model()
    estimator.fit(train_features, train_labels)
    prediction = estimator.predict(causal_features(test_day.book, test_indices))
    probabilities = estimator.predict_proba(causal_features(test_day.book, test_indices))
    classification = metrics_from_prediction(test_labels, prediction)
    classification["alpha"] = alpha
    classification["train_samples"] = int(len(train_labels))
    classification["train_labels"] = label_summary(train_labels)
    classification["test_labels"] = label_summary(test_labels)
    majority_class = int(np.bincount(train_labels, minlength=3).argmax())
    classification["majority_baseline"] = constant_baseline(test_labels, majority_class)

    strategies: dict[str, object] = {}
    trade_rows: list[dict[str, object]] = []
    for mode in ("long_short", "long_only"):
        for threshold in CONFIDENCE_THRESHOLDS:
            summary, rows = backtest(
                test_day,
                test_indices,
                prediction,
                probabilities,
                split_name,
                threshold,
                mode,
            )
            strategies[f"{mode}_p{threshold:.1f}"] = summary
            trade_rows.extend(rows)
    result = {
        "name": split_name,
        "family": test_day.family,
        "train_dates": [day.date for day in train_days],
        "test_date": test_day.date,
        "classification": classification,
        "strategies": strategies,
    }
    return result, trade_rows, classification["confusion_matrix"]


def per_day_split(day: DayData) -> tuple[np.ndarray, np.ndarray]:
    split = int(np.floor(len(day.book) * 0.70))
    train_idx = interval_indices(
        day.returns, day.segments, 0, split, SEQUENCE_LENGTH, HORIZON
    )
    test_idx = interval_indices(
        day.returns, day.segments, split, len(day.book), SEQUENCE_LENGTH, HORIZON
    )
    return train_idx, test_idx


def quality_record(day: DayData) -> dict[str, object]:
    return {
        "date": day.date,
        "family": day.family,
        "path": str(day.path.resolve()),
        "snapshots": int(len(day.book)),
        "eligible_targets": int(len(day.eligible)),
        "first_send_time": int(day.send_times[0]),
        "last_send_time": int(day.send_times[-1]),
        "segments": day.metadata.get("segments"),
        "final_state_tainted": day.metadata.get("final_state_tainted", False),
        "order_errors": day.metadata.get("order_errors", {}),
        "quality_counts": day.metadata.get("quality_counts", {}),
    }


def aggregate_strategy(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for mode in ("long_short", "long_only"):
        for threshold in CONFIDENCE_THRESHOLDS:
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["confidence_threshold"] == threshold
            ]
            gross = np.asarray(
                [row["gross_bps_after_spread"] for row in selected], dtype=np.float64
            )
            summary = summarize_trade_returns(gross)
            summary["long_trades"] = sum(row["side"] == "long" for row in selected)
            summary["short_trades"] = sum(row["side"] == "short" for row in selected)
            result[f"{mode}_p{threshold:.1f}"] = summary
    return result


def main() -> None:
    mbo_dates = (
        "2025-11-06",
        "2025-11-13",
        "2025-12-09",
        "2025-12-19",
        "2026-07-09",
        "2026-07-21",
        "2026-08-04",
        "2026-08-07",
    )
    mbo = {
        date: load_day(
            date, "strict_mbo", HERE / "data" / f"hk07709_{date}_mbo_strict_s10.npz"
        )
        for date in mbo_dates
    }
    mbp_dates = ("2026-07-21", "2026-07-22", "2026-08-07")
    mbp = {
        date: load_day(date, "absolute_mbp", ensure_mbp_cache(date))
        for date in mbp_dates
    }

    complete_mbo = [day for day in mbo.values() if not day.metadata.get("final_state_tainted")]
    quality = [quality_record(day) for day in mbo.values()] + [
        quality_record(day) for day in mbp.values()
    ]
    experiments: dict[str, list[dict[str, object]]] = {
        "per_day_70_30_mbo": [],
        "walk_forward_mbo_recent": [],
        "walk_forward_mbo_all_history": [],
        "walk_forward_mbp_adjacent": [],
        "pnl_aligned_walkforward": [],
    }
    all_trades: list[dict[str, object]] = []

    for day in complete_mbo:
        train_idx, test_idx = per_day_split(day)
        result, trades, _ = fit_evaluate(
            [day],
            day,
            f"per_day_70_30_mbo_{day.date}",
            test_indices=test_idx,
            train_slices=[train_idx],
        )
        experiments["per_day_70_30_mbo"].append(result)
        all_trades.extend(trades)
        print(
            "PER_DAY",
            day.date,
            f"accuracy={result['classification']['accuracy']:.4f}",
            f"macro_f1={result['classification']['macro_f1']:.4f}",
            flush=True,
        )

    recent_specs = [
        ([mbo["2026-07-09"]], mbo["2026-07-21"], "mbo_recent_0709_to_0721"),
        (
            [mbo["2026-07-09"], mbo["2026-07-21"]],
            mbo["2026-08-07"],
            "mbo_recent_0709_0721_to_0807",
        ),
    ]
    for train_days, test_day, name in recent_specs:
        result, trades, _ = fit_evaluate(train_days, test_day, name)
        experiments["walk_forward_mbo_recent"].append(result)
        all_trades.extend(trades)
        print(
            "WALK_MBO_RECENT",
            name,
            f"accuracy={result['classification']['accuracy']:.4f}",
            f"macro_f1={result['classification']['macro_f1']:.4f}",
            flush=True,
        )

    for position in range(1, len(complete_mbo)):
        train_days = complete_mbo[:position]
        test_day = complete_mbo[position]
        name = f"mbo_cumulative_to_{test_day.date}"
        result, trades, _ = fit_evaluate(train_days, test_day, name)
        experiments["walk_forward_mbo_all_history"].append(result)
        all_trades.extend(trades)

    mbp_specs = [
        ([mbp["2026-07-21"]], mbp["2026-07-22"], "mbp_0721_to_0722"),
        (
            [mbp["2026-07-21"], mbp["2026-07-22"]],
            mbp["2026-08-07"],
            "mbp_0721_0722_to_0807",
        ),
    ]
    for train_days, test_day, name in mbp_specs:
        result, trades, _ = fit_evaluate(train_days, test_day, name)
        experiments["walk_forward_mbp_adjacent"].append(result)
        all_trades.extend(trades)
        print(
            "WALK_MBP",
            name,
            f"accuracy={result['classification']['accuracy']:.4f}",
            f"macro_f1={result['classification']['macro_f1']:.4f}",
            flush=True,
        )

    pnl_specs = recent_specs + mbp_specs
    for train_days, test_day, base_name in pnl_specs:
        for holding_horizon in (20, 50, 100):
            for class_weight in (None, "balanced"):
                suffix = "natural" if class_weight is None else "balanced"
                name = f"pnl_{base_name}_h{holding_horizon}_{suffix}"
                result, trades = fit_evaluate_pnl(
                    train_days,
                    test_day,
                    name,
                    class_weight,
                    holding_horizon,
                )
                experiments["pnl_aligned_walkforward"].append(result)
                all_trades.extend(trades)
                strategy = result["strategies"]["long_short_p0.0"]
                print(
                    "PNL_ALIGNED",
                    name,
                    f"trades={strategy['trades']}",
                    f"gross_mean_bps={strategy['gross_mean_bps_after_spread']}",
                    flush=True,
                )

    walkforward_names = {
        result["name"]
        for group in (
            experiments["walk_forward_mbo_recent"],
            experiments["walk_forward_mbp_adjacent"],
        )
        for result in group
    }
    primary_rows = [row for row in all_trades if row["split"] in walkforward_names]
    output = {
        "as_of": "2026-08-24",
        "instrument": {
            "stock_code": "07709",
            "name": "CSOP SK Hynix Daily (2x) Leveraged Product",
            "type": "Leveraged and Inverse Product (L&I Product)",
        },
        "method": {
            "model": "StandardScaler + class-balanced multinomial LogisticRegression",
            "model_c": MODEL_C,
            "features": "25 causal-in-time order-book features",
            "sequence_length_for_eligibility": SEQUENCE_LENGTH,
            "label_horizon_snapshots": HORIZON,
            "stationary_share_train_only": STATIONARY_SHARE,
            "execution": "signal at t; enter at displayed ask/bid at t+1; exit at opposite displayed bid/ask at t+1+20; no overlapping positions",
            "spread": "embedded in executable entry and exit prices",
            "fee_scenarios_round_trip_bps_beyond_spread": list(FEE_SCENARIOS_BPS),
            "official_fixed_round_trip_cost_bps": 2.54,
            "official_fixed_cost_components_per_side_percent": {
                "SFC_transaction_levy": 0.0027,
                "AFRC_transaction_levy": 0.00015,
                "HKEX_trading_fee": 0.00565,
                "HKSCC_stock_settlement_fee": 0.0042,
            },
            "assumed_brokerage_for_5_54_bps_scenario_round_trip_bps": 3.0,
            "stamp_duty": "exempt for all L&I Products per HKEX exemption list",
        },
        "data_quality": quality,
        "experiments": experiments,
        "primary_walkforward_strategy_aggregate": aggregate_strategy(primary_rows),
        "limitations": [
            "Only a handful of non-contiguous days are available.",
            "Displayed depth does not guarantee full execution; queue position and market impact are omitted.",
            "Sampling every ten changed timestamp groups makes t+1 a variable latency.",
            "Short borrow availability, borrow fees, tick-rule constraints, broker minimum fees, and latency are omitted.",
            "The features are causal in time but this is not causal inference.",
        ],
    }
    output_path = HERE / "output" / "multiday_strategy_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    trade_path = HERE / "output" / "multiday_strategy_trades.csv"
    if all_trades:
        with trade_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_trades[0]))
            writer.writeheader()
            writer.writerows(all_trades)
    print("WROTE", output_path, trade_path, flush=True)


if __name__ == "__main__":
    main()
