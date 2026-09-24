#!/usr/bin/env python3
"""Walk-forward direction classification on causal July L2/L3 features.

The primary target is the sign of the endpoint midpoint move at 20 snapshots.
Flat endpoints are reported separately and excluded from binary direction
metrics. Every fitted model uses at most five preceding complete-MBO dates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


HERE = Path(__file__).resolve().parent
DATA = HERE / "output/direction_dataset"
DEFAULT_OUTPUT = HERE / "output/direction_benchmark"


def features(days: list[dict], kind: str) -> np.ndarray:
    if kind == "l2":
        return np.concatenate([day["l2"] for day in days])
    if kind == "l2_l3_3":
        return np.concatenate([np.column_stack((day["l2"], day["l3"][:, 1:4]))
                               for day in days])
    if kind == "l2_l3_6":
        return np.concatenate([np.column_stack((day["l2"], day["l3"]))
                               for day in days])
    if kind == "l3_6_only":
        return np.concatenate([day["l3"] for day in days])
    raise ValueError(kind)


def fit_scores(method: str, x_train: np.ndarray, move_train: np.ndarray,
               x_test: np.ndarray) -> tuple[np.ndarray, bool]:
    if method == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
        model.fit(x_train, move_train)
        return model.predict(x_test), False
    directional = move_train != 0
    y_train = (move_train[directional] > 0).astype(np.int8)
    if len(np.unique(y_train)) != 2:
        raise ValueError("only one training direction")
    if method == "logit":
        model = make_pipeline(
            StandardScaler(), LogisticRegression(
                C=0.05, solver="liblinear", max_iter=200, random_state=23
            ),
        )
        model.fit(x_train[directional], y_train)
        return model.predict_proba(x_test)[:, 1], True
    if method == "svm":
        model = make_pipeline(
            StandardScaler(), LinearSVC(C=0.01, dual=False, max_iter=2000,
                                        random_state=23),
        )
        model.fit(x_train[directional], y_train)
        return model.decision_function(x_test), False
    if method == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=60, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=10.0,
            max_bins=64, early_stopping=False, random_state=23,
        )
        model.fit(x_train[directional], y_train)
        return model.predict_proba(x_test)[:, 1], True
    raise ValueError(method)


def direction_metrics(move: np.ndarray, score: np.ndarray,
                      probability: bool) -> dict:
    if len(move) != len(score) or not np.isfinite(score).all():
        raise AssertionError("score length or finiteness differs")
    selected = move != 0
    true = (move[selected] > 0).astype(np.int8)
    selected_score = score[selected]
    predicted = (selected_score >= (0.5 if probability else 0.0)).astype(np.int8)
    result = {
        "quotes": int(len(move)),
        "directional_quotes": int(np.count_nonzero(selected)),
        "flat_quotes": int(np.count_nonzero(~selected)),
        "up_share_directional": float(np.mean(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "roc_auc": float(roc_auc_score(true, selected_score)),
        "predicted_up_share": float(np.mean(predicted)),
    }
    if probability:
        high = np.abs(selected_score - 0.5) >= 0.05
        result["confidence_55_coverage"] = float(np.mean(high))
        result["confidence_55_accuracy"] = (
            float(accuracy_score(true[high], predicted[high])) if high.any() else None
        )
    return result


def method_specs(suite: str) -> list[tuple[str, str, str]]:
    basic = [
        ("ridge_l2", "ridge", "l2"),
        ("ridge_l2_l3_6", "ridge", "l2_l3_6"),
        ("logit_l2", "logit", "l2"),
        ("logit_l2_l3_6", "logit", "l2_l3_6"),
    ]
    if suite == "short":
        return basic
    return basic + [
        ("logit_l2_l3_3", "logit", "l2_l3_3"),
        ("logit_l3_6_only", "logit", "l3_6_only"),
        ("svm_l2", "svm", "l2"),
        ("svm_l2_l3_6", "svm", "l2_l3_6"),
        ("hgb_l2", "hgb", "l2"),
        ("hgb_l2_l3_6", "hgb", "l2_l3_6"),
    ]


def load_data(dates: list[str]) -> dict[str, dict]:
    data = {}
    for date in dates:
        with np.load(DATA / f"day_{date}.npz", allow_pickle=False) as archive:
            data[date] = {key: archive[key].copy() for key in archive.files}
        if (data[date]["l3"].shape[1] == 6) != bool(data[date]["l3_available"]):
            raise AssertionError(f"L3 availability differs on {date}")
    return data


def run_day(date: str, ordinal: int, dates: list[str], data: dict[str, dict],
            horizon: int, suite: str, output: Path) -> None:
    prior = [d for d in dates[:ordinal] if bool(data[d]["l3_available"])][-5:]
    if not prior or any(d >= date for d in prior):
        raise AssertionError(f"invalid prior dates on {date}")
    training_days = [data[d] for d in prior]
    test = data[date]
    target = test[f"move_{horizon}_ticks"]
    training_move = np.concatenate([d[f"move_{horizon}_ticks"] for d in training_days])
    directional_train = training_move != 0
    prior_up_share = float(np.mean(training_move[directional_train] > 0))
    fallback_up = prior_up_share >= 0.5
    micro = test["microprice_move_ticks"].astype(np.float64)
    micro[micro == 0] = 1e-9 if fallback_up else -1e-9
    scores = {
        "prior_majority": np.full(len(target), prior_up_share),
        "microprice_sign": micro,
    }
    probabilities = {"prior_majority": True, "microprice_sign": False}
    matrix_train = {}
    matrix_test = {}
    available = bool(test["l3_available"])
    for name, method, kind in method_specs(suite):
        if kind == "l3_6_only" and not available:
            continue
        if kind != "l2" and not available:
            book_name = f"{method}_l2"
            if book_name not in scores:
                raise AssertionError(f"missing fallback {book_name}")
            scores[name] = scores[book_name].copy()
            probabilities[name] = probabilities[book_name]
            continue
        if kind not in matrix_train:
            matrix_train[kind] = features(training_days, kind)
            matrix_test[kind] = features([test], kind)
        score, probability = fit_scores(
            method, matrix_train[kind], training_move, matrix_test[kind]
        )
        scores[name] = score
        probabilities[name] = probability
        print("FIT", date, horizon, name, flush=True)
    records = {
        name: {"method": name, "used_l3": available and "l3" in name,
               "probability_score": probabilities[name],
               **direction_metrics(target, score, probabilities[name])}
        for name, score in scores.items()
    }
    out = output / "per_day" / f"{date}_h{horizon}"
    np.savez_compressed(out.with_suffix(".npz"), move_ticks=target,
                        **{f"score_{k}": v.astype(np.float32)
                           for k, v in scores.items()})
    out.with_suffix(".json").write_text(json.dumps({
        "date": date, "horizon_snapshots": horizon,
        "l3_available": available, "prior_dates": prior,
        "prior_directional_up_share": prior_up_share,
        "methods": records,
    }, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print("DONE", date, horizon, "methods", len(records), flush=True)


def aggregate(output: Path, dates: list[str], horizons: tuple[int, ...], suite: str) -> None:
    daily = []
    for date in dates[1:]:
        for horizon in horizons:
            daily.append(json.loads((output / "per_day" /
                                     f"{date}_h{horizon}.json").read_text()))
    summary = []
    for horizon in horizons:
        for subset in ("all", "l3_available", "l3_unavailable"):
            selected = [d for d in daily if d["horizon_snapshots"] == horizon
                        and (subset == "all" or d["l3_available"] ==
                             (subset == "l3_available"))]
            names = sorted(set().union(*(d["methods"] for d in selected)))
            for name in names:
                chosen = [d for d in selected if name in d["methods"]]
                ys, scores = [], []
                for d in chosen:
                    with np.load(output / "per_day" /
                                 f"{d['date']}_h{horizon}.npz", allow_pickle=False) as archive:
                        ys.append(archive["move_ticks"])
                        scores.append(archive[f"score_{name}"])
                y, score = np.concatenate(ys), np.concatenate(scores)
                probability = chosen[0]["methods"][name]["probability_score"]
                summary.append({
                    "horizon_snapshots": horizon, "subset": subset,
                    "method": name, "days": len(chosen),
                    "mean_daily_accuracy": float(np.mean([
                        d["methods"][name]["accuracy"] for d in chosen
                    ])),
                    **direction_metrics(y, score, probability),
                })
    result = {
        "name": "causal July direction benchmark", "suite": suite,
        "horizons_snapshots": horizons, "train_protocol":
        "at most five prior complete-MBO days; identical prior dates for L2 and L2+L3",
        "target": "sign of rounded half-tick endpoint midpoint change; zero moves excluded from binary metrics",
        "label_policy": "zero price changes reported but not used for binary training or scoring",
        "source": str((DATA / "manifest.json").resolve()),
        "model_settings": {
            "ridge": "StandardScaler + Ridge(alpha=1000), sign of regressed tick move",
            "logit": "StandardScaler + LogisticRegression(C=0.05, solver=liblinear)",
            "svm": "StandardScaler + LinearSVC(C=0.01, dual=False)",
            "hgb": "HistGradientBoostingClassifier(60 iterations, 15 leaves, 100 min leaf, L2=10)",
            "l3_fallback": "exact same-method L2 score on incomplete MBO days",
        },
        "test_dates": dates[1:], "summary": summary, "daily": daily,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    print(json.dumps([s for s in summary if s["subset"] == "all"],
                     ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("full", "short"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+",
                        help="override the suite's default snapshot horizons")
    parser.add_argument("--limit-test-days", type=int)
    args = parser.parse_args()
    manifest = json.loads((DATA / "manifest.json").read_text())
    dates = [d["date"] for d in manifest["dates"]]
    if args.limit_test_days is not None:
        if args.limit_test_days < 1:
            parser.error("test-day limit must be positive")
        dates = dates[:1 + args.limit_test_days]
    horizons = tuple(args.horizons) if args.horizons else (
        (20,) if args.suite == "full" else (5, 10)
    )
    if any(h not in (5, 10, 20) for h in horizons) or len(horizons) != len(set(horizons)):
        parser.error("horizons must be distinct members of 5, 10, 20")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_day").mkdir(exist_ok=True)
    data = load_data(dates)
    for ordinal, date in enumerate(dates[1:], start=1):
        for horizon in horizons:
            out = args.output_dir / "per_day" / f"{date}_h{horizon}"
            if out.with_suffix(".json").exists() and out.with_suffix(".npz").exists():
                print("REUSE", date, horizon, flush=True)
                continue
            run_day(date, ordinal, dates, data, horizon, args.suite, args.output_dir)
    aggregate(args.output_dir, dates, horizons, args.suite)


if __name__ == "__main__":
    main()
