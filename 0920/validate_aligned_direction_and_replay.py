#!/usr/bin/env python3
"""Audit matched-label predictions and same-simulator economic comparisons."""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRED = HERE / "output/aligned_direction_comparison"
REPLAY = HERE / "output/aligned_direction_market_making"


def bootstrap_delta(correct_a: np.ndarray, correct_b: np.ndarray,
                    counts: np.ndarray, seed: int = 27) -> dict:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(counts), size=(20_000, len(counts)))
    values = ((correct_a[index].sum(axis=1) - correct_b[index].sum(axis=1))
              / counts[index].sum(axis=1))
    return {"delta_percentage_points": float(100 * (correct_a.sum() -
                                                       correct_b.sum()) / counts.sum()),
            "day_bootstrap_95pp": (100 * np.quantile(values, [0.025, 0.975])).tolist(),
            "positive_days": int(np.sum(correct_a > correct_b)),
            "days": int(len(counts))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=REPLAY)
    parser.add_argument("--tick-map", choices=("calibrated", "hard_one_tick"),
                        default="calibrated")
    args = parser.parse_args()
    prediction = json.loads((PRED / "results.json").read_text())
    replay = json.loads((args.replay_dir / "results.json").read_text())
    assert replay.get("tick_map", "calibrated") == args.tick_map
    pred_days = sorted({r["date"] for r in prediction["daily"]})
    assert pred_days == replay["test_dates"]
    assert len(pred_days) == 19
    pred_comparison = {}
    for horizon in (5, 20):
        for subset in ("all", "l3_available"):
            selected = [r for r in prediction["daily"] if r["horizon"] == horizon
                        and (subset == "all" or r["l3_available"])]
            correct, counts = defaultdict(list), []
            for row in selected:
                assert all(d < row["date"] for d in row["prior_dates"])
                with np.load(PRED / "per_day" /
                             f"{row['date']}_h{horizon}.npz", allow_pickle=False) as a:
                    y = a["actual"]
                    counts.append(len(y))
                    for name in ("deeplob", "old_hybrid", "new_l3_hybrid"):
                        p = a[f"p_{name}"]
                        assert p.shape == (len(y), 3)
                        assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
                        z = np.array([-1, 0, 1])[np.argmax(p, axis=1)]
                        correct[name].append(int(np.sum(z == y)))
                    if not row["l3_available"]:
                        assert np.array_equal(a["p_old_hybrid"],
                                              a["p_new_l3_hybrid"])
                with np.load(PRED / "signals" / f"day_{row['date']}.npz",
                             allow_pickle=False) as a:
                    if not row["l3_available"]:
                        suffix = ("_hard" if args.tick_map == "hard_one_tick" else "")
                        assert np.array_equal(a[f"expected_old_hybrid{suffix}_ticks"],
                                              a[f"expected_new_l3_hybrid{suffix}_ticks"])
                        assert np.array_equal(a[f"prior_old_hybrid{suffix}_ticks"],
                                              a[f"prior_new_l3_hybrid{suffix}_ticks"])
                    if args.tick_map == "hard_one_tick":
                        for model in ("deeplob", "old_hybrid", "new_l3_hybrid"):
                            assert np.isin(a[f"expected_{model}_hard_ticks"],
                                           (-1.0, 0.0, 1.0)).all()
                            assert np.isin(a[f"prior_{model}_hard_ticks"],
                                           (-1.0, 0.0, 1.0)).all()
                            if horizon == 20:
                                with np.load(PRED / "per_day" /
                                             f"{row['date']}_h20.npz",
                                             allow_pickle=False) as scores:
                                    expected = np.array([-1.0, 0.0, 1.0])[
                                        np.argmax(scores[f"p_{model}"], axis=1)
                                    ]
                                assert np.array_equal(
                                    a[f"expected_{model}_hard_ticks"][
                                        a["has_100_snapshot_history"]
                                    ], expected,
                                )
            counts = np.asarray(counts)
            pred_comparison[f"h{horizon}_{subset}"] = {
                "old_minus_deeplob": bootstrap_delta(
                    np.asarray(correct["old_hybrid"]),
                    np.asarray(correct["deeplob"]), counts),
                "new_minus_old": bootstrap_delta(
                    np.asarray(correct["new_l3_hybrid"]),
                    np.asarray(correct["old_hybrid"]), counts),
            }
    grouped = defaultdict(dict)
    for row in replay["daily"]:
        key = (row["date"], row["family"], row["fill_mode"], row["gamma"])
        assert row["model"] not in grouped[key]
        grouped[key][row["model"]] = row
        assert abs(row["gross_hkd"] - row["fees_hkd"] - row["net_hkd"]) < 1e-6
        assert all(d < row["date"] for d in row["prior_dates"])
    assert len(grouped) == 19 * 2 * 3
    for key, by_model in grouped.items():
        assert set(by_model) == {"no_drift", "deeplob", "old_hybrid",
                                 "new_l3_hybrid"}
        if not by_model["old_hybrid"]["l3_available"]:
            for field in ("net_hkd", "gross_hkd", "fees_hkd", "maker_fills"):
                assert by_model["old_hybrid"][field] == by_model["new_l3_hybrid"][field]
        if key[1] == "GP":
            assert len({r["quote_decisions"] for r in by_model.values()}) == 1
        else:
            assert len({r["quote_decisions"] for r in by_model.values()}) == 1
            reference = grouped[(key[0], "GP", key[2], 5.0)]
            assert by_model["no_drift"]["quote_decisions"] == reference[
                "no_drift"]["quote_decisions"]
    legacy = json.loads((HERE / "output/new_hybrid_month/results.json").read_text())
    old_baseline = {(r["family"], r["fill_mode"], r["gamma"]):
                    r["baseline_net_hkd"] for r in legacy["summary"]
                    if r["subset"] == "all"}
    for (family, mode, gamma), expected in old_baseline.items():
        actual = sum(by_model["no_drift"]["net_hkd"]
                     for key, by_model in grouped.items()
                     if key[1:] == (family, mode, gamma))
        assert abs(actual - expected) < 1e-6, (family, mode, gamma, actual, expected)
    economic = {}
    for family in ("AS", "GP"):
        for mode in ("through", "touch"):
            for gamma in ((None,) if family == "AS" else (5.0, 3.125)):
                selected = [(key, value) for key, value in grouped.items()
                            if key[1:] == (family, mode, gamma)]
                for subset in ("all", "l3_available"):
                    cases = [(key, value) for key, value in selected
                             if subset == "all" or value["old_hybrid"]["l3_available"]]
                    diffs = {label: np.asarray([
                        value[model]["net_hkd"] - value[base]["net_hkd"]
                        for _, value in cases
                    ]) for label, model, base in (
                        ("old_vs_deeplob", "old_hybrid", "deeplob"),
                        ("new_vs_old", "new_l3_hybrid", "old_hybrid"),
                        ("new_vs_no_drift", "new_l3_hybrid", "no_drift"),
                    )}
                    economic[f"{family}_{mode}_{gamma}_{subset}"] = {
                        name: {"net_delta_hkd": float(v.sum()),
                               "day_bootstrap_95_hkd": np.quantile(
                                   v[np.random.default_rng(29).integers(
                                       0, len(v), size=(20_000, len(v)))].sum(axis=1),
                                   [0.025, 0.975]).tolist(),
                               "positive_days": int(np.sum(v > 1e-8)),
                               "negative_days": int(np.sum(v < -1e-8)),
                               "equal_days": int(np.sum(np.abs(v) <= 1e-8))}
                        for name, v in diffs.items()
                    }
    output = {"prediction_day_bootstrap": pred_comparison,
              "economic_paired_days": economic,
              "checks": ("train dates prior; matched quotes; probabilities; "
                         "exact L3 fallback; PnL identities; GP quote counts; "
                         "old no-drift baseline reconciliation"
                         + ("; hard -1/0/+1 argmax mapping"
                            if args.tick_map == "hard_one_tick" else ""))}
    (args.replay_dir / "validation.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
