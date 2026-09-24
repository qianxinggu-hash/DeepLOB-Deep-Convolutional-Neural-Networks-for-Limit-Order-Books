#!/usr/bin/env python3
"""Train and score the aligned Hybrid probability heads for a later session.

This module only emits probabilities and a confidence gate. A caller supplies
causal features for each snapshot and retains responsibility for order state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from hybrid_labeling import CLASSES, prior_flat_threshold
from run_aligned_direction_comparison import DATA, FEATURES, fit, x_of


HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "output/hybrid_month/pre_july_hybrid.pt"
DEFAULT_BUNDLE = HERE / "output/hybrid_signal_bundle/balanced_h20.joblib"


def train_bundle(available_through: str, data_dir: Path = DATA) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows = [row for row in manifest["dates"] if row["date"] <= available_through]
    if not rows or rows[-1]["date"] != available_through:
        raise ValueError("available_through must be a date in the dataset")
    prior = [row["date"] for row in rows if row["l3_available"]][-5:]
    if not prior:
        raise ValueError("no complete L3 training dates")
    data = {}
    for date in prior:
        with np.load(data_dir / f"day_{date}.npz", allow_pickle=False) as archive:
            data[date] = {key: archive[key].copy() for key in archive.files}
    train = [data[date] for date in prior]
    moves = np.concatenate([day["move_20_ticks"] for day in train])
    threshold = prior_flat_threshold(moves)
    models = {name: fit(train, 20, name, threshold) for name in FEATURES}
    return {
        "version": 1,
        "available_through": available_through,
        "train_dates": prior,
        "horizon_snapshots": 20,
        "label_threshold_ticks": threshold,
        "class_order": CLASSES.tolist(),
        "confidence_threshold": 0.9,
        "feature_dimensions": {"embedding": 64, "l2": 92, "l3": 6},
        "encoder_checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "models": models,
    }


def score_batch(bundle: dict, l2: np.ndarray,
                l3: np.ndarray | None = None,
                confidence_threshold: float | None = None
                ) -> dict[str, dict[str, np.ndarray]]:
    """Score one or more causal snapshots; fall back to old Hybrid without L3."""
    threshold = (bundle["confidence_threshold"] if confidence_threshold is None
                 else confidence_threshold)
    if not 0 < threshold < 1:
        raise ValueError("confidence_threshold must be between zero and one")
    l2 = np.asarray(l2, dtype=np.float64)
    if l2.ndim != 2 or l2.shape[1] != 92 or not np.isfinite(l2).all():
        raise ValueError("l2 must be a finite N x 92 feature matrix")
    if l3 is not None:
        l3 = np.asarray(l3, dtype=np.float64)
        if l3.shape == (len(l2), 0):
            l3 = None
        elif l3.shape != (len(l2), 6) or not np.isfinite(l3).all():
            raise ValueError("l3 must be a finite N x 6 feature matrix")
    x = {"deeplob": l2[:, :64], "old_hybrid": l2}
    if l3 is not None:
        x["new_l3_hybrid"] = np.column_stack((l2, l3))
    result = {}
    for name in FEATURES:
        if name == "new_l3_hybrid" and l3 is None:
            probabilities = result["old_hybrid"]["probabilities"].copy()
        else:
            probabilities = bundle["models"][name].predict_proba(x[name])
        if probabilities.shape != (len(l2), 3) or not np.allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError("invalid three-class probabilities")
        chosen = CLASSES[np.argmax(probabilities, axis=1)]
        confidence = np.max(probabilities, axis=1)
        passes_gate = confidence > threshold
        result[name] = {
            "probabilities": probabilities,
            "class": chosen,
            "confidence": confidence,
            "passes_confidence_gate": passes_gate,
            "directional_quote_candidate": passes_gate & (chosen != 0),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--available-through", required=True)
    train.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    score = sub.add_parser("score")
    score.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    score.add_argument("--features", type=Path, required=True,
                       help="NPZ with l2 (N x 92) and optional l3 (N x 6)")
    score.add_argument("--confidence-threshold", type=float,
                       help="override the bundle's confidence threshold")
    args = parser.parse_args()
    if args.command == "train":
        bundle = train_bundle(args.available_through)
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.bundle)
        print(json.dumps({key: value for key, value in bundle.items()
                          if key != "models"}, indent=2, ensure_ascii=False))
    else:
        bundle = joblib.load(args.bundle)
        with np.load(args.features, allow_pickle=False) as archive:
            result = score_batch(bundle, archive["l2"],
                                 archive["l3"] if "l3" in archive.files else None,
                                 args.confidence_threshold)
        print(json.dumps({"confidence_threshold": (
            bundle["confidence_threshold"] if args.confidence_threshold is None
            else args.confidence_threshold), "models": {name: {
            "probabilities": value["probabilities"].tolist(),
            "class": value["class"].tolist(),
            "confidence": value["confidence"].tolist(),
            "passes_confidence_gate": value["passes_confidence_gate"].tolist(),
            "directional_quote_candidate": value["directional_quote_candidate"].tolist(),
        } for name, value in result.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
