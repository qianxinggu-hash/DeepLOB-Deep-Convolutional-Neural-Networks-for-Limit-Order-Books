#!/usr/bin/env python3
"""Rolling July endpoint forecasts from frozen DeepLOB plus L3 lifecycles.

The pre-July DeepLOB encoder is frozen.  A Ridge head fits each test date on
up to five earlier complete MBO dates.  The book-only head uses the same 64
encoder coordinates and 28 causal L2 summaries as the fused head; the latter
adds three OrderID lifecycle coordinates.  On feed-gap dates we use the
book-only forecast, since order identity/age cannot be audited after a gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "0824"))
sys.path.insert(0, str(ROOT / "0906"))

from hybrid_drift_adapter import HybridDriftAdapter  # noqa: E402
from run_direct_drift_signal import (  # noqa: E402
    RIDGE_ALPHA, decile_spread, prepare_day, scalar_metrics,
)
from run_multiday_strategy import load_day  # noqa: E402
from run_hybrid_month_experiment import verify_checkpoint  # noqa: E402
from l3_order_signal import extract_l3_features  # noqa: E402


MANIFEST = HERE / "output/gap_2m_books/filtered_manifest.json"
CHECKPOINT = HERE / "output/hybrid_month/pre_july_hybrid.pt"
DEFAULT_OUTPUT = HERE / "output/new_hybrid_signal"
PRICE_COLUMNS = np.arange(0, 40, 2)
SIZE_COLUMNS = np.arange(1, 40, 2)
CLIP_TICKS = 3.0


def embedding_for_day(adapter: HybridDriftAdapter, day: object,
                      indices: np.ndarray, batch_size: int) -> np.ndarray:
    model_book = day.book.astype(np.float32, copy=True)
    model_book[:, PRICE_COLUMNS] /= 100.0
    model_book[:, SIZE_COLUMNS] /= 100_000.0
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            windows = np.stack([
                adapter._transform_window(
                    model_book[int(i) - adapter.sequence_length + 1:int(i) + 1],
                    adapter.book_mean, adapter.book_std,
                ) for i in selected
            ])
            tensor = torch.from_numpy(windows).unsqueeze(1).to(adapter.device)
            encoded = adapter.model.lob_embedding(tensor)
            output.append(encoded.cpu().numpy().astype(np.float32))
    return np.concatenate(output) if output else np.empty((0, 64), np.float32)


def model() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def prediction(model_object: object, features: np.ndarray) -> np.ndarray:
    return np.clip(model_object.predict(features), -CLIP_TICKS, CLIP_TICKS)


def score_rows(rows: list[dict], name: str) -> dict:
    if not rows:
        return {"days": 0, "quotes": 0}
    y = np.concatenate([row["y"] for row in rows])
    p = np.concatenate([row[name] for row in rows])
    return {"days": len(rows), "quotes": int(len(y)),
            **scalar_metrics(y, p), **decile_spread(y, p)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = args.output_dir / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    l3_dir = args.output_dir / "l3_safe"
    l3_dir.mkdir(exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    dates = manifest["date_order"]
    checkpoint_info = verify_checkpoint(CHECKPOINT, dates[0])
    checkpoint_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    adapter = HybridDriftAdapter(CHECKPOINT, args.device)
    data: dict[str, dict] = {}
    for date in dates:
        path = ROOT / manifest["book_cache_by_date"][date]
        prepared = prepare_day(date, path, safe_eligibility=True)
        book_day = load_day(date, "tolerant_mbo", path)
        valid = prepared.has_100_snapshot_history
        indices = prepared.indices[valid]
        cache_path = embedding_dir / f"{date}.npz"
        if cache_path.exists() and not args.rebuild_embeddings:
            with np.load(cache_path, allow_pickle=False) as archive:
                if not np.array_equal(archive["quote_indices"], indices):
                    raise ValueError(f"stale quote-index embedding cache for {date}")
                if str(archive["checkpoint_sha256"]) != checkpoint_hash:
                    raise ValueError(f"stale checkpoint embedding cache for {date}")
                embedding = archive["embedding"].copy()
        else:
            embedding = embedding_for_day(adapter, book_day, indices, args.batch_size)
            np.savez_compressed(cache_path, quote_indices=indices,
                                embedding=embedding,
                                checkpoint_sha256=np.asarray(checkpoint_hash))
        if embedding.shape != (len(indices), 64) or not np.isfinite(embedding).all():
            raise ValueError(f"invalid DeepLOB embedding on {date}")
        l2 = np.column_stack((embedding, prepared.features)).astype(np.float64)
        l3 = None
        if prepared.zero_recorded_errors:
            l3_path = l3_dir / f"{date}.npz"
            if l3_path.exists():
                with np.load(l3_path, allow_pickle=False) as archive:
                    if not np.array_equal(archive["quote_indices"], prepared.indices):
                        raise ValueError(f"stale safe L3 quote starts on {date}")
                    full_l3 = archive["l3_features"].copy()
            else:
                raw = Path.home() / f"Downloads/7709_202607/hk07709_{date}.csv"
                full_l3, diagnostics = extract_l3_features(
                    raw, book_day.send_times[prepared.indices], book_day.book[prepared.indices]
                )
                np.savez_compressed(l3_path, quote_indices=prepared.indices,
                                    l3_features=full_l3,
                                    diagnostics=np.asarray(json.dumps(diagnostics)))
            l3 = full_l3[valid, 1:4].astype(np.float64)
            if not np.isfinite(l3).all():
                raise ValueError(f"nonfinite L3 lifecycles on {date}")
        data[date] = {
            "prepared": prepared, "l2": l2, "l3": l3,
            "target": prepared.actual_move_ticks[valid], "valid": valid,
        }
        print("PREPARED", date, "valid", len(indices),
              "L3", l3 is not None, flush=True)

    daily = []
    scored_rows = []
    for ordinal, date in enumerate(dates[1:], start=1):
        prior = [d for d in dates[:ordinal] if data[d]["l3"] is not None][-5:]
        if not prior:
            raise ValueError(f"{date}: no prior complete L3 date")
        training_y = np.concatenate([data[d]["target"] for d in prior])
        book_model = model()
        book_model.fit(np.concatenate([data[d]["l2"] for d in prior]), training_y)
        fused_model = model()
        fused_model.fit(np.concatenate([
            np.column_stack((data[d]["l2"], data[d]["l3"])) for d in prior
        ]), training_y)
        current = data[date]
        book_prediction = prediction(book_model, current["l2"])
        fused_prediction = (
            prediction(fused_model, np.column_stack((current["l2"], current["l3"])))
            if current["l3"] is not None else book_prediction.copy()
        )
        prepared = current["prepared"]
        valid = current["valid"]
        full_book = np.zeros(len(prepared.indices), dtype=np.float64)
        full_fused = np.zeros(len(prepared.indices), dtype=np.float64)
        full_book[valid] = book_prediction
        full_fused[valid] = fused_prediction
        # Current fitted heads on completed prior dates provide a consistent
        # prior-only drift-state calibration set for the subsequent QVI.
        prior_dates = []
        prior_preds = []
        prior_book_preds = []
        prior_seconds = []
        prior_ticks = []
        prior_segments = []
        prior_quote_indices = []
        prior_quote_counts = []
        prior_intervals = []
        for d in prior:
            item = data[d]
            p = item["prepared"]
            q = np.zeros(len(p.indices), dtype=np.float64)
            q_book = np.zeros(len(p.indices), dtype=np.float64)
            q[item["valid"]] = prediction(
                fused_model, np.column_stack((item["l2"], item["l3"]))
            )
            q_book[item["valid"]] = prediction(book_model, item["l2"])
            prior_dates.append(d)
            prior_preds.append(q)
            prior_book_preds.append(q_book)
            prior_seconds.append(p.causal_horizon_seconds)
            prior_ticks.append(np.full(len(q), p.tick_hkd))
            prior_segments.append(p.segment_ids)
            prior_quote_indices.append(p.indices)
            prior_quote_counts.append(len(q))
            prior_intervals.append(p.quote_intervals_seconds)
        cache = prediction_dir / f"day_{date}.npz"
        np.savez_compressed(
            cache,
            quote_indices=prepared.indices,
            has_100_snapshot_history=valid,
            expected_move_ticks=full_fused,
            book_only_expected_move_ticks=full_book,
            causal_horizon_seconds=prepared.causal_horizon_seconds,
            actual_move_ticks_diagnostic=prepared.actual_move_ticks,
            tick_hkd=np.asarray(prepared.tick_hkd),
            l3_available=np.asarray(current["l3"] is not None),
            prior_dates=np.asarray(prior_dates),
            prior_quote_indices=np.concatenate(prior_quote_indices),
            prior_expected_move_ticks=np.concatenate(prior_preds),
            prior_book_only_expected_move_ticks=np.concatenate(prior_book_preds),
            prior_causal_horizon_seconds=np.concatenate(prior_seconds),
            prior_tick_hkd=np.concatenate(prior_ticks),
            prior_segment_ids=np.concatenate(prior_segments),
            prior_day_ordinals=np.concatenate([
                np.full(len(prior_preds[i]), i, dtype=np.int16)
                for i in range(len(prior))
            ]),
            prior_day_quote_counts=np.asarray(prior_quote_counts, dtype=np.int32),
            prior_quote_intervals_seconds=np.concatenate(prior_intervals),
        )
        y = current["target"]
        row = {
            "date": date, "prior_dates": prior,
            "l3_available": current["l3"] is not None,
            "quotes": int(len(prepared.indices)), "scored_quotes": int(len(y)),
            "book_only": {**scalar_metrics(y, book_prediction),
                          **decile_spread(y, book_prediction)},
            "new_hybrid": {**scalar_metrics(y, fused_prediction),
                           **decile_spread(y, fused_prediction)},
        }
        b_mse = float(np.mean(np.square(y - book_prediction)))
        f_mse = float(np.mean(np.square(y - fused_prediction)))
        row["mse_reduction_vs_book_pct"] = 100 * (b_mse - f_mse) / b_mse
        daily.append(row)
        scored_rows.append({"date": date, "clean": row["l3_available"],
                            "y": y, "book": book_prediction, "fused": fused_prediction})
        print("SCORED", date, "L3", row["l3_available"],
              "MSE lift", round(row["mse_reduction_vs_book_pct"], 4), flush=True)

    result = {
        "name": "new hybrid",
        "architecture": "frozen pre-July DeepLOB 64-d encoder + 28 causal L2 summaries; Ridge endpoint head; fused head appends three OrderID lifecycle features",
        "checkpoint": checkpoint_info,
        "checkpoint_sha256": checkpoint_hash,
        "target": "(mid[t+20]-mid[t])/tick_hkd, numeric endpoint ticks",
        "train": "at most five prior complete MBO dates, StandardScaler and Ridge alpha=1000",
        "clip_prediction_ticks": CLIP_TICKS,
        "gap_policy": "2-minute blackout book; on any date with recorded gaps/order errors, no L3 and exact book-only forecast fallback",
        "dates": dates,
        "daily": daily,
        "pooled": {
            subset: {
                name: score_rows(
                    [r for r in scored_rows if subset == "all" or
                     (subset == "l3_available" and r["clean"]) or
                     (subset == "l3_unavailable" and not r["clean"])], name
                ) for name in ("book", "fused")
            } for subset in ("all", "l3_available", "l3_unavailable")
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
