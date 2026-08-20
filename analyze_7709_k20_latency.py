#!/usr/bin/env python3
"""Compare k=20 snapshot horizon time with DeepLOB forward latency on HK 07709."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_7709_deeplob import DayData, date_from_path, prepare_day
from train_7709_hybrid_deeplob import HybridDeepLOB
from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
K = 20
SEQUENCE_LENGTH = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path.home() / "Desktop/7709_tickdata")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed/7709")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints/7709")
    parser.add_argument("--security-id", type=int, default=7709)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/results/7709_k20_latency.json",
    )
    return parser.parse_args()


def packed_tod_to_ms(send_time: np.ndarray) -> np.ndarray:
    tod = send_time.astype(np.int64) % 1_000_000_000
    hour = tod // 10_000_000
    minute = (tod % 10_000_000) // 100_000
    second = (tod % 100_000) // 1_000
    millis = tod % 1_000
    return ((hour * 60 + minute) * 60 + second) * 1_000 + millis


def horizon_durations_ms(send_times: np.ndarray, sessions: np.ndarray, horizon: int) -> np.ndarray:
    clock = packed_tod_to_ms(send_times)
    chunks = []
    for session in np.unique(sessions):
        index = np.flatnonzero(sessions == session)
        if len(index) <= horizon:
            continue
        delta = clock[index[horizon:]] - clock[index[:-horizon]]
        chunks.append(delta[delta > 0])
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def summarise_ms(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def load_days(args: argparse.Namespace) -> list[DayData]:
    days: list[DayData] = []
    for raw_path in sorted(args.raw_dir.glob("hk07709_*.csv"), key=date_from_path):
        try:
            prepared_path, metadata = prepare_day(
                raw_path, args.processed_dir, args.snapshot_every, args.security_id, False
            )
        except RuntimeError as error:
            print(f"Skipping {raw_path.name}: {error}", flush=True)
            continue
        archive = np.load(prepared_path, allow_pickle=False)
        days.append(
            DayData(
                date_from_path(raw_path),
                prepared_path,
                archive["features"].astype(np.float32, copy=False),
                archive["sessions"].astype(np.uint8, copy=False),
                {**metadata, "send_times": archive["send_times"].astype(np.int64, copy=False)},
            )
        )
    return days


def reconstruction_ms_per_horizon(days: list[DayData], horizon: int) -> dict[str, float]:
    records = []
    for day in days:
        duration = day.metadata.get("duration_seconds")
        count = int(day.metadata.get("snapshot_count") or len(day.features))
        if duration is None or count <= 0:
            continue
        records.append(
            {
                "date": day.date,
                "snapshot_count": count,
                "reconstruct_seconds": float(duration),
                "ms_per_snapshot": 1000.0 * float(duration) / count,
                "ms_per_k_snapshots": 1000.0 * float(duration) / count * horizon,
            }
        )
    if not records:
        return {"days": []}
    total_seconds = sum(item["reconstruct_seconds"] for item in records)
    total_snapshots = sum(item["snapshot_count"] for item in records)
    return {
        "days": records,
        "ms_per_snapshot": 1000.0 * total_seconds / total_snapshots,
        "ms_per_k_snapshots": 1000.0 * total_seconds / total_snapshots * horizon,
    }


def benchmark_forward(
    model: torch.nn.Module,
    example: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    hybrid: bool,
    aux: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    example = example.to(device)
    if aux is not None:
        aux = aux.to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            if hybrid:
                model(example, aux)
            else:
                model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            if hybrid:
                model(example, aux)
            else:
                model(example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "batch_size": int(example.shape[0]),
        **summarise_ms(values),
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    print(f"device={device}", flush=True)
    days = load_days(args)
    if not days:
        raise FileNotFoundError("No reconstructed 07709 days")

    per_day = []
    all_k = []
    all_one = []
    for day in days:
        send_times = np.asarray(day.metadata["send_times"], dtype=np.int64)
        k_ms = horizon_durations_ms(send_times, day.sessions, K)
        one_ms = horizon_durations_ms(send_times, day.sessions, 1)
        all_k.append(k_ms)
        all_one.append(one_ms)
        record = {
            "date": day.date,
            "snapshots": int(len(day.features)),
            "one_snapshot": summarise_ms(one_ms),
            "k20_snapshots": summarise_ms(k_ms),
        }
        per_day.append(record)
        k_summary = record["k20_snapshots"]
        print(
            f"{day.date}: snapshots={record['snapshots']:,}, "
            f"k=20 mean={k_summary.get('mean_ms', 0):.1f} ms, "
            f"median={k_summary.get('median_ms', 0):.1f} ms",
            flush=True,
        )

    k_all = np.concatenate(all_k) if all_k else np.empty(0)
    one_all = np.concatenate(all_one) if all_one else np.empty(0)

    checkpoint = torch.load(
        args.checkpoint_dir / "deeplob_7709_k20_best.pt",
        map_location=device,
        weights_only=True,
    )
    model = DeepLOB().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    book = torch.from_numpy(days[-1].features[-SEQUENCE_LENGTH:]).unsqueeze(0).unsqueeze(0)
    print("Benchmarking DeepLOB batch=1 forward...", flush=True)
    deeplob_forward = benchmark_forward(model, book, device, args.warmup, args.repeats, False)
    print(
        f"DeepLOB forward: mean={deeplob_forward['mean_ms']:.3f} ms, "
        f"median={deeplob_forward['median_ms']:.3f} ms",
        flush=True,
    )

    hybrid_ckpt = torch.load(
        args.checkpoint_dir / "deeplob_7709_hybrid_k20_best.pt",
        map_location=device,
        weights_only=True,
    )
    hybrid = HybridDeepLOB(int(hybrid_ckpt["preprocessing"]["auxiliary_features"])).to(device)
    hybrid.load_state_dict(hybrid_ckpt["model_state_dict"])
    aux = torch.zeros((1, int(hybrid_ckpt["preprocessing"]["auxiliary_features"])), dtype=torch.float32)
    print("Benchmarking HybridDeepLOB batch=1 forward...", flush=True)
    hybrid_forward = benchmark_forward(hybrid, book, device, args.warmup, args.repeats, True, aux)
    print(
        f"Hybrid forward: mean={hybrid_forward['mean_ms']:.3f} ms, "
        f"median={hybrid_forward['median_ms']:.3f} ms",
        flush=True,
    )

    reconstruction = reconstruction_ms_per_horizon(days, K)
    k_summary = summarise_ms(k_all)
    result = {
        "experiment": "HK 07709 k=20 snapshot horizon vs neural forward latency",
        "device": str(device),
        "k": K,
        "snapshot_every_book_update_groups": args.snapshot_every,
        "sequence_length": SEQUENCE_LENGTH,
        "exchange_time": {
            "one_snapshot": summarise_ms(one_all),
            "k20_snapshots": k_summary,
            "by_day": per_day,
        },
        "reconstruction_cpu": reconstruction,
        "forward": {
            "deeplob_k20_batch1": deeplob_forward,
            "hybrid_k20_batch1": hybrid_forward,
        },
        "comparison": {
            "k20_exchange_mean_ms": k_summary.get("mean_ms"),
            "k20_exchange_median_ms": k_summary.get("median_ms"),
            "deeplob_forward_mean_ms": deeplob_forward["mean_ms"],
            "hybrid_forward_mean_ms": hybrid_forward["mean_ms"],
            "deeplob_forward_share_of_k20_median": (
                deeplob_forward["mean_ms"] / k_summary["median_ms"]
                if k_summary.get("median_ms")
                else None
            ),
            "hybrid_forward_share_of_k20_median": (
                hybrid_forward["mean_ms"] / k_summary["median_ms"]
                if k_summary.get("median_ms")
                else None
            ),
        },
        "note": (
            "Exchange time is SendTime spanned by 20 consecutive reconstructed snapshots "
            "inside one continuous session. Forward times are batch=1 inference on CPU/GPU "
            "after warmup. Reconstruction ms/k is wall time of the offline book builder "
            "scaled to 20 snapshots, not live matching-engine latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
