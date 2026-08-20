#!/usr/bin/env python3
"""Frozen 07709 checkpoint classification reports on train and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn

from analyze_7709_causal_baselines import causal_features as logistic_causal_features
from train_7709_deeplob import (
    CLASS_NAMES,
    DayData,
    PreparedDay,
    date_from_path,
    eligible_targets,
    equation4_returns,
    make_loader,
    prepare_day,
    run_epoch,
)
from train_7709_hybrid_deeplob import (
    HybridDeepLOB,
    PreparedDay as HybridPreparedDay,
    causal_features as hybrid_causal_features,
    make_loader as make_hybrid_loader,
    run_epoch as run_hybrid_epoch,
)
from train_pytorch_mac import DeepLOB, choose_device


PROJECT_ROOT = Path(__file__).resolve().parent
VALIDATION_DATE = "2026-08-04"
TEST_DATE = "2026-08-07"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path.home() / "Desktop/7709_tickdata")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed/7709")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints/7709")
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--security-id", type=int, default=7709)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/results/7709_checkpoint_split_reports.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "output/report/7709_checkpoint_performance.md",
    )
    return parser.parse_args()


def report_dict(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return classification_report(
        truth,
        prediction,
        labels=[0, 1, 2],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
        digits=4,
    )


def report_from_confusion(metrics: dict[str, object]) -> dict[str, object]:
    truth_pred = []
    prediction_pred = []
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    for truth, row in enumerate(matrix):
        for pred, count in enumerate(row):
            if count:
                truth_pred.append(np.full(count, truth, dtype=np.int64))
                prediction_pred.append(np.full(count, pred, dtype=np.int64))
    return report_dict(np.concatenate(truth_pred), np.concatenate(prediction_pred))


def print_split(title: str, report: dict[str, object]) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(f"{'':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}", flush=True)
    for name in CLASS_NAMES:
        row = report[name]
        print(
            f"{name:<12} {row['precision']:10.4f} {row['recall']:10.4f} "
            f"{row['f1-score']:10.4f} {int(row['support']):10d}",
            flush=True,
        )
    print(
        f"{'accuracy':<12} {'':>10} {'':>10} {report['accuracy']:10.4f} "
        f"{int(report['macro avg']['support']):10d}",
        flush=True,
    )
    macro = report["macro avg"]
    print(
        f"{'macro avg':<12} {macro['precision']:10.4f} {macro['recall']:10.4f} "
        f"{macro['f1-score']:10.4f} {int(macro['support']):10d}",
        flush=True,
    )


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def markdown_table(report: dict[str, object]) -> str:
    lines = [
        "| 类别 | Precision | Recall | F1 | Support |",
        "|------|-----------|--------|-----|---------|",
    ]
    for name in CLASS_NAMES:
        row = report[name]
        lines.append(
            f"| {name.capitalize()} | {percent(row['precision'])} | {percent(row['recall'])} | "
            f"{percent(row['f1-score'])} | {int(row['support']):,} |"
        )
    lines.append(
        f"| Accuracy | | | {percent(report['accuracy'])} | {int(report['macro avg']['support']):,} |"
    )
    macro = report["macro avg"]
    lines.append(
        f"| Macro avg | {percent(macro['precision'])} | {percent(macro['recall'])} | "
        f"{percent(macro['f1-score'])} | {int(macro['support']):,} |"
    )
    return "\n".join(lines)


def reconstruct_days(args: argparse.Namespace) -> list[DayData]:
    raw_files = sorted(args.raw_dir.glob("hk07709_*.csv"), key=date_from_path)
    if not raw_files:
        raise FileNotFoundError(f"No hk07709_*.csv under {args.raw_dir}")
    days: list[DayData] = []
    for raw_path in raw_files:
        try:
            prepared_path, metadata = prepare_day(
                raw_path,
                args.processed_dir,
                args.snapshot_every,
                args.security_id,
                False,
            )
        except RuntimeError as error:
            print(f"Skipping {raw_path.name}: {error}", flush=True)
            continue
        archive = np.load(prepared_path, allow_pickle=False)
        date = date_from_path(raw_path)
        days.append(
            DayData(
                date,
                prepared_path,
                archive["features"].astype(np.float32, copy=False),
                archive["sessions"].astype(np.uint8, copy=False),
                metadata,
            )
        )
    return days


def labeled_items(
    days: list[DayData],
    train_dates: set[str],
    k: int,
    alpha: float,
    sequence_length: int,
    train_stride: int,
) -> dict[str, PreparedDay]:
    prepared: dict[str, PreparedDay] = {}
    for day in days:
        if day.date in train_dates:
            stride = train_stride
        elif day.date == TEST_DATE:
            stride = 1
        else:
            stride = 5
        returns = equation4_returns(day, k)
        indices = eligible_targets(day, returns, sequence_length, stride)
        values = returns[indices]
        labels = np.where(values < -alpha, 0, np.where(values > alpha, 2, 1)).astype(np.int64)
        prepared[day.date] = PreparedDay(day, indices, labels)
    return prepared


def apply_deeplob_normalizers(days: list[DayData], checkpoint: dict[str, object]) -> None:
    normalizers = checkpoint["preprocessing"]["normalizers"]
    for day in days:
        if day.date not in normalizers:
            raise KeyError(f"Checkpoint has no normalizer for {day.date}")
        record = normalizers[day.date]
        day.mean = np.asarray(record["mean"], dtype=np.float32)
        day.std = np.asarray(record["std"], dtype=np.float32)
        day.normalization_sources = tuple(record["source_dates"])


def evaluate_deeplob(
    args: argparse.Namespace,
    days: list[DayData],
    train_dates: list[str],
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    k = int(checkpoint["labeling"]["k"])
    alpha = float(checkpoint["labeling"]["alpha"])
    config = checkpoint["config"]
    train_stride = int(config.get("train_target_stride", 10))
    apply_deeplob_normalizers(days, checkpoint)
    prepared = labeled_items(days, set(train_dates), k, alpha, args.sequence_length, train_stride)
    model = DeepLOB().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    result: dict[str, object] = {
        "model": "DeepLOB",
        "checkpoint": str(checkpoint_path.resolve()),
        "k": k,
        "alpha": alpha,
        "train_stride": train_stride,
        "best_epoch": int(checkpoint.get("epoch", -1)),
    }
    for split, dates in (("train", train_dates), ("test", [TEST_DATE])):
        loader = make_loader(
            [prepared[date] for date in dates],
            args.sequence_length,
            args.batch_size,
            False,
            0,
            42,
        )
        loss, metrics = run_epoch(model, loader, criterion, device)
        report = report_from_confusion(metrics)
        result[split] = {
            "samples": int(metrics["samples"]),
            "loss": float(loss),
            "classification_report": report,
        }
        print_split(f"DeepLOB k={k} {split}", report)
    return result


def evaluate_hybrid(
    args: argparse.Namespace,
    days: list[DayData],
    train_dates: list[str],
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    k = int(checkpoint["labeling"]["k"])
    alpha = float(checkpoint["labeling"]["alpha"])
    config = checkpoint["config"]
    train_stride = int(config.get("train_stride", 5))
    preprocessing = checkpoint["preprocessing"]
    book_mean = np.asarray(preprocessing["book_mean"], dtype=np.float32)
    book_std = np.asarray(preprocessing["book_std"], dtype=np.float32)
    auxiliary_mean = np.asarray(preprocessing["auxiliary_mean"], dtype=np.float32)
    auxiliary_std = np.asarray(preprocessing["auxiliary_std"], dtype=np.float32)
    prepared: dict[str, HybridPreparedDay] = {}
    for day in days:
        if day.date in set(train_dates):
            stride = train_stride
        elif day.date == TEST_DATE:
            stride = 1
        else:
            stride = int(config.get("validation_stride", 5))
        returns = equation4_returns(day, k)
        indices = eligible_targets(day, returns, args.sequence_length, stride)
        values = returns[indices]
        labels = np.where(values < -alpha, 0, np.where(values > alpha, 2, 1)).astype(np.int64)
        prepared[day.date] = HybridPreparedDay(
            day, indices, labels, hybrid_causal_features(day, indices)
        )
    hybrid_args = SimpleNamespace(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
    )
    model = HybridDeepLOB(int(preprocessing["auxiliary_features"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    result: dict[str, object] = {
        "model": "HybridDeepLOB",
        "checkpoint": str(checkpoint_path.resolve()),
        "k": k,
        "alpha": alpha,
        "train_stride": train_stride,
        "best_epoch": int(checkpoint.get("epoch", -1)),
    }
    for split, dates in (("train", train_dates), ("test", [TEST_DATE])):
        loader = make_hybrid_loader(
            [prepared[date] for date in dates],
            hybrid_args,
            book_mean,
            book_std,
            auxiliary_mean,
            auxiliary_std,
            False,
        )
        loss, metrics = run_hybrid_epoch(model, loader, criterion, device)
        report = report_from_confusion(metrics)
        result[split] = {
            "samples": int(metrics["samples"]),
            "loss": float(loss),
            "classification_report": report,
        }
        print_split(f"HybridDeepLOB k={k} {split}", report)
    return result


def evaluate_logistic(
    days: list[DayData],
    train_dates: list[str],
    model_path: Path,
    k: int,
    alpha: float,
    sequence_length: int,
    train_stride: int,
) -> dict[str, object]:
    model = joblib.load(model_path)
    prepared_x: dict[str, np.ndarray] = {}
    prepared_y: dict[str, np.ndarray] = {}
    for day in days:
        if day.date in set(train_dates):
            stride = train_stride
        elif day.date == TEST_DATE:
            stride = 1
        else:
            stride = 5
        returns = equation4_returns(day, k)
        indices = eligible_targets(day, returns, sequence_length, stride)
        values = returns[indices]
        prepared_x[day.date] = logistic_causal_features(day, indices)
        prepared_y[day.date] = np.where(values < -alpha, 0, np.where(values > alpha, 2, 1)).astype(
            np.int64
        )
    result: dict[str, object] = {
        "model": "LogisticRegression",
        "checkpoint": str(model_path.resolve()),
        "k": k,
        "alpha": alpha,
        "train_stride": train_stride,
    }
    for split, dates in (("train", train_dates), ("test", [TEST_DATE])):
        features = np.concatenate([prepared_x[date] for date in dates])
        labels = np.concatenate([prepared_y[date] for date in dates])
        prediction = model.predict(features)
        report = report_dict(labels, prediction)
        result[split] = {"samples": int(len(labels)), "classification_report": report}
        print_split(f"Logistic k={k} {split}", report)
    return result


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    split = payload["split"]
    models = payload["models"]
    lines = [
        "# HK 07709：冻结 checkpoint 的 train / test classification table",
        "",
        "权重固定，不再训练。数字来自 `evaluate_7709_checkpoint_splits.py`，完整 JSON：`output/results/7709_checkpoint_split_reports.json`。",
        "",
        f"切分：训练 {', '.join(split['train_dates'])}；验证 {split['validation_date']}；测试 {split['test_date']}。",
        "标签为 DeepLOB Equation (4)。DeepLOB 的 k=50 / k=100 测试是事后诊断，不改变当初按验证 Macro-F1 选出的 k=20。",
        "",
        "| 模型 | 集合 | 样本 | Accuracy | Macro-F1 |",
        "|------|------|------|----------|----------|",
    ]
    for item in models:
        for split_name in ("train", "test"):
            report = item[split_name]["classification_report"]
            lines.append(
                f"| {item['title']} | {split_name} | {item[split_name]['samples']:,} | "
                f"{percent(report['accuracy'])} | {percent(report['macro avg']['f1-score'])} |"
            )
    lines.extend(["", "---", ""])
    for index, item in enumerate(models, start=1):
        lines.append(f"## {index}. {item['title']}")
        lines.append("")
        lines.append(f"Checkpoint：`{item['checkpoint_name']}`。k={item['k']}，α={item['alpha']:.6g}。")
        if item.get("note"):
            lines.append(item["note"])
        lines.append("")
        lines.append("### 训练集")
        lines.append("")
        lines.append(markdown_table(item["train"]["classification_report"]))
        lines.append("")
        lines.append("### 测试集")
        lines.append("")
        lines.append(markdown_table(item["test"]["classification_report"]))
        lines.append("")
        lines.append("---")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    print(f"device={device}", flush=True)
    days = reconstruct_days(args)
    dates = [day.date for day in days]
    train_dates = [date for date in dates if date < VALIDATION_DATE]
    print(f"days={dates}", flush=True)
    print(f"train={train_dates}", flush=True)

    checkpoint_dir = args.checkpoint_dir
    models_raw = [
        evaluate_deeplob(
            args, days, train_dates, device, checkpoint_dir / "deeplob_7709_k20_best.pt"
        ),
        evaluate_deeplob(
            args, days, train_dates, device, checkpoint_dir / "deeplob_7709_k50_best.pt"
        ),
        evaluate_deeplob(
            args, days, train_dates, device, checkpoint_dir / "deeplob_7709_k100_best.pt"
        ),
        evaluate_hybrid(
            args, days, train_dates, device, checkpoint_dir / "deeplob_7709_hybrid_k20_best.pt"
        ),
        evaluate_logistic(
            days,
            train_dates,
            checkpoint_dir / "7709_logistic_causal.joblib",
            k=20,
            alpha=0.00032704008003969776,
            sequence_length=args.sequence_length,
            train_stride=10,
        ),
    ]
    titles = {
        "DeepLOB": None,
        "HybridDeepLOB": "HybridDeepLOB k=20",
        "LogisticRegression": "Logistic regression k=20",
    }
    models = []
    for item in models_raw:
        title = titles[item["model"]] or f"DeepLOB k={item['k']}"
        models.append(
            {
                **item,
                "title": title,
                "checkpoint_name": Path(item["checkpoint"]).name,
                "note": (
                    "k=50 / k=100 仅作对照，当初没有用测试集选 k。"
                    if item["model"] == "DeepLOB" and int(item["k"]) != 20
                    else ""
                ),
            }
        )
    payload = {
        "experiment": "Frozen HK 07709 checkpoints on train and test",
        "device": str(device),
        "split": {
            "train_dates": train_dates,
            "validation_date": VALIDATION_DATE,
            "test_date": TEST_DATE,
        },
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.report, payload)
    print(f"Saved: {args.output.resolve()}", flush=True)
    print(f"Report: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
