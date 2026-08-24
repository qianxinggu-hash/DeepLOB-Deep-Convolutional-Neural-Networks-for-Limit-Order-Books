#!/usr/bin/env python3
"""Validation-selected, regularized single-day DeepLOB improvement experiment."""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment import (  # noqa: E402
    CLASS_NAMES,
    confusion_metrics,
    constant_baseline,
    symmetric_returns,
    transform_features,
)
from train_pytorch_mac import DeepLOB, choose_device  # noqa: E402


SEQUENCE_LENGTH = 100
OUTER_TRAIN_RATIO = 0.70
INNER_FIT_RATIO = 0.75
HORIZON_CANDIDATES = (20, 50, 100)
C_CANDIDATES = (0.01, 0.1, 1.0)
NUM_CLASSES = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interval_indices(
    returns: np.ndarray,
    segments: np.ndarray,
    interval_start: int,
    interval_stop: int,
    sequence_length: int,
    horizon: int,
) -> np.ndarray:
    """Targets whose complete input, past label, and future label stay in interval."""

    selected: list[np.ndarray] = []
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        segment_start = int(positions[0])
        segment_stop = int(positions[-1]) + 1
        start = max(segment_start, interval_start)
        stop = min(segment_stop, interval_stop)
        first_target = start + max(sequence_length, horizon) - 1
        target_stop = stop - horizon
        if first_target >= target_stop:
            continue
        candidates = np.arange(first_target, target_stop, dtype=np.int64)
        candidates = candidates[np.isfinite(returns[candidates])]
        if len(candidates):
            selected.append(candidates)
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def make_labels(returns: np.ndarray, indices: np.ndarray, alpha: float) -> np.ndarray:
    values = returns[indices]
    return np.where(values < -alpha, 0, np.where(values > alpha, 2, 1)).astype(np.int64)


def causal_features(book: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """25 features known at the prediction timestamp; no future information."""

    mid = (book[:, 0].astype(np.float64) + book[:, 2]) / 2
    columns: list[np.ndarray] = []
    for horizon in (1, 2, 5, 10, 20, 50, 100):
        columns.append((mid[indices] / mid[indices - horizon] - 1) * 10_000)
    cumulative = np.concatenate(([0.0], np.cumsum(mid, dtype=np.float64)))
    for horizon in (5, 20, 50):
        trailing = (cumulative[indices + 1] - cumulative[indices - horizon + 1]) / horizon
        columns.append((mid[indices] / trailing - 1) * 10_000)
    columns.append((book[indices, 0] - book[indices, 2]) / mid[indices] * 10_000)
    asks = book[indices][:, np.arange(1, 40, 4)].astype(np.float64)
    bids = book[indices][:, np.arange(3, 40, 4)].astype(np.float64)
    for levels in (1, 2, 3, 5, 10):
        ask_depth = asks[:, :levels].sum(axis=1)
        bid_depth = bids[:, :levels].sum(axis=1)
        columns.append((bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12))
    for level in range(5):
        columns.append(
            (bids[:, level] - asks[:, level])
            / (bids[:, level] + asks[:, level] + 1e-12)
        )
    columns.extend(
        [
            (book[indices, 4] - book[indices, 0]) / mid[indices] * 10_000,
            (book[indices, 2] - book[indices, 6]) / mid[indices] * 10_000,
            np.log1p(asks.sum(axis=1)),
            np.log1p(bids.sum(axis=1)),
        ]
    )
    result = np.column_stack(columns).astype(np.float32)
    if result.shape[1] != 25 or not np.isfinite(result).all():
        raise ValueError("invalid causal feature matrix")
    return result


def metrics_from_prediction(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    confusion = np.bincount(
        truth * NUM_CLASSES + prediction, minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    result = confusion_metrics(confusion)
    result["prediction_counts"] = np.bincount(prediction, minlength=NUM_CLASSES).tolist()
    return result


def label_summary(labels: np.ndarray) -> dict[str, object]:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {"counts": counts.tolist(), "shares": (counts / counts.sum()).tolist()}


@dataclass
class PreparedPartition:
    indices: np.ndarray
    labels: np.ndarray
    auxiliary: np.ndarray


class HybridWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        book: np.ndarray,
        partition: PreparedPartition,
        auxiliary_mean: np.ndarray,
        auxiliary_std: np.ndarray,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        self.book = book
        self.partition = partition
        self.auxiliary_mean = auxiliary_mean
        self.auxiliary_std = auxiliary_std
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.partition.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = int(self.partition.indices[item])
        start = target - self.sequence_length + 1
        window = np.ascontiguousarray(self.book[start : target + 1])
        auxiliary = np.ascontiguousarray(
            (self.partition.auxiliary[item] - self.auxiliary_mean) / self.auxiliary_std
        )
        return (
            torch.from_numpy(window).unsqueeze(0),
            torch.from_numpy(auxiliary),
            torch.tensor(int(self.partition.labels[item]), dtype=torch.int64),
        )


class RegularizedDeepLOB(nn.Module):
    def __init__(self, auxiliary_features: int = 25) -> None:
        super().__init__()
        self.backbone = DeepLOB(NUM_CLASSES)
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.40), nn.Linear(64, NUM_CLASSES))

    def forward(self, book: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        del auxiliary
        return self.backbone(book)


class CausalHybridDeepLOB(nn.Module):
    def __init__(self, auxiliary_features: int = 25) -> None:
        super().__init__()
        self.backbone = DeepLOB(NUM_CLASSES)
        self.auxiliary_net = nn.Sequential(
            nn.Linear(auxiliary_features, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.40),
            nn.Linear(64, NUM_CLASSES),
        )

    def lob_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        net = self.backbone
        features = net.conv3(net.conv2(net.conv1(inputs)))
        features = torch.cat(
            (net.inception1(features), net.inception2(features), net.inception3(features)),
            dim=1,
        )
        features = features.permute(0, 2, 1, 3)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features, _ = net.lstm(features)
        return features[:, -1]

    def forward(self, book: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            torch.cat((self.lob_embedding(book), self.auxiliary_net(auxiliary)), dim=1)
        )


MODEL_FACTORIES = {
    "regularized_deeplob": RegularizedDeepLOB,
    "causal_hybrid_deeplob": CausalHybridDeepLOB,
}


def make_loader(
    book: np.ndarray,
    partition: PreparedPartition,
    auxiliary_mean: np.ndarray,
    auxiliary_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        HybridWindowDataset(book, partition, auxiliary_mean, auxiliary_std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    model.train()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    for book, auxiliary, targets in loader:
        book = book.to(device, dtype=torch.float32, non_blocking=True)
        auxiliary = auxiliary.to(device, dtype=torch.float32, non_blocking=True)
        target_device = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(book, auxiliary)
        loss = criterion(logits, target_device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        prediction = logits.argmax(dim=1).detach().cpu().numpy()
        truth = targets.numpy()
        confusion += np.bincount(
            truth * NUM_CLASSES + prediction, minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        total_loss += float(loss.item()) * len(truth)
    return total_loss / int(confusion.sum()), confusion_metrics(confusion)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    logits_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    total_loss = 0.0
    with torch.inference_mode():
        for book, auxiliary, targets in loader:
            book = book.to(device, dtype=torch.float32, non_blocking=True)
            auxiliary = auxiliary.to(device, dtype=torch.float32, non_blocking=True)
            logits = model(book, auxiliary)
            total_loss += float(criterion(logits, targets.to(device)).item()) * len(targets)
            logits_rows.append(logits.cpu().numpy())
            truth_rows.append(targets.numpy())
    logits = np.concatenate(logits_rows)
    truth = np.concatenate(truth_rows)
    metrics = metrics_from_prediction(truth, logits.argmax(axis=1))
    return total_loss / len(truth), metrics, logits, truth


def fit_with_validation(
    model_name: str,
    fit_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    seed: int,
    max_epochs: int = 15,
    patience: int = 3,
) -> dict[str, object]:
    seed_everything(seed)
    model = MODEL_FACTORIES[model_name]().to(device)
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    other = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": 3e-4},
            {"params": other, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_score = -math.inf
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_logits: np.ndarray | None = None
    best_validation_truth: np.ndarray | None = None
    history: list[dict[str, object]] = []
    unimproved = 0
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        train_loss, train_metrics = train_epoch(model, fit_loader, optimizer, criterion, device)
        val_loss, val_metrics, val_logits, val_truth = evaluate(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train": train_metrics,
                "validation": val_metrics,
                "duration_seconds": time.perf_counter() - started,
            }
        )
        score = float(val_metrics["macro_f1"])
        improved = score > best_score + 1e-8 or (
            abs(score - best_score) <= 1e-8 and val_loss < best_loss
        )
        if improved:
            best_score = score
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_validation_logits = val_logits.copy()
            best_validation_truth = val_truth.copy()
            unimproved = 0
        else:
            unimproved += 1
        print(
            f"{model_name} epoch {epoch:02d}: train_f1={train_metrics['macro_f1']:.4f}, "
            f"val_f1={val_metrics['macro_f1']:.4f}, val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        if unimproved >= patience:
            break
    if best_state is None or best_validation_logits is None or best_validation_truth is None:
        raise RuntimeError("model selection failed")
    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "best_validation_loss": best_loss,
        "best_state": best_state,
        "best_validation_logits": best_validation_logits,
        "best_validation_truth": best_validation_truth,
        "history": history,
    }


def train_fixed_epochs(
    model_name: str,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, list[dict[str, object]]]:
    seed_everything(seed)
    model = MODEL_FACTORIES[model_name]().to(device)
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    other = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 3e-4}, {"params": other, "lr": 1e-3}],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    history = []
    for epoch in range(1, epochs + 1):
        loss, metrics = train_epoch(model, loader, optimizer, criterion, device)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        print(
            f"final {model_name} epoch {epoch:02d}/{epochs}: "
            f"train_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    return model, history


def log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def select_calibration(
    neural_logits: np.ndarray,
    logistic_log_probability: np.ndarray,
    truth: np.ndarray,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    neural_log_probability = log_softmax_numpy(neural_logits)
    for neural_weight in np.linspace(0.0, 1.0, 5):
        for stationary_bias in np.linspace(-0.75, 0.75, 7):
            combined = (
                neural_weight * neural_log_probability
                + (1 - neural_weight) * logistic_log_probability
            )
            combined = combined.copy()
            combined[:, 1] += stationary_bias
            metrics = metrics_from_prediction(truth, combined.argmax(axis=1))
            record = {
                "neural_weight": float(neural_weight),
                "logistic_weight": float(1 - neural_weight),
                "stationary_logit_bias": float(stationary_bias),
                "validation": metrics,
            }
            key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if best is None or key > (
                float(best["validation"]["macro_f1"]),
                float(best["validation"]["accuracy"]),
            ):
                best = record
    if best is None:
        raise RuntimeError("calibration selection failed")
    return best


def normalized_book(features: np.ndarray, fit_stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = transform_features(features)
    mean = transformed[:fit_stop].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = transformed[:fit_stop].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return np.ascontiguousarray((transformed - mean) / std, dtype=np.float32), mean, std


def main() -> None:
    seed = 42
    batch_size = 256
    device = choose_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    cache = HERE / "data/hk07709_2026-08-07_l2_s10.npz"
    with np.load(cache, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        segments = archive["segments"].astype(np.int16, copy=False)
        send_times = archive["send_times"].astype(np.int64, copy=False)
    outer_stop = int(math.floor(len(features) * OUTER_TRAIN_RATIO))
    inner_stop = int(math.floor(outer_stop * INNER_FIT_RATIO))
    print(
        f"outer development/test={outer_stop:,}/{len(features)-outer_stop:,}; "
        f"inner fit/validation boundary={inner_stop:,}; device={device}",
        flush=True,
    )

    # Phase 1: select label horizon and logistic regularization on inner validation only.
    horizon_diagnostics = []
    selected_horizon: dict[str, object] | None = None
    selected_inner: dict[str, object] | None = None
    for horizon in HORIZON_CANDIDATES:
        returns = symmetric_returns(
            (features[:, 0].astype(np.float64) + features[:, 2]) / 2,
            segments,
            horizon,
        )
        fit_indices = interval_indices(returns, segments, 0, inner_stop, SEQUENCE_LENGTH, horizon)
        validation_indices = interval_indices(
            returns, segments, inner_stop, outer_stop, SEQUENCE_LENGTH, horizon
        )
        alpha = float(np.quantile(np.abs(returns[fit_indices]), 1 / 3))
        fit_labels = make_labels(returns, fit_indices, alpha)
        validation_labels = make_labels(returns, validation_indices, alpha)
        fit_auxiliary = causal_features(features, fit_indices)
        validation_auxiliary = causal_features(features, validation_indices)
        for c_value in C_CANDIDATES:
            logistic = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
            logistic.fit(fit_auxiliary, fit_labels)
            metrics = metrics_from_prediction(
                validation_labels, logistic.predict(validation_auxiliary)
            )
            record = {
                "horizon": horizon,
                "C": c_value,
                "alpha": alpha,
                "fit_targets": len(fit_indices),
                "validation_targets": len(validation_indices),
                "fit_labels": label_summary(fit_labels),
                "validation_labels": label_summary(validation_labels),
                "validation": metrics,
            }
            horizon_diagnostics.append(record)
            key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if selected_horizon is None or key > (
                float(selected_horizon["validation"]["macro_f1"]),
                float(selected_horizon["validation"]["accuracy"]),
            ):
                selected_horizon = record
                selected_inner = {
                    "returns": returns,
                    "fit_indices": fit_indices,
                    "validation_indices": validation_indices,
                    "fit_labels": fit_labels,
                    "validation_labels": validation_labels,
                    "fit_auxiliary": fit_auxiliary,
                    "validation_auxiliary": validation_auxiliary,
                    "logistic": logistic,
                }
        print(
            f"horizon {horizon}: best val_f1="
            f"{max(row['validation']['macro_f1'] for row in horizon_diagnostics if row['horizon']==horizon):.4f}",
            flush=True,
        )
    if selected_horizon is None or selected_inner is None:
        raise RuntimeError("horizon selection failed")
    horizon = int(selected_horizon["horizon"])
    print(
        f"selected horizon={horizon}, C={selected_horizon['C']}, "
        f"logistic val_f1={selected_horizon['validation']['macro_f1']:.4f}",
        flush=True,
    )

    # Phase 2: select pure regularized vs causal-hybrid DeepLOB on inner validation.
    selection_book, selection_book_mean, selection_book_std = normalized_book(features, inner_stop)
    fit_partition = PreparedPartition(
        selected_inner["fit_indices"],
        selected_inner["fit_labels"],
        selected_inner["fit_auxiliary"],
    )
    validation_partition = PreparedPartition(
        selected_inner["validation_indices"],
        selected_inner["validation_labels"],
        selected_inner["validation_auxiliary"],
    )
    auxiliary_mean = fit_partition.auxiliary.mean(axis=0).astype(np.float32)
    auxiliary_std = np.maximum(fit_partition.auxiliary.std(axis=0), 1e-6).astype(np.float32)
    fit_loader = make_loader(
        selection_book, fit_partition, auxiliary_mean, auxiliary_std, batch_size, True, seed
    )
    validation_loader = make_loader(
        selection_book,
        validation_partition,
        auxiliary_mean,
        auxiliary_std,
        batch_size,
        False,
        seed,
    )
    model_selections = []
    for model_name in MODEL_FACTORIES:
        model_selections.append(
            fit_with_validation(model_name, fit_loader, validation_loader, device, seed)
        )
    chosen_model = max(
        model_selections,
        key=lambda row: (row["best_validation_macro_f1"], -row["best_validation_loss"]),
    )
    logistic_validation_log_probability = selected_inner["logistic"].predict_log_proba(
        validation_partition.auxiliary
    )
    calibration = select_calibration(
        chosen_model["best_validation_logits"],
        logistic_validation_log_probability,
        chosen_model["best_validation_truth"],
    )
    print(
        f"selected model={chosen_model['model_name']} epoch={chosen_model['best_epoch']}; "
        f"val_f1={chosen_model['best_validation_macro_f1']:.4f}; "
        f"ensemble val_f1={calibration['validation']['macro_f1']:.4f}",
        flush=True,
    )

    # Phase 3: refit the selected recipe on all first 70%; evaluate frozen final 30%.
    final_returns = symmetric_returns(
        (features[:, 0].astype(np.float64) + features[:, 2]) / 2, segments, horizon
    )
    development_indices = interval_indices(
        final_returns, segments, 0, outer_stop, SEQUENCE_LENGTH, horizon
    )
    test_indices = interval_indices(
        final_returns, segments, outer_stop, len(features), SEQUENCE_LENGTH, horizon
    )
    final_alpha = float(np.quantile(np.abs(final_returns[development_indices]), 1 / 3))
    development_labels = make_labels(final_returns, development_indices, final_alpha)
    test_labels = make_labels(final_returns, test_indices, final_alpha)
    development_auxiliary = causal_features(features, development_indices)
    test_auxiliary = causal_features(features, test_indices)
    final_book, final_book_mean, final_book_std = normalized_book(features, outer_stop)
    final_auxiliary_mean = development_auxiliary.mean(axis=0).astype(np.float32)
    final_auxiliary_std = np.maximum(development_auxiliary.std(axis=0), 1e-6).astype(np.float32)
    development_partition = PreparedPartition(
        development_indices, development_labels, development_auxiliary
    )
    test_partition = PreparedPartition(test_indices, test_labels, test_auxiliary)
    development_loader = make_loader(
        final_book,
        development_partition,
        final_auxiliary_mean,
        final_auxiliary_std,
        batch_size,
        True,
        seed,
    )
    test_loader = make_loader(
        final_book,
        test_partition,
        final_auxiliary_mean,
        final_auxiliary_std,
        batch_size,
        False,
        seed,
    )
    final_model, final_history = train_fixed_epochs(
        chosen_model["model_name"],
        development_loader,
        int(chosen_model["best_epoch"]),
        device,
        seed,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    final_loss, neural_metrics, neural_logits, final_truth = evaluate(
        final_model, test_loader, criterion, device
    )

    final_logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(selected_horizon["C"]),
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    final_logistic.fit(development_auxiliary, development_labels)
    logistic_log_probability = final_logistic.predict_log_proba(test_auxiliary)
    logistic_metrics = metrics_from_prediction(
        test_labels, logistic_log_probability.argmax(axis=1)
    )
    neural_log_probability = log_softmax_numpy(neural_logits)
    combined = (
        float(calibration["neural_weight"]) * neural_log_probability
        + float(calibration["logistic_weight"]) * logistic_log_probability
    )
    combined[:, 1] += float(calibration["stationary_logit_bias"])
    ensemble_metrics = metrics_from_prediction(test_labels, combined.argmax(axis=1))
    stationary_baseline = constant_baseline(test_labels, 1)

    checkpoint_path = HERE / "checkpoints/deeplob_7709_2026-08-07_improved.pt"
    logistic_path = HERE / "checkpoints/7709_2026-08-07_improved_logistic.joblib"
    results_path = HERE / "output/improved_results.json"
    torch.save(
        {
            "model_name": chosen_model["model_name"],
            "model_state_dict": final_model.state_dict(),
            "horizon": horizon,
            "alpha": final_alpha,
            "book_mean": final_book_mean,
            "book_std": final_book_std,
            "auxiliary_mean": final_auxiliary_mean,
            "auxiliary_std": final_auxiliary_std,
            "calibration": calibration,
            "split_index": outer_stop,
            "seed": seed,
        },
        checkpoint_path,
    )
    joblib.dump(final_logistic, logistic_path)

    serializable_selections = []
    for row in model_selections:
        serializable_selections.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "best_state",
                    "best_validation_logits",
                    "best_validation_truth",
                }
            }
        )
    payload = {
        "experiment": "Improved HK 07709 single-day DeepLOB with validation-only selection",
        "status": "complete",
        "source_cache": str(cache.resolve()),
        "outer_split": {
            "development_ratio": OUTER_TRAIN_RATIO,
            "test_ratio": 1 - OUTER_TRAIN_RATIO,
            "development_stop": outer_stop,
            "split_send_time_hkt": int(send_times[outer_stop]),
            "test_first_target_send_time_hkt": int(send_times[test_indices[0]]),
            "shared_input_snapshots": 0,
        },
        "inner_selection_split": {
            "fit_fraction_of_development": INNER_FIT_RATIO,
            "fit_stop": inner_stop,
            "fit_stop_send_time_hkt": int(send_times[inner_stop]),
            "validation_stop": outer_stop,
            "test_not_used_for_selection": True,
        },
        "diagnosis_and_selection": {
            "horizon_logistic_grid": horizon_diagnostics,
            "selected_horizon": selected_horizon,
            "model_grid": serializable_selections,
            "selected_model": chosen_model["model_name"],
            "selected_epoch": int(chosen_model["best_epoch"]),
            "calibration": calibration,
        },
        "final_labeling": {
            "horizon": horizon,
            "alpha": final_alpha,
            "alpha_fit": "eligible first-70% development targets only",
        },
        "final_data": {
            "development_targets": len(development_indices),
            "test_targets": len(test_indices),
            "development_labels": label_summary(development_labels),
            "test_labels": label_summary(test_labels),
        },
        "final_training": {
            "epochs": int(chosen_model["best_epoch"]),
            "history": final_history,
            "optimizer": "AdamW(backbone_lr=3e-4, head_lr=1e-3, weight_decay=1e-4)",
            "regularization": "dropout, label_smoothing=0.03, gradient_clip=5",
        },
        "final_test": {
            "neural_loss": final_loss,
            "selected_deeplob": neural_metrics,
            "causal_logistic": logistic_metrics,
            "validation_selected_ensemble": ensemble_metrics,
            "stationary_constant": stationary_baseline,
        },
        "artifacts": {
            "deeplob_checkpoint": str(checkpoint_path.resolve()),
            "logistic_model": str(logistic_path.resolve()),
        },
        "limitations": [
            "The final 30% was already observed in the earlier baseline experiment; this is an iterative holdout, not a pristine new holdout.",
            "Single-day overlapping windows remain strongly autocorrelated.",
            "No fees, execution, latency, or market-impact backtest is included.",
        ],
    }
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"FINAL neural acc={neural_metrics['accuracy']:.4f} f1={neural_metrics['macro_f1']:.4f}; "
        f"logistic acc={logistic_metrics['accuracy']:.4f} f1={logistic_metrics['macro_f1']:.4f}; "
        f"ensemble acc={ensemble_metrics['accuracy']:.4f} f1={ensemble_metrics['macro_f1']:.4f}; "
        f"stationary acc={stationary_baseline['accuracy']:.4f}",
        flush=True,
    )
    print(f"Saved {results_path}", flush=True)


if __name__ == "__main__":
    main()

