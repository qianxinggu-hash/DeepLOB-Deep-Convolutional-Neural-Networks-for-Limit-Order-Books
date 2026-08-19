#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "torch>=2.1",
# ]
# ///
"""Train DeepLOB efficiently on Apple Silicon with PyTorch MPS.

The original notebook materializes every overlapping input window. This script
keeps the FI-2010 matrix in memory once and creates each window on demand,
which substantially reduces memory use on a 16 GB Mac.

Example:
    uv run --python 3.12 train_pytorch_mac.py --epochs 50
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "data.zip"
TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILES = (
    "Test_Dst_NoAuction_DecPre_CF_7.txt",
    "Test_Dst_NoAuction_DecPre_CF_8.txt",
    "Test_Dst_NoAuction_DecPre_CF_9.txt",
)
NUM_FEATURES = 40
NUM_CLASSES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-efficient DeepLOB training for Apple Silicon"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument(
        "--label-index",
        type=int,
        default=4,
        choices=range(5),
        metavar="{0,1,2,3,4}",
        help="FI-2010 prediction-horizon column (default: 4, as in notebook)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 is recommended on macOS to avoid copying the dataset to workers",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early-stopping patience; 0 disables early stopping",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "deeplob_mac_best.pt",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="print training progress every N batches; 0 disables batch logs",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="limit training samples for a quick smoke test",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="limit validation samples for a quick smoke test",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="skip final evaluation on the three test files",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.sequence_length < 1:
        parser.error("--sequence-length must be at least 1")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be between 0 and 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    elif requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch build")
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_matrix(archive: Path, member: str) -> np.ndarray:
    """Load one FI-2010 text matrix directly from the zip as float32."""
    started = time.perf_counter()
    with zipfile.ZipFile(archive) as zf:
        try:
            with zf.open(member) as source:
                matrix = np.loadtxt(source, dtype=np.float32)
        except KeyError as exc:
            raise FileNotFoundError(f"{member!r} is not present in {archive}") from exc
    elapsed = time.perf_counter() - started
    print(f"Loaded {member}: {matrix.shape} in {elapsed:.1f}s")
    return matrix


class LOBWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create overlapping LOB windows lazily instead of copying all windows."""

    def __init__(
        self,
        matrix: np.ndarray,
        label_index: int,
        sequence_length: int,
        max_samples: int | None = None,
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[0] < NUM_FEATURES + 5:
            raise ValueError(f"Unexpected FI-2010 matrix shape: {matrix.shape}")
        if matrix.shape[1] < sequence_length:
            raise ValueError("The data split is shorter than the requested sequence length")

        # The transpose is only a view; no large window tensor is allocated here.
        self.features = matrix[:NUM_FEATURES].T
        self.labels = matrix[-5 + label_index]
        self.sequence_length = sequence_length
        available = matrix.shape[1] - sequence_length + 1
        self.length = min(available, max_samples) if max_samples else available

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stop = index + self.sequence_length
        # ascontiguousarray copies only this one small [T, 40] window.
        window = np.ascontiguousarray(self.features[index:stop])
        target = int(self.labels[stop - 1]) - 1
        return torch.from_numpy(window).unsqueeze(0), torch.tensor(target)


class DeepLOB(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.inception1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inception2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inception3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, batch_first=True)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv3(self.conv2(self.conv1(inputs)))
        features = torch.cat(
            (
                self.inception1(features),
                self.inception2(features),
                self.inception3(features),
            ),
            dim=1,
        )
        features = features.permute(0, 2, 1, 3)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features, _ = self.lstm(features)
        # CrossEntropyLoss expects raw logits, so softmax is intentionally omitted.
        return self.classifier(features[:, -1])


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    started = time.perf_counter()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.int64)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_count = targets.shape[0]
            total_loss += loss.item() * batch_count
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_seen += batch_count

            if training and log_every and batch_number % log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  batch {batch_number}/{len(loader)} "
                    f"({total_seen / elapsed:.0f} samples/s)"
                )

    return total_loss / total_seen, total_correct / total_seen


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def evaluate_test(
    model: nn.Module,
    archive: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    matrices = [load_matrix(archive, member) for member in TEST_FILES]
    test_matrix = np.concatenate(matrices, axis=1)
    del matrices
    gc.collect()

    test_dataset = LOBWindowDataset(
        test_matrix,
        label_index=args.label_index,
        sequence_length=args.sequence_length,
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, args.num_workers, args.seed
    )
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            predictions = model(inputs).argmax(dim=1).cpu()
            indices = targets.to(torch.int64) * NUM_CLASSES + predictions
            confusion += torch.bincount(
                indices, minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    elapsed = time.perf_counter() - started
    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / total
    print(f"Test: accuracy={accuracy:.4f}, samples={total:,}, duration={elapsed:.1f}s")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion.numpy())
    for class_index in range(NUM_CLASSES):
        true_positive = confusion[class_index, class_index].item()
        predicted = confusion[:, class_index].sum().item()
        actual = confusion[class_index].sum().item()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"class {class_index}: precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}, support={actual:,}"
        )


def main() -> None:
    args = parse_args()
    archive = args.data.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Data archive not found: {archive}")

    seed_everything(args.seed)
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    print(f"Data: {archive}")

    train_matrix = load_matrix(archive, TRAIN_FILE)
    split = math.floor(train_matrix.shape[1] * args.train_ratio)
    train_view = train_matrix[:, :split]
    val_view = train_matrix[:, split:]
    train_dataset = LOBWindowDataset(
        train_view,
        args.label_index,
        args.sequence_length,
        args.max_train_samples,
    )
    val_dataset = LOBWindowDataset(
        val_view,
        args.label_index,
        args.sequence_length,
        args.max_val_samples,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, args.seed
    )
    print(
        f"Samples: train={len(train_dataset):,}, validation={len(val_dataset):,}"
    )

    model = DeepLOB().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            log_every=args.log_every,
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.perf_counter() - started
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(checkpoint, model, optimizer, epoch, val_loss, args)
        else:
            epochs_without_improvement += 1

        marker = " saved" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, "
            f"duration={elapsed:.1f}s{marker}"
        )
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} unimproved epoch(s)")
            break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    print(
        f"Loaded best checkpoint from epoch {saved['epoch']} "
        f"(val_loss={saved['val_loss']:.4f}): {checkpoint}"
    )

    # Release training data before loading the test matrices.
    del train_loader, val_loader, train_dataset, val_dataset
    del train_view, val_view, train_matrix
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    if not args.skip_test:
        evaluate_test(model, archive, args, device)

    run_summary = {
        "device": str(device),
        "best_epoch": saved["epoch"],
        "best_val_loss": saved["val_loss"],
        "checkpoint": str(checkpoint),
    }
    print("Run summary:", json.dumps(run_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
