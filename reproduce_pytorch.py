"""Modern, memory-efficient runner for the official DeepLOB PyTorch notebook.

The model architecture, FI-2010 split, prediction horizon, optimizer, and loss
match the authors' notebook. Window construction is lazy so the experiment can
run on machines that cannot hold every overlapping 100-event window in memory.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class FI2010Windows(Dataset):
    """Lazy FI-2010 windows with the layout used by the official notebook."""

    def __init__(self, matrix: np.ndarray, horizon_index: int = 4, window: int = 100):
        features = np.ascontiguousarray(matrix[:40, :].T, dtype=np.float32)
        labels = np.ascontiguousarray(matrix[-5:, :].T[:, horizon_index] - 1, dtype=np.int64)
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.window = window

    def __len__(self) -> int:
        return self.features.shape[0] - self.window + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[index : index + self.window].unsqueeze(0)
        y = self.labels[index + self.window - 1]
        return x, y


class DeepLOB(nn.Module):
    """Architecture from the authors' official PyTorch notebook."""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
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
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.inception_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inception_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat(
            (self.inception_1(x), self.inception_2(x), self.inception_3(x)),
            dim=1,
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        h0 = x.new_zeros((1, x.shape[0], 64))
        c0 = x.new_zeros((1, x.shape[0], 64))
        x, _ = self.lstm(x, (h0, c0))
        logits = self.output(x[:, -1, :])

        # Kept for parity with the official notebook, which passes probabilities
        # rather than raw logits to CrossEntropyLoss.
        return torch.softmax(logits, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the official DeepLOB FI-2010 experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--horizon-index",
        type=int,
        default=4,
        choices=range(5),
        help="FI-2010 label column: 0,1,2,3,4 correspond to horizons 1,2,3,5,10.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fi2010(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_and_val = np.loadtxt(data_dir / "Train_Dst_NoAuction_DecPre_CF_7.txt")
    split = int(np.floor(train_and_val.shape[1] * 0.8))
    train = train_and_val[:, :split]
    val = train_and_val[:, split:]

    test = np.hstack(
        [
            np.loadtxt(data_dir / "Test_Dst_NoAuction_DecPre_CF_7.txt"),
            np.loadtxt(data_dir / "Test_Dst_NoAuction_DecPre_CF_8.txt"),
            np.loadtxt(data_dir / "Test_Dst_NoAuction_DecPre_CF_9.txt"),
        ]
    )
    return train, val, test


def maybe_limit(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.int64)
            losses.append(criterion(model(inputs), targets).item())
    return float(np.mean(losses))


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            targets_all.append(targets.numpy())
            predictions_all.append(predictions.cpu().numpy())
    return np.concatenate(targets_all), np.concatenate(predictions_all)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("Loading FI-2010 files...")
    train_matrix, val_matrix, test_matrix = load_fi2010(args.data_dir)

    train_dataset = maybe_limit(
        FI2010Windows(train_matrix, args.horizon_index, args.window),
        args.max_train_samples,
    )
    val_dataset = maybe_limit(
        FI2010Windows(val_matrix, args.horizon_index, args.window),
        args.max_val_samples,
    )
    test_dataset = maybe_limit(
        FI2010Windows(test_matrix, args.horizon_index, args.window),
        args.max_test_samples,
    )
    del train_matrix, val_matrix, test_matrix

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)

    model = DeepLOB().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"samples: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}; parameters={parameter_count:,}"
    )

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "best_deeplob_state.pt"

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses: list[float] = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.int64)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = mean_loss(model, val_loader, criterion, device)
        duration = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "duration_seconds": duration,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} duration={duration:.1f}s"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    targets, predictions = predict(model, test_loader, device)
    test_accuracy = float(accuracy_score(targets, predictions))
    report = classification_report(
        targets,
        predictions,
        labels=[0, 1, 2],
        target_names=["class_0", "class_1", "class_2"],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window": args.window,
        "horizon_index": args.horizon_index,
        "parameter_count": parameter_count,
        "sample_counts": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_validation_loss": best_val_loss,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "history": history,
    }
    results_path = args.output_dir / "reproduction_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"test_accuracy={test_accuracy:.4f}")
    print(classification_report(targets, predictions, zero_division=0))
    print(f"results={results_path}")


if __name__ == "__main__":
    main()
