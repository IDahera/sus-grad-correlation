"""Training and evaluation routines."""

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from susgrad.training._checks import assert_normalized, check_model_dataset_compatible
from susgrad.utils.devices import DEVICE

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a model."""

    accuracy: float  # percentage in [0, 100]
    loss: float
    n_samples: int


@dataclass
class EpochResult:
    """Outcome of training a single epoch."""

    epoch: int  # 1-based
    avg_loss: float
    duration_s: float


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device = DEVICE,
    progress: bool = False,
    desc: str = "",
) -> float:
    """Train *model* for one epoch; return the average loss."""
    model.to(device)
    model.train()
    iterator = tqdm(train_loader, desc=desc) if progress else train_loader

    loss_sum = 0.0
    n_batches = 0
    for data, target in iterator:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        n_batches += 1

    return loss_sum / n_batches if n_batches else 0.0


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    n_epochs: int,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    progress: bool = False,
) -> Iterator[EpochResult]:
    """Train epoch-by-epoch, yielding an :class:`EpochResult` after each one.

    Lets a caller (e.g. the capture scripts) snapshot the model between epochs.
    Validates model/dataset compatibility once, up front.
    """
    check_model_dataset_compatible(model, train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        start = time.perf_counter()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device=device,
            progress=progress, desc=f"Epoch {epoch}/{n_epochs}",
        )
        yield EpochResult(epoch=epoch, avg_loss=avg_loss, duration_s=time.perf_counter() - start)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    progress: bool = True,
) -> nn.Module:
    """Train *model* on *train_loader* with cross-entropy + Adam.

    Raises :class:`DimensionMismatchError` if the model and dataset shapes do
    not line up. Returns the trained model (trained in place).
    """
    check_model_dataset_compatible(model, train_loader)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_one_epoch(
            model, train_loader, optimizer, device=device,
            progress=progress, desc=f"Epoch {epoch + 1}/{epochs}",
        )

    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    device: torch.device = DEVICE,
    require_normalized: bool = True,
) -> EvalResult:
    """Evaluate *model* on *test_loader*.

    Raises :class:`DimensionMismatchError` on a shape mismatch and, when
    ``require_normalized`` is True, :class:`DataNotNormalizedError` if the inputs
    look un-normalised.
    """
    check_model_dataset_compatible(model, test_loader)
    if require_normalized:
        assert_normalized(test_loader)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += F.cross_entropy(output, target, reduction="sum").item()
            predicted = output.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total if total else 0.0
    mean_loss = loss_sum / total if total else 0.0
    return EvalResult(accuracy=accuracy, loss=mean_loss, n_samples=total)
