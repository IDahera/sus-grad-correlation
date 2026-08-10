"""Shared model architectures.

Import these from any script so model definitions live in exactly one place:

    from susgrad.models import TabularMLP, LeNet5

Every model exposes ``input_shape`` (the expected shape of a single, un-batched
input) and ``output_dim`` (number of output logits). The training utilities use
these attributes to check that a model and a dataset are compatible before
training or evaluating.
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """A small feed-forward network for tabular (binary) classification.

    The architecture is an explicit Linear -> ReLU stack. Layers are kept
    unfused (no ``nn.Sequential`` shortcuts that hide ReLUs) so individual
    neuron activations stay easy to hook and visualise for SBFL analysis.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(hidden_dims)
        # Shape of a single sample (no batch dimension).
        self.input_shape: Tuple[int, ...] = (self.input_dim,)

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    """A plain fully-connected classifier that flattens its input first.

    A simple dense baseline (no convolutions) usable on image data too — handy as
    an easy-to-reason-about counterpart to LeNet. Its layers are all ``nn.Linear``,
    so per-neuron values are 1-D and render as the square heatmap overview.
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        num_classes: int = 10,
        hidden_dims: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        self.input_shape: Tuple[int, ...] = tuple(int(d) for d in input_shape)
        self.output_dim = int(num_classes)
        in_features = int(math.prod(self.input_shape))

        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeNet5(nn.Module):
    """LeNet-5 style convnet, size-adaptive to the input shape.

    Five trainable layers (conv×2 + fc×3). Works for any ``(C, H, W)`` input —
    e.g. ``(1, 28, 28)`` MNIST/Fashion-MNIST or ``(3, 32, 32)`` CIFAR-10 — by
    inferring the flattened feature size from a dummy forward pass:

        conv1 (C->6, 5x5, padded) -> pool -> conv2 (6->16, 5x5) -> pool
        -> fc1 (auto -> 120) -> fc2 (120 -> 84) -> fc3 (84 -> num_classes)
    """

    def __init__(self, num_classes: int = 10,
                 input_shape: Sequence[int] = (1, 28, 28)) -> None:
        super().__init__()
        self.input_shape: Tuple[int, ...] = tuple(int(d) for d in input_shape)
        self.output_dim = int(num_classes)
        in_channels = self.input_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Infer the flattened conv-feature size for fc1 (handles any H, W, C).
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, *self.input_shape)).shape[1]
        self.fc1 = nn.Linear(n_flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
