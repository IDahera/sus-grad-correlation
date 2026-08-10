"""Addressing a single neuron inside a model's per-layer tensors.

Experiment 2 follows ONE neuron across training, so it needs a way to name a
neuron unambiguously, pick one at random reproducibly, and pull its value out of
every per-epoch dump. All three live here rather than in the script, because
"which neuron was this?" has to survive into the log file and be re-checkable
later.

A neuron is identified by its layer plus its **flat index** into that layer's
tensor (row-major). For conv layers the equivalent ``(channel, y, x)``
coordinates are carried alongside, since the flat index alone is meaningless to
a reader.
"""

import math
import random
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class NeuronRef:
    """One neuron: its layer, flat index and the layer's shape."""

    layer: str
    index: int                    # flat, row-major index into the layer tensor
    shape: Tuple[int, ...]        # the layer's per-neuron shape, e.g. (6, 28, 28)

    @property
    def coords(self) -> Tuple[int, ...]:
        """Multi-dimensional position, e.g. ``(channel, y, x)`` for a conv layer."""
        return tuple(int(c) for c in _unravel(self.index, self.shape))

    @property
    def n_neurons(self) -> int:
        return int(math.prod(self.shape)) if self.shape else 0

    def describe(self) -> str:
        """Human-readable id, e.g. ``conv1[1234] = (channel 1, y 15, x 22) of (6, 28, 28)``."""
        if len(self.shape) <= 1:
            return f"{self.layer}[{self.index}] of {self.n_neurons} neurons"
        names = ("channel", "y", "x") if len(self.shape) == 3 else \
            tuple(f"dim{i}" for i in range(len(self.shape)))
        position = ", ".join(f"{n} {c}" for n, c in zip(names, self.coords))
        return f"{self.layer}[{self.index}] = ({position}) of {tuple(self.shape)}"

    def slug(self) -> str:
        """Filename-safe id, e.g. ``conv1-n1234``."""
        return f"{re.sub(r'[^0-9a-zA-Z]+', '-', self.layer).strip('-')}-n{self.index}"


def _unravel(index: int, shape: Sequence[int]) -> Tuple[int, ...]:
    coords = []
    for dim in reversed(tuple(shape)):
        coords.append(index % dim)
        index //= dim
    return tuple(reversed(coords))


def layer_shapes(mapping: Mapping[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
    """``{layer: shape}`` from one ``{layer: tensor}`` dump."""
    return {name: tuple(tensor.shape) for name, tensor in mapping.items()}


def pick_random_neuron(
    shapes: Mapping[str, Sequence[int]],
    rng: Optional[random.Random] = None,
    *,
    layers: Optional[Sequence[str]] = None,
) -> NeuronRef:
    """Pick one neuron uniformly at random from ALL neurons of the model.

    Uniform over neurons, not over layers: every neuron in the network has the
    same chance, so a big layer is proportionally more likely to supply the pick
    (picking a layer first would over-represent the small output layer). Pass
    *layers* to restrict the draw, and a seeded ``rng`` to make it reproducible.
    """
    rng = rng or random.Random()
    names = [n for n in shapes if layers is None or n in layers]
    if not names:
        raise ValueError(f"No layers to choose from (available: {sorted(shapes)}).")

    sizes = [int(math.prod(tuple(shapes[n]))) for n in names]
    total = sum(sizes)
    if total <= 0:
        raise ValueError("The selected layers contain no neurons.")

    target = rng.randrange(total)
    for name, size in zip(names, sizes):
        if target < size:
            return NeuronRef(layer=name, index=target, shape=tuple(shapes[name]))
        target -= size
    raise AssertionError("unreachable: target index exceeded the neuron count")


def parse_neuron_ref(text: str, shapes: Mapping[str, Sequence[int]]) -> NeuronRef:
    """Parse ``"conv1:1234"`` or ``"conv1:1,15,22"`` (coordinates) into a NeuronRef."""
    if ":" not in text:
        raise ValueError(f"Expected '<layer>:<index>' or '<layer>:<c,y,x>', got {text!r}.")
    layer, position = text.split(":", 1)
    layer = layer.strip()
    if layer not in shapes:
        raise ValueError(f"Unknown layer {layer!r}; available: {sorted(shapes)}.")

    shape = tuple(shapes[layer])
    parts = [p.strip() for p in position.split(",") if p.strip()]
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"Neuron position must be integers, got {position!r}.")

    if len(values) == 1:
        index = values[0]
    else:
        if len(values) != len(shape):
            raise ValueError(f"Layer {layer!r} has shape {shape}; got {len(values)} coordinates.")
        index = 0
        for coord, dim in zip(values, shape):
            if not 0 <= coord < dim:
                raise ValueError(f"Coordinate {coord} out of range for shape {shape}.")
            index = index * dim + coord

    n_neurons = int(math.prod(shape))
    if not 0 <= index < n_neurons:
        raise ValueError(f"Index {index} out of range for layer {layer!r} ({n_neurons} neurons).")
    return NeuronRef(layer=layer, index=index, shape=shape)


def neuron_value(mapping: Mapping[str, torch.Tensor], ref: NeuronRef) -> float:
    """This neuron's value in one ``{layer: tensor}`` dump."""
    tensor = mapping[ref.layer]
    if tuple(tensor.shape) != ref.shape:
        raise ValueError(
            f"Layer {ref.layer!r} has shape {tuple(tensor.shape)}, but the neuron "
            f"reference was built for {ref.shape}."
        )
    return float(tensor.detach().cpu().reshape(-1)[ref.index])


def neuron_series(
    per_epoch: Mapping[int, Mapping[str, torch.Tensor]],
    ref: NeuronRef,
) -> Tuple[list, list]:
    """``(epochs, values)`` for one neuron across per-epoch dumps, epoch-ordered."""
    epochs = sorted(per_epoch)
    return epochs, [neuron_value(per_epoch[e], ref) for e in epochs]
