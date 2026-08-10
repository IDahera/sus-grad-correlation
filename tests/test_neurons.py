"""Tests for addressing a single neuron (experiment 2).

The whole point of the trajectory experiment is "this exact neuron, across
training", so three things have to hold: the random pick is reproducible from a
seed, the flat index and the (channel, y, x) coordinates describe the SAME cell,
and the per-epoch series comes back in epoch order.
"""

import random

import pytest
import torch

from susgrad.neurons import (
    NeuronRef,
    layer_shapes,
    neuron_series,
    neuron_value,
    parse_neuron_ref,
    pick_random_neuron,
)

SHAPES = {"conv1": (6, 4, 4), "fc1": (32,), "fc2": (10,)}


def test_layer_shapes_reads_the_dump():
    dump = {"conv1": torch.zeros(6, 4, 4), "fc1": torch.zeros(32)}
    assert layer_shapes(dump) == {"conv1": (6, 4, 4), "fc1": (32,)}


def test_same_seed_picks_the_same_neuron():
    first = pick_random_neuron(SHAPES, random.Random(42))
    second = pick_random_neuron(SHAPES, random.Random(42))
    assert (first.layer, first.index) == (second.layer, second.index)


def test_pick_is_uniform_over_neurons_not_over_layers():
    # conv1 holds 96 of the 138 neurons, so it must win ~70% of the draws --
    # a layer-first pick would give it ~33%.
    rng = random.Random(0)
    picks = [pick_random_neuron(SHAPES, rng).layer for _ in range(2000)]
    share = picks.count("conv1") / len(picks)
    assert 0.62 < share < 0.78


def test_pick_can_be_restricted_to_one_layer():
    rng = random.Random(0)
    refs = [pick_random_neuron(SHAPES, rng, layers=["fc2"]) for _ in range(20)]
    assert {r.layer for r in refs} == {"fc2"}
    assert all(0 <= r.index < 10 for r in refs)


def test_pick_rejects_unknown_or_empty_selection():
    with pytest.raises(ValueError):
        pick_random_neuron(SHAPES, random.Random(0), layers=["nope"])


def test_flat_index_and_coordinates_address_the_same_cell():
    tensor = torch.arange(6 * 4 * 4, dtype=torch.float32).reshape(6, 4, 4)
    ref = NeuronRef(layer="conv1", index=37, shape=(6, 4, 4))

    channel, y, x = ref.coords
    assert float(tensor[channel, y, x]) == 37.0
    assert neuron_value({"conv1": tensor}, ref) == 37.0


def test_parse_neuron_ref_accepts_index_and_coordinates():
    by_index = parse_neuron_ref("conv1:37", SHAPES)
    by_coords = parse_neuron_ref("conv1:2,1,1", SHAPES)
    assert by_index == by_coords          # 2*16 + 1*4 + 1 == 37
    assert by_index.coords == (2, 1, 1)


def test_parse_neuron_ref_validates_its_input():
    for bad in ("conv1", "nope:1", "conv1:x", "conv1:9,9,9", "fc1:999", "conv1:1,2"):
        with pytest.raises(ValueError):
            parse_neuron_ref(bad, SHAPES)


def test_describe_and_slug_identify_the_neuron():
    ref = NeuronRef(layer="conv1", index=37, shape=(6, 4, 4))
    described = ref.describe()
    assert "conv1[37]" in described and "channel 2" in described and "(6, 4, 4)" in described
    assert ref.slug() == "conv1-n37"
    assert NeuronRef("net.1", 5, (128,)).slug() == "net-1-n5"


def test_neuron_series_follows_one_cell_in_epoch_order():
    per_epoch = {e: {"fc1": torch.full((32,), float(e))} for e in (10, 0, 1)}
    ref = NeuronRef(layer="fc1", index=7, shape=(32,))

    epochs, values = neuron_series(per_epoch, ref)
    assert epochs == [0, 1, 10]
    assert values == [0.0, 1.0, 10.0]


def test_neuron_value_rejects_a_shape_mismatch():
    ref = NeuronRef(layer="fc1", index=7, shape=(32,))
    with pytest.raises(ValueError):
        neuron_value({"fc1": torch.zeros(64)}, ref)
