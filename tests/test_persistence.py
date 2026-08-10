"""Tests for save_model / load_model round-tripping, with cleanup."""

import torch

from susgrad.persistence import load_model, save_model


def test_save_creates_loadable_file(model, tmp_path):
    path = save_model(model, "unit_test_model", directory=tmp_path)
    assert path.exists(), "save_model must write a file."

    loaded = load_model(path, map_location=torch.device("cpu"))
    assert loaded is not None
    # A real forward pass proves the loaded object is a usable model.
    out = loaded(torch.randn(2, model.input_dim))
    assert out.shape == (2, model.output_dim)


def test_load_preserves_parameters_and_dims(model, tmp_path):
    path = save_model(model, "roundtrip", directory=tmp_path)
    loaded = load_model(path, map_location=torch.device("cpu"))

    original = model.state_dict()
    restored = loaded.state_dict()

    assert original.keys() == restored.keys(), "Parameter set must be identical."
    for key in original:
        assert original[key].shape == restored[key].shape, f"shape changed: {key}"
        assert torch.equal(original[key], restored[key]), f"values changed: {key}"


def test_tmp_path_cleanup(model, tmp_path):
    # tmp_path is auto-removed by pytest; assert nothing leaks into the default dir.
    path = save_model(model, "ephemeral", directory=tmp_path)
    assert str(tmp_path) in str(path)
