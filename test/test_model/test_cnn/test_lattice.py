"""Tests for training-time sparse lattice-phase randomization."""

import pytest
import torch

from spine.model import sparse
from spine.model.cnn.lattice import LatticePhase


def make_sparse_tensor() -> sparse.SparseTensor:
    """Build two sparse images, including repeated logical coordinates."""
    coordinates = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [1, 4, 5, 6],
        ],
        dtype=torch.int32,
    )
    features = torch.tensor([[1.0], [2.0], [3.0]])
    return sparse.SparseTensor(features, coordinates, batch_size=2)


def test_sparse_tensor_rephase_preserves_geometry_and_rows():
    """Backend coordinates move while canonical rows and features remain stable."""
    tensor = make_sparse_tensor()
    phase = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    shifted = tensor.rephase(phase)

    expected = tensor.C.clone()
    expected[:, 1:] += phase[expected[:, 0].long()]
    assert torch.equal(shifted.C, expected)
    assert torch.equal(shifted.canonical_coordinates, tensor.C)
    assert torch.equal(
        shifted.canonical_reference_coordinates,
        tensor._reference_coordinates,
    )
    assert torch.equal(shifted.aligned_features(), tensor.aligned_features())
    assert shifted.reference_size == tensor.reference_size

    wrapped = shifted.replace_features(shifted.F + 1.0)
    assert torch.equal(wrapped.lattice_phase, phase)
    assert torch.equal(wrapped.canonical_coordinates, tensor.C)


def test_sparse_tensor_rephase_validates_phase():
    """A sparse coordinate map accepts exactly one correctly shaped phase."""
    tensor = make_sparse_tensor()
    with pytest.raises(ValueError, match="shape"):
        tensor.rephase(torch.zeros((1, 3), dtype=torch.int32))

    shifted = tensor.rephase(torch.zeros((2, 3), dtype=torch.int32))
    with pytest.raises(ValueError, match="already"):
        shifted.rephase(torch.zeros((2, 3), dtype=torch.int32))


@pytest.mark.parametrize(
    ("dimension", "period", "error", "message"),
    [
        (0, 2, ValueError, "dimension"),
        (3, True, TypeError, "integer or sequence"),
        (3, [2, 2], ValueError, "dimension"),
        (3, [2, 1.5, 2], TypeError, "integer"),
        (3, [2, 0, 2], ValueError, "positive"),
    ],
)
def test_lattice_phase_validates_configuration(dimension, period, error, message):
    """Malformed phase dimensions and periods fail during construction."""
    with pytest.raises(error, match=message):
        LatticePhase(dimension, period)

    with pytest.raises(TypeError, match="mapping"):
        LatticePhase.from_config(3, True)
    with pytest.raises(ValueError, match="requires `period`"):
        LatticePhase.from_config(3, {})
    with pytest.raises(TypeError, match="Unexpected"):
        LatticePhase.from_config(3, {"period": 2, "extra": True})


def test_lattice_phase_resolves_automatic_period():
    """Automatic configuration adopts the encoder's effective total stride."""
    module = LatticePhase.from_config(3, {"period": "auto"}, total_stride=16)
    assert module.period == (16, 16, 16)

    with pytest.raises(ValueError, match="total stride"):
        LatticePhase.from_config(3, {"period": "auto"})


def test_lattice_phase_runs_only_during_training(monkeypatch):
    """Evaluation is canonical while training samples one phase per image."""
    tensor = make_sparse_tensor()
    module = LatticePhase(3, [2, 3, 4])

    def fixed_randint(period, size, **kwargs):
        return torch.full(size, period - 1, **kwargs)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    shifted = module(tensor)
    assert shifted.lattice_phase.tolist() == [[1, 2, 3], [1, 2, 3]]

    module.eval()
    assert module(tensor) is tensor
