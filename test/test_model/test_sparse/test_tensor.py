"""Tests for the backend-neutral sparse tensor."""

import pytest
import torch

from spine.model import sparse


def test_duplicate_coordinates_restore_original_rows():
    """Sparse processing coalesces sites but retains row provenance."""
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 0, 0]], dtype=torch.int32
    )
    features = torch.tensor([[1.0], [3.0], [5.0]], requires_grad=True)

    tensor = sparse.SparseTensor(features, coordinates, batch_size=3)

    assert len(tensor) == 2
    assert tensor.reference_size == 3
    assert tensor.counts.tolist() == [1, 1, 0]
    expected = torch.tensor([[4.0], [4.0], [5.0]])
    assert torch.equal(tensor.aligned_features(), expected)

    restored = tensor.to_tensor_batch(include_coordinates=False, restore=True)
    assert restored.counts.tolist() == [2, 1, 0]
    assert torch.equal(restored.tensor, torch.tensor([[4.0], [4.0], [5.0]]))

    restored.tensor.sum().backward()
    assert torch.equal(features.grad, torch.tensor([[2.0], [2.0], [1.0]]))


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [("sum", 4.0), ("mean", 2.0), ("first", 1.0)],
)
def test_duplicate_reduction_policies(reduction, expected):
    """Every supported duplicate policy produces one deterministic site."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0], [3.0]]),
        torch.tensor([[0, 2, 3], [0, 2, 3]], dtype=torch.int32),
        duplicate_reduction=reduction,
    )

    assert tensor.shape == (1, 1)
    assert tensor.F.item() == expected
    assert tensor.unique_index.tolist() == [0]
    assert tensor.inverse_mapping.tolist() == [0, 0]


def test_invalid_duplicate_reduction_is_rejected():
    """Unknown duplicate reductions fail before backend construction."""
    with pytest.raises(ValueError, match="Unknown duplicate reduction"):
        sparse.SparseTensor(
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
            duplicate_reduction="maximum",
        )


def test_sparse_export_is_portable_tensor_batch():
    """Feature maps export unique coordinates and features."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 1, 2, 3], [2, 4, 5, 6]], dtype=torch.int32),
        batch_size=3,
    )

    exported = tensor.to_tensor_batch()

    assert exported.counts.tolist() == [1, 0, 1]
    assert exported.has_batch_col
    assert exported.coord_cols == (1, 2, 3)
    assert torch.equal(
        exported.tensor,
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0], [2.0, 4.0, 5.0, 6.0, 2.0]]),
    )


def test_unique_coordinates_preserve_native_order_without_provenance():
    """Unique inputs remain ordered and avoid duplicate-restoration metadata."""
    coordinates = torch.tensor(
        [[1, 4, 2], [0, 9, 1], [1, 0, 7], [0, 3, 5]],
        dtype=torch.int32,
    )
    features = torch.tensor([[10.0], [20.0], [30.0], [40.0]])

    tensor = sparse.SparseTensor(features, coordinates, batch_size=2)

    assert torch.equal(tensor.C, coordinates)
    assert torch.equal(tensor.F, features)
    assert tensor._reference_coordinates is None
    assert tensor.aligned_features().data_ptr() == tensor.F.data_ptr()
    assert tensor.unique_index.tolist() == [0, 1, 2, 3]
    assert tensor.inverse_mapping.tolist() == [0, 1, 2, 3]


def test_tensor_feature_operations_preserve_coordinates():
    """Feature replacement, arithmetic, casting, and detach retain metadata."""
    tensor = sparse.SparseTensor(
        torch.tensor([[2.0], [4.0]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=2,
    )

    replaced = tensor.replace_features(torch.tensor([[3.0], [5.0]]))
    added = tensor + tensor
    multiplied = tensor * 2
    divided = tensor / 2

    assert torch.equal(replaced.C, tensor.C)
    expected_double = torch.tensor([[4.0], [8.0]], dtype=torch.float64)
    expected_half = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    assert torch.equal(added.F, expected_double)
    assert torch.equal(multiplied.F, expected_double)
    assert torch.equal(divided.F, expected_half)
    assert tensor.float().dtype == torch.float32
    assert not tensor.detach().F.requires_grad


def test_constructor_aliases_and_validation():
    """Aliases work and malformed inputs fail before the backend."""
    tensor = sparse.SparseTensor(
        feats=torch.tensor([1.0, 2.0]),
        coords=torch.tensor([[0, 0], [1, 1]], dtype=torch.int32),
        tensor_stride=torch.tensor([2]),
        batch_size=2,
    )

    assert tensor.shape == (2, 1)
    assert tensor.tensor_stride == (2,)

    with pytest.raises(ValueError, match="requires `features`"):
        sparse.SparseTensor(coords=torch.empty((0, 2), dtype=torch.int32))
    with pytest.raises(ValueError, match="must be matrices"):
        sparse.SparseTensor(torch.ones((1, 1, 1)), torch.zeros((1, 2)))
    with pytest.raises(ValueError, match="equal length"):
        sparse.SparseTensor(torch.ones((2, 1)), torch.zeros((1, 2)))
    with pytest.raises(ValueError, match="coordinate map key"):
        sparse.SparseTensor(torch.ones((1, 1)))


def test_coordinate_map_construction_and_source_propagation():
    """Coordinate maps support aliases, lazy coordinates, and sources."""
    source = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=3,
    )
    tensor = sparse.SparseTensor(
        feats=torch.tensor([[3.0], [4.0]]),
        coords_key=source.coordinate_map_key,
        coords_manager=source.coordinate_manager,
        source=source,
    )

    assert torch.equal(tensor.C, source.C)
    assert tensor.batch_size == 3
    assert tensor.reference_size == 2
    assert tensor.coords_key == source.coordinate_map_key
    assert tensor.coords_man == source.coordinate_manager

    explicit = sparse.SparseTensor(
        torch.tensor([[5.0], [6.0]]),
        coordinate_map_key=source.coordinate_map_key,
        coordinate_manager=source.coordinate_manager,
    )
    assert torch.equal(explicit.C, source.C)

    empty = sparse.SparseTensor(
        torch.empty((0, 1)),
        coordinate_map_key=source.coordinate_map_key,
        coordinate_manager=source.coordinate_manager,
    )
    assert empty.shape == (0, 1)
    assert empty.backend_tensor is None


def test_backend_wrapping_without_a_source_infers_metadata():
    """A bare native result lazily exposes backend metadata and delegation."""
    source = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 0, 0], [2, 1, 0]], dtype=torch.int32),
    )
    tensor = sparse.SparseTensor.from_backend(source.backend_tensor)

    assert tensor.batch_size == 3
    assert tensor.reference_size == 2
    assert tensor.device == source.device
    assert tensor.coordinate_map_key is not None
    assert tensor.coordinate_manager is not None
    assert tensor.tensor_stride == (1, 1)
    assert tensor.D == 2
    assert torch.equal(tensor.aligned_features(), tensor.F)
    assert tensor.features_at(0).tolist() == [[1.0]]
    assert tensor.coordinates_at(2).tolist() == [[1, 0]]
    assert [len(value) for value in tensor.decomposed_features] == [1, 0, 1]
    assert [len(value) for value in tensor.decomposed_coordinates] == [1, 0, 1]
    coords, features = tensor.decomposed_coordinates_and_features
    assert len(coords) == len(features) == 3
    native_manager = tensor.backend_tensor.coordinate_manager
    assert tensor.coordinate_manager == native_manager

    with pytest.raises(AttributeError):
        _ = tensor._missing
    with pytest.raises(AttributeError):
        _ = sparse.SparseTensor.empty_like(tensor).missing


def test_empty_feature_replacement_and_alignment():
    """Empty tensors replace features and restore referenced rows safely."""
    source = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor(
            [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
            dtype=torch.int32,
        ),
        batch_size=2,
    )
    empty = sparse.SparseTensor.empty_like(source, tensor_stride=2)
    replaced = empty.replace_features(torch.empty((0, 3)))

    assert replaced.shape == (0, 3)
    assert replaced.tensor_stride == (2, 2)
    assert replaced.aligned_features().shape == (3, 3)
    assert replaced.coordinate_map_key is None
    assert replaced.coordinate_manager is None

    restored = replaced.to_tensor_batch(restore=True)
    assert restored.counts.tolist() == [2, 1]
    assert restored.shape == (3, 6)


def test_tensor_arithmetic_covers_tensor_and_scalar_operands():
    """All arithmetic operations preserve sparse wrapper semantics."""
    tensor = sparse.SparseTensor(
        torch.tensor([[2.0], [4.0]]),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=2,
    )
    other = tensor.replace_features(torch.tensor([[1.0], [2.0]]))

    assert (tensor + 1).F.tolist() == [[3.0], [5.0]]
    assert (tensor * other).F.tolist() == [[2.0], [8.0]]
    assert (tensor / other).F.tolist() == [[2.0], [2.0]]

    empty = sparse.SparseTensor.empty_like(tensor)
    assert (empty + empty).shape == (0, 1)
    assert (empty * empty).shape == (0, 1)
    assert (empty / empty).shape == (0, 1)
