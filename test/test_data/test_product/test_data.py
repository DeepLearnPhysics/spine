"""Tests for self-describing event-level data products."""

import numpy as np
import pytest

from spine.data import (
    EdgeIndexData,
    IndexData,
    IndexListData,
    Particle,
    TensorData,
)


def test_tensor_data_feature_metadata_is_feature_relative():
    """TensorData overlay metadata should use feature-relative columns."""
    tensor = TensorData(
        features=np.ones((2, 4), dtype=np.float32),
        index_cols=np.array([1, 3], dtype=np.int64),
        sum_cols=np.array([2], dtype=np.int64),
        avg_cols=np.array([0], dtype=np.int64),
        prec_col=3,
    )

    assert np.array_equal(tensor.index_cols, np.array([1, 3], dtype=np.int64))
    assert np.array_equal(tensor.sum_cols, np.array([2], dtype=np.int64))
    assert np.array_equal(tensor.avg_cols, np.array([0], dtype=np.int64))
    assert tensor.prec_col == 3


def test_tensor_data_feature_metadata_preserves_none_and_negative():
    """TensorData metadata should preserve absent and sentinel values."""
    tensor = TensorData(
        features=np.ones((1, 2), dtype=np.float32),
        index_cols=None,
        sum_cols=None,
        prec_col=-1,
    )

    assert tensor.index_cols is None
    assert tensor.sum_cols is None
    assert tensor.prec_col == -1


def test_index_data_products_store_specialized_contracts():
    """Index-style data products should preserve their specialized shape."""
    index = IndexData(features=np.asarray([0, 2, 4]), span=5)
    index_list = IndexListData(
        features=[np.asarray([0, 2]), np.asarray([1])],
        span=3,
        single_counts=np.asarray([2, 1]),
    )
    edge_index = EdgeIndexData(
        features=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        span=3,
    )

    np.testing.assert_array_equal(index.features, np.asarray([0, 2, 4]))
    assert index.span == 5
    assert len(index_list.features) == 2
    np.testing.assert_array_equal(index_list.single_counts, np.asarray([2, 1]))
    np.testing.assert_array_equal(
        edge_index.features, np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    )
    np.testing.assert_array_equal(edge_index.index_t, [[0, 1], [1, 2]])


def test_tensor_data_named_coordinate_and_feature_fields():
    """Named fields should disambiguate tensors containing two points."""
    tensor = TensorData(
        coords=np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
        features=np.asarray([[7.0, 8.0]], dtype=np.float32),
        coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
        feature_fields={"time": (0,), "shape": (1,)},
    )

    np.testing.assert_array_equal(tensor.coordinates("start"), [[1, 2, 3]])
    np.testing.assert_array_equal(tensor.coordinates("end"), [[4, 5, 6]])
    np.testing.assert_array_equal(tensor.coordinate_data, [[1, 2, 3, 4, 5, 6]])
    np.testing.assert_array_equal(tensor.feature("shape"), [[8.0]])
    with pytest.raises(ValueError, match="ambiguous"):
        tensor.coordinates()
    with pytest.raises(ValueError, match="ambiguous"):
        _ = tensor.coords


def test_tensor_data_coords_alias_for_unambiguous_coordinates():
    """The coords alias should expose a product's sole coordinate group."""
    tensor = TensorData(
        coords=np.asarray([[1, 2, 3]], dtype=np.int32),
        features=np.asarray([[4.0]], dtype=np.float32),
    )

    np.testing.assert_array_equal(tensor.coords, [[1, 2, 3]])
    np.testing.assert_array_equal(tensor.coordinates(), tensor.coords)


def test_data_objects_expose_public_index_attrs():
    """Data objects should expose index metadata through a public property."""
    particle = Particle()

    assert "id" in particle.index_attrs
    assert "parent_id" in particle.index_attrs
    assert "children_id" in particle.index_attrs
