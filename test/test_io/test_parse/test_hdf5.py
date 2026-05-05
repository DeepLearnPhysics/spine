"""Tests for lightweight HDF5 cache parsers."""

import numpy as np
import pytest

from spine.io.parse import (
    HDF5EdgeIndexParser,
    HDF5FeatureTensorParser,
    HDF5IndexListParser,
)
from spine.io.parse.data import ParserTensor


def test_hdf5_feature_tensor_parser():
    """Feature-only cached arrays should rebuild a ParserTensor directly."""
    parser = HDF5FeatureTensorParser(dtype="float32", tensor_event="node_features")
    trees = {"node_features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)}

    result = parser(trees)

    assert isinstance(result, ParserTensor)
    assert result.feats_only is True
    assert result.coords is None
    np.testing.assert_allclose(result.features, trees["node_features"])
    assert result.features.dtype == np.float32


def test_hdf5_index_list_parser_with_count_event():
    """Cached index lists should retain per-cluster sizes and infer offset span."""
    parser = HDF5IndexListParser(
        dtype="float32",
        index_event="clusts",
        count_event="node_features",
    )
    trees = {
        "clusts": np.asarray(
            [
                np.asarray([0, 2], dtype=np.int64),
                np.asarray([1, 3, 4], dtype=np.int64),
            ],
            dtype=object,
        ),
        "node_features": np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]]),
    }

    result = parser(trees)

    assert isinstance(result, ParserTensor)
    assert isinstance(result.features, list)
    assert result.global_shift == 5
    np.testing.assert_array_equal(result.single_counts, np.asarray([2, 3]))
    np.testing.assert_array_equal(result.features[0], np.asarray([0, 2]))
    np.testing.assert_array_equal(result.features[1], np.asarray([1, 3, 4]))


def test_hdf5_index_list_parser_infers_global_shift_from_max_index():
    """If no count tensor is provided, infer the offset range from the indexes."""
    parser = HDF5IndexListParser(dtype="float32", index_event="clusts")
    trees = {
        "clusts": np.asarray(
            [np.asarray([4, 6], dtype=np.int64), np.asarray([5], dtype=np.int64)],
            dtype=object,
        )
    }

    result = parser(trees)

    assert result.global_shift == 7
    np.testing.assert_array_equal(result.single_counts, np.asarray([2, 1]))


def test_hdf5_index_list_parser_empty_and_scalar_count_event():
    """Empty cached index lists and scalar count hints should both be supported."""
    parser = HDF5IndexListParser(
        dtype="float32",
        index_event="clusts",
        count_event="num_nodes",
    )
    trees = {"clusts": np.asarray([], dtype=object), "num_nodes": np.asarray(4)}

    result = parser(trees)

    assert result.global_shift == 4
    assert result.features == []
    np.testing.assert_array_equal(result.single_counts, np.asarray([], dtype=np.int32))


def test_hdf5_index_list_parser_empty_without_count_event():
    """Empty cached index lists should fall back to a zero offset span."""
    parser = HDF5IndexListParser(dtype="float32", index_event="clusts")

    result = parser({"clusts": np.asarray([], dtype=object)})

    assert result.global_shift == 0
    assert result.features == []


def test_hdf5_edge_index_parser_accepts_transposed_input():
    """Cached edge indexes stored as (E, 2) should be transposed on load."""
    parser = HDF5EdgeIndexParser(
        dtype="float32",
        index_event="edge_index",
        count_event="node_features",
    )
    trees = {
        "edge_index": np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        "node_features": np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
    }

    result = parser(trees)

    assert isinstance(result, ParserTensor)
    np.testing.assert_array_equal(
        result.features,
        np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
    )
    assert result.global_shift == 4


def test_hdf5_edge_index_parser_rejects_non_2d_input():
    """Edge-index parser should reject non-2D cached arrays."""
    parser = HDF5EdgeIndexParser(dtype="float32", index_event="edge_index")

    with pytest.raises(ValueError, match="must be 2D"):
        parser({"edge_index": np.asarray([0, 1, 2], dtype=np.int64)})


def test_hdf5_edge_index_parser_rejects_wrong_2d_shape():
    """Edge-index parser should reject 2D arrays that are not edge lists."""
    parser = HDF5EdgeIndexParser(dtype="float32", index_event="edge_index")

    with pytest.raises(ValueError, match=r"shape \(2, E\) or \(E, 2\)"):
        parser({"edge_index": np.asarray([[0, 1, 2]], dtype=np.int64)})
