"""Tests for lightweight HDF5 cache parsers."""

import numpy as np
import pytest

from spine.data import Meta, ObjectList, Particle
from spine.io.parse import (
    HDF5ClusterTensorParser,
    HDF5EdgeIndexParser,
    HDF5FeatureTensorParser,
    HDF5IndexListParser,
    HDF5ObjectListParser,
    HDF5ObjectParser,
    HDF5TensorParser,
)
from spine.io.parse.data import ParserObjectList, ParserTensor


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


def test_hdf5_feature_tensor_parser_feature_ablation():
    """Feature parser should support selecting a subset of cached columns."""
    parser = HDF5FeatureTensorParser(
        dtype="float32",
        tensor_event="node_features",
        feature_cols=[2, 0],
    )
    trees = {
        "node_features": np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )
    }

    result = parser(trees)

    np.testing.assert_allclose(
        result.features,
        np.asarray([[3.0, 1.0], [6.0, 4.0]], dtype=np.float32),
    )


def test_hdf5_feature_tensor_parser_ablation_requires_2d_input():
    """Feature ablation should reject non-2D cached tensors."""
    parser = HDF5FeatureTensorParser(
        dtype="float32",
        tensor_event="node_features",
        feature_cols=[0],
    )

    with pytest.raises(ValueError, match="requires a 2D cached feature tensor"):
        parser({"node_features": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)})


def test_hdf5_tensor_parser_splits_coords_features_and_meta():
    """Generic cached tensor parser should split batch/coords/features."""
    meta = Meta()
    parser = HDF5TensorParser(
        dtype="float32",
        tensor_event="data",
        meta_event="meta",
    )
    trees = {
        "data": np.asarray(
            [
                [0.0, 10.0, 11.0, 12.0, 1.0, 2.0],
                [0.0, 20.0, 21.0, 22.0, 3.0, 4.0],
            ],
            dtype=np.float64,
        ),
        "meta": meta,
    }

    result = parser(trees)

    assert isinstance(result, ParserTensor)
    np.testing.assert_array_equal(
        result.coords, np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    )
    np.testing.assert_allclose(
        result.features, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    assert result.meta is meta


def test_hdf5_tensor_parser_supports_feature_ablation():
    """Generic cached tensor parser should support feature-column selection."""
    parser = HDF5TensorParser(
        dtype="float32",
        tensor_event="data",
        feature_cols=[1],
    )
    trees = {
        "data": np.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
            dtype=np.float32,
        )
    }

    result = parser(trees)

    np.testing.assert_allclose(result.features, np.asarray([[5.0]], dtype=np.float32))


def test_hdf5_tensor_parser_requires_2d_input():
    """Generic cached tensor parser should reject non-2D tensors."""
    parser = HDF5TensorParser(dtype="float32", tensor_event="data")

    with pytest.raises(ValueError, match="must be 2D"):
        parser({"data": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)})


def test_hdf5_cluster_tensor_parser_restores_cluster_metadata():
    """Cluster-tensor parser should restore duplicate-handling semantics."""
    parser = HDF5ClusterTensorParser(
        dtype="float32",
        tensor_event="clust_label",
        meta_event="meta",
        index_cols=[0, 3],
        sum_cols=[1],
        prec_col=2,
        precedence=[4, 3, 2],
    )
    meta = Meta()
    trees = {
        "clust_label": np.asarray([[0.0, 1.0, 2.0, 3.0, 9.0, 8.0]], dtype=np.float32),
        "meta": meta,
    }

    result = parser(trees)

    assert result.remove_duplicates is True
    np.testing.assert_array_equal(result.index_cols, np.asarray([0, 3]))
    np.testing.assert_array_equal(result.sum_cols, np.asarray([1]))
    assert result.prec_col == 2
    np.testing.assert_array_equal(result.precedence, np.asarray([4, 3, 2]))
    assert result.meta is meta


def test_hdf5_tensor_parser_rejects_invalid_batch_column_layout():
    """Generic cached tensor parser should validate batch-column assumptions."""
    parser = HDF5TensorParser(
        dtype="float32",
        tensor_event="data",
        has_batch_col=True,
        coord_start_col=0,
    )

    with pytest.raises(ValueError, match="coord_start_col"):
        parser({"data": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)})


def test_hdf5_object_parser_returns_cached_object():
    """Object parser should forward reconstructed cached objects unchanged."""
    parser = HDF5ObjectParser(dtype="float32", object_event="meta")
    meta = Meta()

    result = parser({"meta": meta})

    assert result is meta


def test_hdf5_object_list_parser_wraps_typed_object_list():
    """Object-list parser should preserve explicit ObjectList typing."""
    particles = ObjectList([Particle(id=1), Particle(id=2)], default=Particle())
    parser = HDF5ObjectListParser(dtype="float32", object_list_event="particles")

    result = parser({"particles": particles})

    assert isinstance(result, ParserObjectList)
    assert isinstance(result.default, Particle)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[1].id == 2


def test_hdf5_object_list_parser_infers_default_from_first_element():
    """Object-list parser should infer typing from non-empty plain lists."""
    parser = HDF5ObjectListParser(dtype="float32", object_list_event="particles")

    result = parser({"particles": [Particle(id=3)]})

    assert isinstance(result, ParserObjectList)
    assert isinstance(result.default, Particle)
    assert len(result) == 1
    assert result[0].id == 3


def test_hdf5_object_list_parser_rejects_empty_untyped_lists():
    """Empty plain lists should fail because their object type is ambiguous."""
    parser = HDF5ObjectListParser(dtype="float32", object_list_event="particles")

    with pytest.raises(ValueError, match="Cannot infer the default type"):
        parser({"particles": []})


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
