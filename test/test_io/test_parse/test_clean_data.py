"""Tests for sparse data cleanup helpers."""

import numba as nb
import numpy as np

from spine.io.parse.clean_data import (
    aggregate_features,
    clean_sparse_data,
    filter_duplicate_voxels,
    filter_duplicate_voxels_group,
    filter_voxels_ref,
)


def test_filter_duplicate_voxels():
    """Duplicate voxel filtering should keep the last duplicate."""
    voxels = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
    mask = filter_duplicate_voxels(voxels)
    assert np.array_equal(mask, np.asarray([False, True, True]))


def test_filter_duplicate_voxels_group_without_reference():
    """Grouped duplicate filtering should keep the last item per group."""
    voxels = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.int32)
    mask, groups = filter_duplicate_voxels_group(voxels)
    assert np.array_equal(mask, np.asarray([False, True, False, True]))
    assert np.array_equal(groups[1], np.asarray([0, 1]))
    assert np.array_equal(groups[3], np.asarray([2, 3]))


def test_filter_duplicate_voxels_group_with_precedence():
    """Grouped duplicate filtering should honor precedence ordering."""
    voxels = np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int32)
    reference = np.asarray([1, 2], dtype=np.int32)
    precedence = [2, 1]
    mask, groups = filter_duplicate_voxels_group(voxels, reference, precedence)
    assert np.array_equal(mask, np.asarray([False, True]))
    assert np.array_equal(groups[1], np.asarray([0, 1]))


def test_filter_voxels_ref():
    """Reference filtering should drop voxels absent from the sparse reference."""
    voxels = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32)
    ref = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.int32)
    mask = filter_voxels_ref(voxels, ref)
    assert np.array_equal(mask, np.asarray([True, False, True]))


def test_aggregate_features():
    """Feature aggregation should sum the requested columns over groups."""
    data = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    groups = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64[:])
    groups[1] = np.asarray([0, 1], dtype=np.int64)
    result = aggregate_features(data.copy(), groups, np.asarray([0], dtype=np.int64))
    assert result[1, 0] == 3.0
    assert result[1, 1] == 20.0


def test_clean_sparse_data_with_duplicates_and_reference():
    """Sparse cleanup should sort, deduplicate, aggregate, and filter."""
    cluster_voxels = np.asarray(
        [[1, 0, 0], [0, 0, 0], [0, 0, 0], [3, 0, 0]], dtype=np.int32
    )
    cluster_data = np.asarray(
        [[10.0, 2.0], [1.0, 0.0], [2.0, 1.0], [5.0, 0.0]], dtype=np.float32
    )
    sparse_voxels = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32)

    voxels, data = clean_sparse_data(
        cluster_voxels,
        cluster_data,
        sparse_voxels=sparse_voxels,
        sum_cols=np.asarray([0], dtype=np.int64),
        prec_col=1,
        precedence=[1, 0],
    )

    assert np.array_equal(voxels, sparse_voxels)
    assert np.array_equal(data[:, 0], np.asarray([3.0, 10.0], dtype=np.float32))
    assert np.array_equal(data[:, 1], np.asarray([1.0, 2.0], dtype=np.float32))


def test_clean_sparse_data_without_reference_or_aggregation():
    """Sparse cleanup should also work in the simple duplicate-removal mode."""
    cluster_voxels = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
    cluster_data = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)

    voxels, data = clean_sparse_data(cluster_voxels, cluster_data)
    assert np.array_equal(voxels, np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32))
    assert np.array_equal(data, np.asarray([[1.0], [3.0]], dtype=np.float32))


def test_clean_sparse_data_without_precedence_or_sum_cols_uses_simple_filter():
    """Sparse cleanup should fall back to simple duplicate filtering when requested."""
    voxels, data = clean_sparse_data(
        np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32),
        np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        prec_col=None,
        sum_cols=None,
    )

    assert np.array_equal(voxels, np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32))
    assert np.array_equal(data[:, 0], np.asarray([2.0, 3.0], dtype=np.float32))


def test_clean_data_python_paths_for_coverage():
    """Execute numba helpers through their Python implementations for coverage."""
    voxels = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)
    ref = np.asarray([1, 2], dtype=np.int32)
    precedence = nb.typed.List([2, 1])

    assert np.array_equal(
        filter_duplicate_voxels.py_func(voxels),
        np.asarray([False, True, True]),
    )

    mask, groups = filter_duplicate_voxels_group.py_func(
        np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int32),
        ref,
        precedence,
    )
    assert np.array_equal(mask, np.asarray([False, True]))
    assert np.array_equal(groups[1], np.asarray([0, 1]))

    assert np.array_equal(
        filter_voxels_ref.py_func(
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
            np.asarray([[0, 0, 0]], dtype=np.int32),
        ),
        np.asarray([True, False]),
    )

    groups_dict = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64[:])
    groups_dict[1] = np.asarray([0, 1], dtype=np.int64)
    result = aggregate_features.py_func(
        np.asarray([[1.0], [2.0]], dtype=np.float32),
        groups_dict,
        np.asarray([0], dtype=np.int64),
    )
    assert result[1, 0] == 3.0

    mask, groups = filter_duplicate_voxels_group.py_func(
        np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    )
    assert np.array_equal(mask, np.asarray([False, False, True]))
    assert np.array_equal(groups[2], np.asarray([0, 1, 2]))

    assert np.array_equal(
        filter_voxels_ref.py_func(
            np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32),
            np.asarray([[0, 0, 0], [3, 0, 0]], dtype=np.int32),
        ),
        np.asarray([True, False, False]),
    )
