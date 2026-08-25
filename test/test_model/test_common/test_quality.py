"""Tests for shared overlap-quality threshold handling."""

from typing import Any, cast

import numpy as np
import pytest

from spine.cluster.quality import ClusterOverlapBatch
from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.common.quality import ClusterQualityFilter, OverlapThresholds


def _overlap() -> ClusterOverlapBatch:
    """Build three compact best-match records for threshold tests."""
    counts = [3]
    return ClusterOverlapBatch(
        TensorBatch(np.array([0, 1, -1]), counts),
        TensorBatch(np.array([6, 7, 0]), counts),
        TensorBatch(np.array([0.9, 0.9, 0.0]), counts),
        TensorBatch(np.array([0.8, 0.8, 0.0]), counts),
        TensorBatch(np.array([0.6, 0.7, 0.0]), counts),
    )


def test_overlap_thresholds_apply_scalar_and_class_values():
    """All active metrics should gate the best match conjunctively."""
    thresholds = OverlapThresholds(
        min_iou=[0.5, 0.8],
        min_purity=0.85,
        min_efficiency=0.75,
        num_classes=2,
    )

    assert thresholds.active
    assert thresholds.class_dependent
    np.testing.assert_array_equal(
        thresholds.mask(_overlap(), np.array([0, 1, 0])),
        [True, False, False],
    )


def test_overlap_thresholds_validate_configuration_and_classes():
    """Malformed thresholds and missing class selectors fail explicitly."""
    assert not OverlapThresholds().active
    assert not OverlapThresholds().class_dependent

    for value in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            OverlapThresholds(min_iou=value)
    with pytest.raises(TypeError, match="numeric"):
        OverlapThresholds(min_iou=cast(Any, "bad"))
    with pytest.raises(ValueError, match="one-dimensional"):
        OverlapThresholds(min_iou=cast(Any, [[0.5]]))
    with pytest.raises(ValueError, match="positive"):
        OverlapThresholds(num_classes=0)
    with pytest.raises(ValueError, match="exactly 2 values"):
        OverlapThresholds(min_iou=[0.5], num_classes=2)
    with pytest.raises(ValueError, match="requires `num_classes`"):
        OverlapThresholds(min_iou=[0.5], require_num_classes=True)

    thresholds = OverlapThresholds(min_iou=[0.5, 0.5])
    with pytest.raises(ValueError, match="exactly 3 values"):
        thresholds.validate_num_classes(3)
    with pytest.raises(ValueError, match="positive"):
        thresholds.validate_num_classes(0)
    with pytest.raises(ValueError, match="require class labels"):
        thresholds.mask(_overlap())
    with pytest.raises(ValueError, match="align"):
        thresholds.mask(_overlap(), np.array([0, 1]))

    np.testing.assert_array_equal(
        thresholds.mask(_overlap(), np.array([0, 4, 0])),
        [True, False, False],
    )


def test_cluster_quality_filter_reuses_cached_overlap(monkeypatch):
    """Several objectives should share one geometrical overlap calculation."""
    calls = []

    def overlap(*args):
        calls.append(args)
        return _overlap()

    monkeypatch.setattr(
        "spine.model.common.quality.get_cluster_overlap_batch",
        overlap,
    )
    clusters = IndexBatch(
        [np.array([0]), np.array([1]), np.array([2])],
        spans=[3],
        counts=[3],
        single_counts=[1, 1, 1],
    )
    cache = {}
    first = ClusterQualityFilter(min_iou=0.5)
    second = ClusterQualityFilter(min_purity=0.8)

    np.testing.assert_array_equal(
        first.node_mask(cast(Any, object()), clusters, cache=cache),
        [True, True, False],
    )
    np.testing.assert_array_equal(
        second.node_mask(cast(Any, object()), clusters, cache=cache),
        [True, True, False],
    )
    assert len(calls) == 1


def test_cluster_quality_filter_projects_class_dependent_edge_policy(monkeypatch):
    """An edge-class policy must be satisfied independently by both endpoints."""
    monkeypatch.setattr(
        "spine.model.common.quality.get_cluster_overlap_batch",
        lambda *args: _overlap(),
    )
    clusters = IndexBatch(
        [np.array([0]), np.array([1]), np.array([2])],
        spans=[3],
        counts=[3],
        single_counts=[1, 1, 1],
    )
    edges = EdgeIndexBatch(
        np.array([[0, 1], [1, 2]]),
        counts=[2],
        spans=[3],
        directed=True,
    )
    quality_filter = ClusterQualityFilter(
        min_iou=[0.5, 0.8],
        num_classes=2,
    )

    np.testing.assert_array_equal(
        quality_filter.edge_mask(
            cast(Any, object()),
            clusters,
            edges,
            np.array([0, 1]),
        ),
        [True, False],
    )
    with pytest.raises(ValueError, match="require edge labels"):
        quality_filter.edge_mask(cast(Any, object()), clusters, edges)
    with pytest.raises(ValueError, match="align with edges"):
        quality_filter.edge_mask(
            cast(Any, object()),
            clusters,
            edges,
            np.array([0]),
        )

    bad_edges = EdgeIndexBatch(
        np.array([[0], [4]]),
        counts=[1],
        spans=[3],
        directed=True,
    )
    with pytest.raises(IndexError, match="index the node-quality mask"):
        quality_filter.edge_mask(
            cast(Any, object()),
            clusters,
            bad_edges,
            np.array([0]),
        )


def test_inactive_cluster_quality_filter_accepts_all_objects():
    """An unconfigured filter should avoid overlap work and retain all items."""
    clusters = IndexBatch(
        [np.array([0]), np.array([1])],
        spans=[2],
        counts=[2],
        single_counts=[1, 1],
    )
    edges = EdgeIndexBatch(
        np.array([[0], [1]]),
        counts=[1],
        spans=[2],
        directed=True,
    )
    quality_filter = ClusterQualityFilter()

    np.testing.assert_array_equal(
        quality_filter.node_mask(cast(Any, object()), clusters),
        [True, True],
    )
    np.testing.assert_array_equal(
        quality_filter.edge_mask(cast(Any, object()), clusters, edges),
        [True],
    )
