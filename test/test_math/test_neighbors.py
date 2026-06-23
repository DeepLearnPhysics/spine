"""Tests for neighbor classifiers."""

import numpy as np
import pytest

from spine.math.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


def test_radius_neighbors_assigns_and_reports_orphans():
    """Radius classifier should assign nearby labels and report unassigned queries."""
    x = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    y = np.array([1, 2], dtype=np.int64)
    xq = np.array([[0.2, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=np.float32)

    clf = RadiusNeighborsClassifier(radius=0.5, iterate=False)
    labels, orphan_index = clf.fit_predict(x, y, xq)

    np.testing.assert_array_equal(labels, [1, -1])
    np.testing.assert_array_equal(orphan_index, [1])


def test_radius_neighbors_iterates_over_new_labels():
    """Iterative radius classifier should use newly assigned labels."""
    x = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    y = np.array([7], dtype=np.int64)
    xq = np.array([[0.4, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)

    clf = RadiusNeighborsClassifier(radius=0.5, iterate=True)
    labels, orphan_index = clf.fit_predict(x, y, xq)

    np.testing.assert_array_equal(labels, [7, 7])
    np.testing.assert_array_equal(orphan_index, [])


def test_radius_neighbors_stops_when_no_assignments_change():
    """Radius classifier should stop when every query remains orphaned."""
    x = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    y = np.array([1], dtype=np.int64)
    xq = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)

    clf = RadiusNeighborsClassifier(radius=0.5, iterate=True)
    labels, orphan_index = clf.fit_predict(x, y, xq)

    np.testing.assert_array_equal(labels, [-1, -1])
    np.testing.assert_array_equal(orphan_index, [0, 1])


def test_k_neighbors_assigns_mode_labels():
    """KNN classifier should assign majority labels."""
    x = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    y = np.array([1, 1, 2], dtype=np.int64)
    xq = np.array([[0.05, 0.0, 0.0]], dtype=np.float32)

    clf = KNeighborsClassifier(k=2)
    labels, orphan_index = clf.fit_predict(x, y, xq)

    np.testing.assert_array_equal(labels, [1])
    np.testing.assert_array_equal(orphan_index, [])


def test_k_neighbors_handles_empty_reference_set():
    """KNN classifier should mark every query orphaned without references."""
    xq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    clf = KNeighborsClassifier(k=1)

    labels, orphan_index = clf.fit_predict(
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.int64),
        xq,
    )

    np.testing.assert_array_equal(labels, [-1, -1])
    np.testing.assert_array_equal(orphan_index, [0, 1])


def test_neighbor_classifiers_reject_invalid_configuration():
    """Invalid neighborhood parameters should fail at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        RadiusNeighborsClassifier(radius=-1.0)

    with pytest.raises(ValueError, match="positive"):
        KNeighborsClassifier(k=0)
