"""Regression tests for base math helpers."""

import numpy as np
import pytest

from spine.math.base import (
    all,
    amax,
    amin,
    argmax,
    log_loss,
    mean,
    softmax,
    sum,
    unique,
)


def test_min_max_preserve_float_dtype_and_values():
    """Min/max reductions should not truncate floating point values."""
    x = np.array([[1.25, -2.5], [3.75, 4.5]], dtype=np.float32)

    mins = amin(x, 0)
    maxs = amax(x, 1)

    assert mins.dtype == x.dtype
    assert maxs.dtype == x.dtype
    np.testing.assert_allclose(mins, [1.25, -2.5])
    np.testing.assert_allclose(maxs, [1.25, 4.5])


def test_axis_one_reductions_match_numpy():
    """Axis-one reduction branches should match numpy."""
    x = np.array([[1.25, -2.5], [3.75, 4.5]], dtype=np.float32)

    np.testing.assert_allclose(sum(x, 1), np.sum(x, axis=1))
    np.testing.assert_allclose(mean(x, 1), np.mean(x, axis=1))
    np.testing.assert_array_equal(argmax(x, 1), np.argmax(x, axis=1))
    np.testing.assert_allclose(amin(x, 1), np.min(x, axis=1))


def test_axis_zero_all_matches_numpy():
    """Axis-zero all branch should match numpy."""
    x = np.array([[True, True], [True, False]])

    np.testing.assert_array_equal(all(x, 0), np.all(x, axis=0))


def test_axis_reductions_cover_axis_one_and_invalid_axis():
    """Base reductions should support axis 1 and reject other axes."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    np.testing.assert_allclose(softmax(x, 1).sum(axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(all(x > 0.0, 1), [True, True])

    with pytest.raises(AssertionError):
        softmax(x, 2)


def test_unique_empty_and_log_loss_empty_inputs():
    """Helpers should handle empty arrays."""
    values, counts = unique(np.empty(0, dtype=np.int64))

    assert len(values) == 0
    assert len(counts) == 0
    assert log_loss(np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float32)) == 0.0
