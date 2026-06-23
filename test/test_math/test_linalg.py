"""Tests for linear algebra helpers."""

import numpy as np
import pytest

from spine.math.linalg import contingency_table, norm, submatrix


def test_norm_matches_numpy_by_axis():
    """Norm should match numpy along both supported axes."""
    x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)

    np.testing.assert_allclose(norm(x, 0), np.linalg.norm(x, axis=0))
    np.testing.assert_allclose(norm(x, 1), np.linalg.norm(x, axis=1))


def test_submatrix_extracts_row_column_product():
    """Submatrix should select the Cartesian product of row/column indexes."""
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    rows = np.array([2, 0], dtype=np.int32)
    cols = np.array([3, 1], dtype=np.int32)

    np.testing.assert_array_equal(submatrix(x, rows, cols), x[np.ix_(rows, cols)])


def test_contingency_table_infers_and_accepts_shape():
    """Contingency table should infer dimensions or use explicit dimensions."""
    x = np.array([0, 0, 1, 2], dtype=np.int32)
    y = np.array([1, 1, 0, 1], dtype=np.int32)

    np.testing.assert_array_equal(
        contingency_table(x, y),
        [[0, 2], [1, 0], [0, 1]],
    )
    np.testing.assert_array_equal(
        contingency_table(x, y, nx=4, ny=3),
        [[0, 2, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
    )


def test_contingency_table_handles_empty_inputs():
    """Empty label inputs should produce a one-cell empty table."""
    table = contingency_table(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )

    np.testing.assert_array_equal(table, [[0]])


def test_contingency_table_rejects_length_mismatch():
    """Label arrays must describe the same samples."""
    with pytest.raises(ValueError, match="same length"):
        contingency_table(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )
