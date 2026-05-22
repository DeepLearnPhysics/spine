"""Tests for visualization utility helpers."""

import numpy as np
import pytest

from spine.vis.trace.utils import (
    is_numeric_sequence,
    is_scalar_sequence,
    require_matching_length,
    rotation_matrix_from_z,
    select_numeric_or_sequence,
    select_scalar_or_sequence,
)


def test_rotation_matrix_from_z_handles_antiparallel_direction():
    rotmat = rotation_matrix_from_z(np.array([0.0, 0.0, -2.0]))

    np.testing.assert_allclose(rotmat @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])
    np.testing.assert_allclose(rotmat.T @ rotmat, np.eye(3))


def test_rotation_matrix_from_z_rejects_zero_direction():
    with pytest.raises(ValueError, match="zero direction"):
        rotation_matrix_from_z(np.zeros(3))


def test_scalar_sequence_helpers_cover_common_visualization_inputs():
    assert not is_scalar_sequence("label")
    assert is_scalar_sequence(["label"])
    assert is_numeric_sequence(np.array([1.0, 2.0]))
    assert is_scalar_sequence(np.array([1.0, 2.0]))
    assert select_scalar_or_sequence("label", 0) == "label"
    assert select_scalar_or_sequence(["a", "b"], 1) == "b"
    assert select_numeric_or_sequence(1.5, 0) == 1.5
    assert select_numeric_or_sequence(np.array([1.0, 2.0]), 1) == 2.0
    with pytest.raises(TypeError, match="scalar-like input"):
        select_scalar_or_sequence({"a": 1}, 0)
    with pytest.raises(TypeError, match="numeric input"):
        select_numeric_or_sequence("bad", 0)

    require_matching_length(["a", "b"], 2, "bad")
    with pytest.raises(ValueError, match="bad"):
        require_matching_length(["a", "b"], 1, "bad")
