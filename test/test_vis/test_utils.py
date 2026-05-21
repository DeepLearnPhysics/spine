"""Tests for visualization utility helpers."""

import numpy as np
import pytest

from spine.vis.utils import rotation_matrix_from_z


def test_rotation_matrix_from_z_handles_antiparallel_direction():
    rotmat = rotation_matrix_from_z(np.array([0.0, 0.0, -2.0]))

    np.testing.assert_allclose(rotmat @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])
    np.testing.assert_allclose(rotmat.T @ rotmat, np.eye(3))


def test_rotation_matrix_from_z_rejects_zero_direction():
    with pytest.raises(ValueError, match="zero direction"):
        rotation_matrix_from_z(np.zeros(3))
