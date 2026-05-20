"""Shared helpers for visualization primitives."""

from __future__ import annotations

import numpy as np

__all__ = ["rotation_matrix_from_z"]


def rotation_matrix_from_z(direction: np.ndarray) -> np.ndarray:
    """Build a rotation matrix which maps the z-axis onto a direction.

    Parameters
    ----------
    direction : np.ndarray
        (3,) Target direction vector.

    Returns
    -------
    np.ndarray
        (3, 3) Rotation matrix.
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("Cannot build a rotation matrix from a zero direction.")

    target = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(target, z_axis):
        return np.eye(3)
    if np.allclose(target, -z_axis):
        return np.diag([1.0, -1.0, -1.0])

    vec = np.cross(z_axis, target)
    cos_angle = np.dot(z_axis, target)
    sin_angle = np.linalg.norm(vec)
    cross_mat = np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ]
    )

    return (
        np.eye(3)
        + cross_mat
        + cross_mat.dot(cross_mat) * ((1.0 - cos_angle) / sin_angle**2)
    )
