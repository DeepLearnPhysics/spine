"""Shared utility helpers for low-level visualization traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeGuard

import numpy as np

NumericLike: TypeAlias = bool | int | float | np.generic
NumericSequence: TypeAlias = Sequence[NumericLike] | np.ndarray
NumericOrSequence: TypeAlias = NumericLike | NumericSequence
ScalarLike: TypeAlias = NumericLike | str
ScalarSequence: TypeAlias = Sequence[ScalarLike] | np.ndarray
ScalarOrSequence: TypeAlias = ScalarLike | ScalarSequence
ColorInput: TypeAlias = ScalarOrSequence | None
HoverTextInput: TypeAlias = ScalarOrSequence | None
IntensityInput: TypeAlias = NumericOrSequence | None

__all__ = [
    "ColorInput",
    "HoverTextInput",
    "IntensityInput",
    "NumericLike",
    "NumericOrSequence",
    "NumericSequence",
    "ScalarLike",
    "ScalarOrSequence",
    "ScalarSequence",
    "is_scalar_sequence",
    "is_scalar_value",
    "is_numeric_sequence",
    "is_numeric_value",
    "require_matching_length",
    "select_numeric_or_sequence",
    "select_scalar_or_sequence",
    "rotation_matrix_from_z",
]


def is_scalar_sequence(value: object) -> TypeGuard[ScalarSequence]:
    """Return ``True`` when a value should be treated as a per-item sequence.

    Parameters
    ----------
    value : object
        Candidate scalar-or-sequence input.

    Returns
    -------
    bool
        ``True`` when the input is a sequence-like container of scalar values.
        Strings and bytes are treated as scalar values, not sequences.
    """
    return isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value, (str, bytes)
    )


def is_scalar_value(value: object) -> TypeGuard[ScalarLike]:
    """Return ``True`` when a value should be treated as one scalar input.

    Parameters
    ----------
    value : object
        Candidate scalar-or-sequence input.

    Returns
    -------
    bool
        ``True`` when the input is one scalar-like value accepted by the
        visualization helpers.
    """
    return isinstance(value, (bool, int, float, str, np.generic))


def is_numeric_sequence(value: object) -> TypeGuard[NumericSequence]:
    """Return ``True`` when a value should be treated as a numeric sequence.

    Parameters
    ----------
    value : object
        Candidate numeric-or-sequence input.

    Returns
    -------
    bool
        ``True`` when the input is a sequence-like container of numeric
        values. Strings and bytes are treated as scalar values, not sequences.
    """
    return isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value, (str, bytes)
    )


def is_numeric_value(value: object) -> TypeGuard[NumericLike]:
    """Return ``True`` when a value should be treated as one numeric input.

    Parameters
    ----------
    value : object
        Candidate numeric-or-sequence input.

    Returns
    -------
    bool
        ``True`` when the input is one numeric-like value accepted by the
        visualization helpers.
    """
    return isinstance(value, (bool, int, float, np.generic))


def require_matching_length(
    value: ScalarOrSequence | None, count: int, message: str
) -> None:
    """Validate that a per-item value sequence matches the expected length.

    Parameters
    ----------
    value : Union[ScalarOrSequence, None]
        Value provided by the caller.
    count : int
        Number of items that the per-item input should describe.
    message : str
        Error message to raise when the length does not match.
    """
    if value is not None and is_scalar_sequence(value) and len(value) != count:
        raise ValueError(message)


def select_scalar_or_sequence(
    value: ScalarOrSequence | None, index: int
) -> ScalarLike | None:
    """Fetch one element from a per-item input, or return a scalar unchanged.

    Parameters
    ----------
    value : Union[ScalarOrSequence, None]
        Scalar value or sequence of per-item values.
    index : int
        Index of the requested item.

    Returns
    -------
    Union[ScalarLike, None]
        The selected value for the requested item.
    """
    if value is None:
        return None

    if is_numeric_sequence(value):
        return value[index]

    if is_scalar_value(value):
        return value

    raise TypeError("Value must be a scalar-like input or a scalar sequence.")


def select_numeric_or_sequence(value: NumericOrSequence, index: int) -> NumericLike:
    """Fetch one numeric element from a per-item input, or return a scalar unchanged.

    Parameters
    ----------
    value : NumericOrSequence
        Numeric value or sequence of per-item numeric values.
    index : int
        Index of the requested item.

    Returns
    -------
    NumericLike
        The selected numeric value for the requested item.
    """
    if is_numeric_sequence(value):
        return value[index]

    if is_numeric_value(value):
        return value

    raise TypeError("Value must be a numeric input or a numeric sequence.")


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
