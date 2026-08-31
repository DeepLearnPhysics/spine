"""Shared schema validation and compatibility helpers for HDF5 readers."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

import spine.data


def require_dataset(parent: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """Return a named child dataset.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        Container expected to own the dataset.
    name : str
        Dataset name relative to ``parent``.

    Returns
    -------
    h5py.Dataset
        Validated child dataset.

    Raises
    ------
    TypeError
        If the named child is not an HDF5 dataset.
    """
    child = parent[name]
    if not isinstance(child, h5py.Dataset):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 dataset.")
    return child


def require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """Return a named child group.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        Container expected to own the group.
    name : str
        Group name relative to ``parent``.

    Returns
    -------
    h5py.Group
        Validated child group.

    Raises
    ------
    TypeError
        If the named child is not an HDF5 group.
    """
    child = parent[name]
    if not isinstance(child, h5py.Group):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 group.")
    return child


def decode_string_attribute(value: Any, name: str) -> str:
    """Normalize one required byte- or string-valued HDF5 attribute.

    Parameters
    ----------
    value : Any
        Raw HDF5 attribute value.
    name : str
        Attribute name used in validation errors.

    Returns
    -------
    str
        Decoded attribute value.

    Raises
    ------
    TypeError
        If ``value`` is neither bytes nor a string.
    """
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise TypeError(f"HDF5 attribute '{name}' must be a string.")
    return value


def contiguous_runs(entries: np.ndarray) -> list[tuple[int, int]]:
    """Convert ordered entry IDs into inclusive-exclusive contiguous runs.

    Parameters
    ----------
    entries : np.ndarray
        Ordered file-local entry identifiers.
    """
    if len(entries) == 0:
        return []

    # Start a new run whenever consecutive file-local entry IDs are not
    # adjacent. Each returned boundary can be used directly as a slice.
    runs = []
    first = previous = int(entries[0])
    for raw_value in entries[1:]:
        value = int(raw_value)
        if value != previous + 1:
            runs.append((first, previous + 1))
            first = value
        previous = value
    runs.append((first, previous + 1))
    return runs


def resolve_object_class(class_name: str, array: np.ndarray) -> type:
    """Resolve a stored object name to its concrete SPINE data class.

    Older region-reference files stored image metadata under the ambiguous
    name ``Meta``. The structured ``count`` field is used to distinguish its
    two- and three-dimensional variants; current files store explicit names.

    Parameters
    ----------
    class_name : str
        Class name stored in the HDF5 object metadata.
    array : numpy.ndarray
        Structured array slice containing serialized objects.

    Returns
    -------
    type
        Concrete SPINE data class used for reconstruction.
    """
    if class_name != "Meta":
        return getattr(spine.data, class_name)

    from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D

    # Legacy metadata needs a structured count field to reveal dimensionality.
    names = getattr(array.dtype, "names", None)
    if names is None or "count" not in names:
        raise TypeError(
            "Legacy HDF5 class_name='Meta' requires a structured dtype "
            "with a 'count' field."
        )
    sample = array[0] if len(array) else None
    if sample is None:
        return ImageMeta3D

    dimension = len(sample["count"])
    if dimension == 2:
        return ImageMeta2D
    if dimension == 3:
        return ImageMeta3D

    raise ValueError(
        f"Unsupported legacy Meta dimensionality: {dimension}. Expected 2 or 3."
    )
