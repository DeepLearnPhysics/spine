"""Shared schema descriptions and validation helpers for HDF5 writers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py


@dataclass
class DataFormat:
    """Describe the physical representation of one logical data product.

    Attributes
    ----------
    dtype : Any, optional
        Scalar, NumPy, HDF5, or compound dtype accepted by ``h5py``.
    class_name : str, optional
        SPINE object class reconstructed from compound rows.
    width : int or list[int], default 0
        Fixed array width, or one width for each element of a nested list.
    merge : bool, default False
        Whether equally shaped nested arrays share one values dataset.
    scalar : bool, default False
        Whether each event contains exactly one logical value.
    """

    # h5py accepts scalar classes, NumPy dtypes, enums, VLEN descriptors, and
    # compound field specifications. There is no narrower public common type.
    dtype: Any = None
    class_name: str | None = None
    width: int | list[int] = 0
    merge: bool = False
    scalar: bool = False


def require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """Return a named child group.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        Container expected to own the group.
    name : str
        Child name relative to ``parent``.

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


def require_dataset(parent: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """Return a named child dataset.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        Container expected to own the dataset.
    name : str
        Child name relative to ``parent``.

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
        Decoded string value.
    """
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise TypeError(f"HDF5 attribute '{name}' must be a string.")
    return value
