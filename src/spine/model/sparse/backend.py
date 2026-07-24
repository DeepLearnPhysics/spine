"""Select and access the sparse-convolution backend.

This module defines the small semantic contract that concrete sparse backends
must implement. Model code should use :mod:`spine.model.sparse` instead of
calling this registry or importing a backend directly.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

_BACKEND_NAME = os.getenv("SPINE_SPARSE_BACKEND", "minkowski")
_ADAPTER: Any = None


def name() -> str:
    """Return the configured sparse backend name.

    Returns
    -------
    str
        Backend name selected by ``SPINE_SPARSE_BACKEND``. The default is
        ``"minkowski"``.
    """
    return _BACKEND_NAME


def adapter() -> Any:
    """Load and cache the configured backend adapter.

    Returns
    -------
    module
        Module implementing the sparse backend contract.

    Raises
    ------
    ValueError
        If no adapter exists for the configured backend name.
    """
    global _ADAPTER  # pylint: disable=global-statement
    if _ADAPTER is None:
        try:
            _ADAPTER = importlib.import_module(
                f"{__package__}.backends.{_BACKEND_NAME}"
            )
        except ModuleNotFoundError as exc:
            if exc.name == f"{__package__}.backends.{_BACKEND_NAME}":
                raise ValueError(
                    f"Unsupported sparse backend: {_BACKEND_NAME}"
                ) from exc
            raise
    return _ADAPTER


def module(operation: str) -> type:
    """Resolve a semantic sparse operation to a backend module class.

    Parameters
    ----------
    operation : str
        Backend-neutral operation name, such as ``"Convolution"``.

    Returns
    -------
    type
        Backend implementation of the requested operation.
    """
    return adapter().module(operation)


def create_tensor(**kwargs: Any) -> Any:
    """Create a native sparse tensor.

    Parameters
    ----------
    **kwargs : Any
        Arguments forwarded to the selected backend tensor constructor.

    Returns
    -------
    Any
        Native sparse tensor owned by the backend.
    """
    return adapter().create_tensor(**kwargs)


def concatenate(*tensors: Any) -> Any:
    """Concatenate native sparse tensors on a shared coordinate map.

    Parameters
    ----------
    *tensors : Any
        Native sparse tensors with compatible coordinates.

    Returns
    -------
    Any
        Native sparse tensor containing the concatenated features.
    """
    return adapter().concatenate(*tensors)


def coordinates(tensor: Any) -> Any:
    """Return the coordinate matrix of a native sparse tensor."""
    return adapter().coordinates(tensor)


def features(tensor: Any) -> Any:
    """Return the feature matrix of a native sparse tensor."""
    return adapter().features(tensor)


def tensor_stride(tensor: Any) -> tuple[int, ...]:
    """Return the spatial stride of a native sparse tensor."""
    return adapter().tensor_stride(tensor)


def coordinate_map_key(tensor: Any) -> Any:
    """Return the coordinate-map key of a native sparse tensor."""
    return adapter().coordinate_map_key(tensor)


def coordinate_manager(tensor: Any) -> Any:
    """Return the coordinate manager of a native sparse tensor."""
    return adapter().coordinate_manager(tensor)


def features_at_coordinates(tensor: Any, queries: Any) -> Any:
    """Query native sparse features at continuous coordinates."""
    return adapter().features_at_coordinates(tensor, queries)
