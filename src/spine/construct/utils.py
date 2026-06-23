"""Shared typing and validation helpers for construct modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np

BuildMode: TypeAlias = Literal["reco", "truth"]
RunMode: TypeAlias = Literal["reco", "truth", "both", "all"]
Units: TypeAlias = Literal["cm", "px"]


def is_single_index(index: Any) -> bool:
    """Return ``True`` when an entry index represents a single event."""
    return np.isscalar(index) or (isinstance(index, np.ndarray) and index.ndim == 0)


def get_batch_size(index: Any) -> int:
    """Return the number of entries in a batched index container."""
    if is_single_index(index):
        raise TypeError("Expected batched `index`, but received a scalar value.")

    if isinstance(index, np.ndarray):
        return len(index)

    if isinstance(index, Sequence) and not isinstance(index, (str, bytes)):
        return len(index)

    raise TypeError(
        "`index` must be a scalar entry identifier or a sized batch container."
    )
