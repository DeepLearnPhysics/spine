"""Internal helpers shared by cached HDF5 parsers."""

from __future__ import annotations

import numpy as np

__all__ = ["resolve_index_span"]


def resolve_index_span(count_event: np.ndarray) -> int:
    """Determine the offset span covered by one cached index entry.

    Parameters
    ----------
    count_event : np.ndarray
        Cached tensor or scalar count used to infer the indexed parent span.

    Returns
    -------
    int
        Parent-entry span associated with the cached entry.
    """
    count_array = np.asarray(count_event)
    if count_array.ndim == 0:
        return int(count_array)
    return int(len(count_array))
