"""Internal helpers shared by cached HDF5 parsers."""

from __future__ import annotations

import numpy as np

__all__ = ["resolve_index_global_shift"]


def resolve_index_global_shift(
    index_list: list[np.ndarray], count_event: np.ndarray | None = None
) -> int:
    """Determine the offset span covered by one cached index entry.

    Parameters
    ----------
    index_list : list[np.ndarray]
        Normalized 1D index arrays for one cached entry.
    count_event : np.ndarray, optional
        Cached tensor or scalar count used to infer the span directly.

    Returns
    -------
    int
        Global index span associated with the cached entry.
    """
    if count_event is not None:
        count_array = np.asarray(count_event)
        if count_array.ndim == 0:
            return int(count_array)
        return int(len(count_array))

    if not index_list:
        return 0

    full_index = np.concatenate(index_list) if len(index_list) > 1 else index_list[0]
    return int(np.max(full_index, initial=-1) + 1)
