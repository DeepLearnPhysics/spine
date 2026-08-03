"""Self-describing event-level index products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.utils.conditional import torch

from .base import DataProduct

__all__ = ["IndexData", "IndexListData"]

ArrayLike = np.ndarray | torch.Tensor


@dataclass
class IndexData(DataProduct):
    """Flat event-local indexes into a parent product.

    ``span`` records the size of the parent namespace. Collation uses it to
    offset indexes from successive events without relying on a parallel
    sidecar key.

    Attributes
    ----------
    features : np.ndarray
        One-dimensional index array.
    span : int
        Parent-entry span used when batching entries.
    """

    product_type = "index"

    features: ArrayLike
    span: int

    def __len__(self) -> int:
        """Return the number of indexes."""
        return len(self.features)

    def __getitem__(self, index: Any) -> Any:
        """Return one index or slice from the underlying values."""
        return self.features[index]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Expose event-local indexes through NumPy's array protocol.

        PyTorch values are detached and transferred to CPU before optional
        dtype conversion and copying.
        """
        # Keep NumPy storage zero-copy while explicitly detaching PyTorch values
        if isinstance(self.features, np.ndarray):
            array = np.asarray(self.features, dtype=dtype)
        else:
            array = self.features.detach().cpu().numpy()
            if dtype is not None:
                array = array.astype(dtype, copy=False)

        return array.copy() if copy else array


@dataclass
class IndexListData(DataProduct):
    """Jagged collection of event-local indexes into a parent product.

    A common example is a list of voxel indexes, one array per cluster.
    ``single_counts`` preserves each member length when the list is flattened
    and batched.

    Attributes
    ----------
    features : list[np.ndarray]
        List of one-dimensional index arrays.
    span : int
        Parent-entry span used when batching entries.
    single_counts : np.ndarray, optional
        Per-index sizes used to restore jagged list structure after batching.
    """

    product_type = "index_list"

    features: list[ArrayLike]
    span: int
    single_counts: np.ndarray | None = None

    def __len__(self) -> int:
        """Return the number of index arrays."""
        return len(self.features)

    def __iter__(self) -> Any:
        """Iterate over the event-local index arrays in stored order."""
        return iter(self.features)

    def __getitem__(self, index: Any) -> Any:
        """Return one event-local index array."""
        return self.features[index]
