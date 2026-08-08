"""Self-describing event-level graph edge-index products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.utils.conditional import torch

from .base import DataProduct

__all__ = ["EdgeIndexData"]

ArrayLike = np.ndarray | torch.Tensor


@dataclass
class EdgeIndexData(DataProduct):
    """Graph incidence indexes for one event.

    Edges are stored in canonical ``(2, E)`` form. ``span`` records the number
    of nodes in the parent graph, allowing collation to offset each event into
    a shared node namespace.

    Attributes
    ----------
    features : np.ndarray
        Two-dimensional edge-index array with shape ``(2, E)``.
    span : int
        Parent-entry node span used when batching entries.
    directed : bool, default True
        Whether edges have a meaningful source-to-target direction.
    """

    product_type = "edge_index"

    features: ArrayLike
    span: int
    directed: bool = True

    def __len__(self) -> int:
        """Return the number of graph edges."""
        return self.features.shape[-1]

    @property
    def index(self) -> ArrayLike:
        """Return the canonical ``(2, E)`` incidence matrix."""
        return self.features

    @property
    def index_t(self) -> ArrayLike:
        """Return the row-oriented ``(E, 2)`` incidence matrix."""
        return self.features.T

    def __getitem__(self, index: Any) -> Any:
        """Index the underlying edge matrix."""
        return self.features[index]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Expose the edge matrix through NumPy's array protocol.

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
