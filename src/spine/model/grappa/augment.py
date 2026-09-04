"""Graph-level augmentations for GrapPA training."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import EdgeIndexBatch, TensorBatch
from spine.utils.conditional import torch

__all__ = ["EdgeDropout", "EdgeSelection"]


class EdgeSelection:
    """Reusable event-aware selection over an original graph edge axis.

    The selection owns the bookkeeping needed to apply one sampled graph
    perturbation consistently to edge indexes, materialized features and
    cached supervision.

    Parameters
    ----------
    keep : TensorBatch
        One-dimensional Boolean mask partitioned like the original graph.
    """

    def __init__(self, keep: TensorBatch) -> None:
        values = keep.to_numpy().data
        if values.ndim != 1:
            raise ValueError("Edge selection must be a one-dimensional mask.")

        self.keep = keep
        self.mask = values.astype(bool, copy=False)
        self.counts = np.zeros(keep.batch_size, dtype=np.int64)
        edges = keep.to_numpy().edges
        for batch_id in range(keep.batch_size):
            lower, upper = edges[batch_id : batch_id + 2]
            self.counts[batch_id] = np.count_nonzero(self.mask[lower:upper])

    def filter_edge_index(self, edge_index: EdgeIndexBatch) -> EdgeIndexBatch:
        """Apply the selection to a graph incidence matrix.

        Parameters
        ----------
        edge_index : EdgeIndexBatch
            Graph whose edge axis matches the original selection.

        Returns
        -------
        EdgeIndexBatch
            Graph containing only retained edges, with updated event counts.
        """
        self._validate(edge_index.counts, edge_index.shape[1], "edge index")
        backend_mask = self._backend_mask(edge_index.is_numpy, edge_index.device)

        return EdgeIndexBatch(
            edge_index.index[:, backend_mask],
            self.counts,
            edge_index.spans,
            edge_index.directed,
        )

    def filter_tensor(self, batch: TensorBatch) -> TensorBatch:
        """Apply the selection to an edge-aligned tensor batch.

        Parameters
        ----------
        batch : TensorBatch
            Edge features, targets or validity values aligned with the
            original graph.

        Returns
        -------
        TensorBatch
            Selected values with recomputed event counts and retained schema.
        """
        self._validate(batch.counts, batch.shape[0], "tensor batch")
        backend_mask = self._backend_mask(batch.is_numpy, batch.device)

        return TensorBatch(
            batch.data[backend_mask],
            self.counts,
            has_batch_col=batch.has_batch_col,
            coord_cols=batch.coord_cols,
            schema=batch.schema,
            meta=batch.meta,
        )

    def _backend_mask(self, is_numpy: bool, device: Any) -> Any:
        """Return the mask on the backend and device of its target."""
        if is_numpy:
            return self.mask
        return torch.as_tensor(self.mask, device=device)

    def _validate(self, counts: Any, size: int, target: str) -> None:
        """Check that a target retains the original edge partition."""
        if not isinstance(counts, np.ndarray):
            counts = counts.detach().cpu().numpy()
        keep_counts = self.keep.counts
        if not isinstance(keep_counts, np.ndarray):
            keep_counts = keep_counts.detach().cpu().numpy()
        if size != self.keep.shape[0] or not np.array_equal(counts, keep_counts):
            raise ValueError(f"Edge selection must align with the {target}.")


class EdgeDropout:
    """Randomly remove graph edges during training.

    Directed edges are sampled independently. Undirected GrapPA graphs store
    each connection as adjacent reciprocal edges, so one decision is sampled
    per pair and applied to both directions. This preserves the graph's
    undirected contract while allowing entire connections to disappear.

    Parameters
    ----------
    probability : float
        Probability of dropping each directed edge or reciprocal edge pair.
        Must lie in the closed interval ``[0, 1]``.
    """

    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Edge dropout probability must be between 0 and 1.")

        self.probability = float(probability)

    def __call__(self, edge_index: EdgeIndexBatch) -> EdgeSelection:
        """Sample an event-aware edge selection.

        Parameters
        ----------
        edge_index : EdgeIndexBatch
            Graph incidence matrix before augmentation.

        Returns
        -------
        EdgeSelection
            Selection aligned with the original edge axis. It can be applied
            consistently to graph indexes, materialized features and cached
            supervision.

        Raises
        ------
        ValueError
            If an undirected event does not contain adjacent reciprocal pairs.
        """
        numpy_index = edge_index.to_numpy()
        index = numpy_index.index
        counts = numpy_index.counts
        keep = np.zeros(index.shape[1], dtype=bool)

        lower = 0
        for count_value in counts:
            count = int(count_value)
            upper = lower + count
            if edge_index.directed:
                event_keep = np.random.random(count) >= self.probability
            else:
                event_keep = self._sample_undirected(index[:, lower:upper])

            keep[lower:upper] = event_keep
            lower = upper

        return EdgeSelection(TensorBatch(keep, counts))

    def _sample_undirected(self, index: np.ndarray) -> np.ndarray:
        """Sample adjacent reciprocal pairs from one undirected event.

        Parameters
        ----------
        index : np.ndarray
            ``(2, E)`` event-local incidence matrix.

        Returns
        -------
        np.ndarray
            Boolean edge selection in which both directions of every pair
            receive the same decision.
        """
        num_edges = index.shape[1]
        if num_edges % 2:
            raise ValueError(
                "Undirected edge dropout requires an even edge count per event."
            )

        # GraphBase guarantees this layout; validate materialized inputs too.
        if num_edges and not np.array_equal(index[:, 1::2], index[::-1, ::2]):
            raise ValueError(
                "Undirected edge dropout requires adjacent reciprocal edge pairs."
            )

        pair_keep = np.random.random(num_edges // 2) >= self.probability
        return np.repeat(pair_keep, 2)
