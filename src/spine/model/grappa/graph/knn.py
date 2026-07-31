"""k Nearest-neighbor (kNN) graph constructor for GNNs."""

from typing import Any

import numba as nb
import numpy as np

from spine.data import IndexBatch, TensorBatch
from spine.math.linalg import submatrix

from .base import GraphBase

__all__ = ["KNNGraph"]


class KNNGraph(GraphBase):
    """Generates graphs based on the k nearest-neighbor (kNN) graph of the
    input node locations.

    Makes an edge for each nearest neighbor connection.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "knn"

    def __init__(self, k: int, **kwargs: Any) -> None:
        """Initialize the graph constructor.

        This adds the possibility to set the `k` parameter of the kNN graph.

        Parameters
        ----------
        k : int
            Maximum number of nodes a node can be connected to
        **kwargs : dict, optional
            Additional parameters to pass to the :class:`GraphBase`
            constructor.
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Store attribute
        if k < 1:
            raise ValueError(f"`k` must be positive, got {k}.")
        self.k = k

    def generate(
        self,
        *,
        data: TensorBatch,
        clusts: IndexBatch,
        dist_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates an incidence matrix that connects nodes that share an
        edge in their corresponding kNN graph.

        Parameters
        ----------
        clusts : IndexBatch
            (C) Cluster indexes
        dist_mat : Union[np.ndarray, torch.Tensor]
            (C, C) Matrix of pair-wise distances between clusters in the batch
        data : TensorBatch
            Batched voxel/value table, unused by this graph.

        Returns
        -------
        tuple of np.ndarray
            Edge index and per-entry edge counts.
        """
        if dist_mat is None:
            raise ValueError("KNN graph construction requires `dist_mat`.")
        edge_index = self._generate(
            np.asarray(clusts.batch_ids),
            self.k,
            dist_mat,
        )
        if not self.directed and edge_index.shape[1] > 0:
            canonical_edges = np.sort(edge_index, axis=0)
            edge_index = np.unique(canonical_edges, axis=1)
        edge_counts = self.edge_counts(
            edge_index,
            np.asarray(clusts.batch_ids),
            clusts.batch_size,
        )
        return edge_index, edge_counts

    @staticmethod
    @nb.njit(cache=True)
    def _generate(
        batch_ids: np.ndarray,
        k: int,
        dist_mat: np.ndarray,
    ) -> np.ndarray:
        # Use the available distance matrix to build a kNN graph
        ret = np.empty((0, 2), dtype=np.int64)
        for batch_id in np.unique(batch_ids):
            clust_ids = np.where(batch_ids == batch_id)[0]
            if len(clust_ids) > 1:
                subk = min(k + 1, len(clust_ids))
                submat = submatrix(dist_mat, clust_ids, clust_ids)
                for diagonal_index in range(len(submat)):
                    submat[diagonal_index, diagonal_index] = np.inf
                for source_index, source_row in enumerate(submat):
                    neighbor_indices = np.argsort(source_row)[: subk - 1]
                    edges = np.empty((subk - 1, 2), dtype=np.int64)
                    for neighbor_rank, neighbor_index in enumerate(
                        np.sort(neighbor_indices)
                    ):
                        edges[neighbor_rank] = [
                            clust_ids[source_index],
                            clust_ids[neighbor_index],
                        ]
                    if len(edges) > 0:
                        ret = np.vstack((ret, edges))

        return ret.T
