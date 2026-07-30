"""Complete graph constructor for GNNs."""

import numba as nb
import numpy as np

from spine.data import IndexBatch, TensorBatch

from .base import GraphBase

__all__ = ["CompleteGraph"]


class CompleteGraph(GraphBase):
    """Generates graphs that connect each node with every other node.

    If two nodes belong to separate batches, they cannot be connected.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "complete"

    def generate(
        self,
        *,
        data: TensorBatch,
        clusts: IndexBatch,
        dist_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates a complete graph on a set of batched nodes.

        Parameters
        ----------
        clusts : IndexBatch
            (C) Cluster indexes
        data : TensorBatch
            Batched voxel/value table, unused by this graph.
        dist_mat : np.ndarray, optional
            Pairwise distance matrix, unused by this graph.

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        np.ndarray
            (B) Number of edges in each entry of the batch
        """
        return self._generate(clusts.counts)

    @staticmethod
    @nb.njit(cache=True)
    def _generate(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Loop over the batches, define the adjacency matrix for each
        edge_counts = counts * (counts - 1) // 2
        num_edges = np.sum(edge_counts)
        edge_index = np.empty((2, num_edges), dtype=np.int64)
        offset, index = 0, 0
        for batch_id in range(len(counts)):
            # Build a list of edges
            node_count = counts[batch_id]
            adj_mat = np.triu(
                np.ones((node_count, node_count)),
                k=1,
            )
            edges = np.vstack(np.where(adj_mat))
            num_edges_b = edges.shape[1]

            edge_index[:, index : index + num_edges_b] = offset + edges
            index += num_edges_b
            offset += node_count

        return edge_index, edge_counts
