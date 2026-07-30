"""Loop graph constructor for GNNs."""

import numpy as np

from spine.data import IndexBatch, TensorBatch

from .base import GraphBase

__all__ = ["LoopGraph"]


class LoopGraph(GraphBase):
    """Generates loop-only graphs.

    Connects every node in the graph with itself but nothing else.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "loop"

    def generate(
        self,
        *,
        data: TensorBatch,
        clusts: IndexBatch,
        dist_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a loop-graph on a set of N nodes.

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
        # There is exactly one edge per cluster
        edge_counts = clusts.counts

        # Define the loop graph
        num_nodes = np.sum(edge_counts)
        edge_index = np.repeat(np.arange(num_nodes)[None, :], 2, axis=0)

        return edge_index, edge_counts
