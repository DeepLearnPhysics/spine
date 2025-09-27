"""Loop graph constructor for GNNs."""

import numpy as np

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

    def generate(self, clusts, **kwargs):
        """Generate a loop-graph on a set of N nodes.

        Parameters
        ----------
        clusts : IndexBatch
            (C) Cluster indexes
        **kwargs : dict, optional
            Unused graph generation arguments

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
