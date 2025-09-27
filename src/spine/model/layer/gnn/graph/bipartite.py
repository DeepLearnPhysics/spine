"""Bipartite graph constructor for GNNs."""

import numba as nb
import numpy as np

from spine.utils.globals import PRINT_COL
from spine.utils.gnn.cluster import get_cluster_label

from .base import GraphBase

__all__ = ["BipartiteGraph"]


class BipartiteGraph(GraphBase):
    """Generates graphs that connect primary nodes to secondary nodes.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "bipartite"

    def __init__(self, directed_to, **kwargs):
        """Initialize the graph constructor.

        This adds the possibility to set the directionality of the
        bipartite graph explicitly.

        Parameters
        ----------
        directed_to : str, default 'secondary'
            Direction of the edge information flow ('primary' or 'secondary')
        **kwargs : dict, optional
            Additional parameters to pass to the :class:`GraphBase` constructor.
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Store attribute
        self.directed_to = directed_to

    def generate(self, data, clusts, **kwargs):
        """Generates an incidence matrix that connects nodes that share an
        edge in their corresponding kNN graph.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
        clusts : IndexBatch
            (C) Cluster indexes
        **kwargs : dict, optional
            Unused graph generation arguments

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        """
        # Get the primary status of each node
        primaries = get_cluster_label(data.tensor, clusts.index_list, column=PRINT_COL)

        return self._generate(
            clusts.batch_ids, primaries, self.directed, self.directed_to
        )

    @nb.njit(cache=True)
    def bipartite_graph(
        batch_ids: nb.int64[:],
        primaries: nb.boolean[:],
        directed: nb.boolean = True,
        directed_to: str = "secondary",
    ) -> nb.int64[:, :]:
        # Create the incidence matrix
        ret = np.empty((0, 2), dtype=np.int64)
        for i in np.where(primaries)[0]:
            for j in np.where(~primaries)[0]:
                if batch_ids[i] == batch_ids[j]:
                    ret = np.vstack((ret, np.array([[i, j]])))

        # Handle directedness, by default graph is directed towards secondaries
        if directed:
            if directed_to == "primary":
                ret = ret[:, ::-1]
            elif directed_to != "secondary":
                raise ValueError("Graph orientation not recognized")

        return ret.T
