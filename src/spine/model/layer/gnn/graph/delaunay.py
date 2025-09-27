"""Delaunay graph constructor for GNNs."""

import numba as nb
import numpy as np
from scipy.spatial import Delaunay

from spine.utils.globals import COORD_COLS

from .base import GraphBase

__all__ = ["DelaunayGraph"]


class DelaunayGraph(GraphBase):
    """Generates graphs based on the Delaunay triangulation of the input
    node locations.

    Triangulates the input, converts the triangles into a list of valid edges.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "delaunay"

    def generate(self, data, clusts, **kwargs):
        """Generates an incidence matrix that connects nodes
        that share an edge in their corresponding Euclidean Delaunay graph.

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
        return self._generate(
            data.tensor,
            nb.typed.List(clusts.index_list),
            clusts.batch_ids,
            self.directed,
        )

    @staticmethod
    @nb.njit(cache=True)
    def _generate(
        data: nb.float64[:, :],
        clusts: nb.types.List(nb.int64[:]),
        batch_ids: nb.int64[:],
        directed: bool = False,
    ) -> nb.int64[:, :]:
        # For each batch, find the list of edges, append it
        edge_list, offset = [], 0
        edge_counts = np.zeros(len(counts), dtype=counts.dtype)
        for b in np.unique(batch_ids):
            # Combine the cluster masks into one
            clust_ids = np.where(batch_ids == b)[0]
            limits = np.array([0] + [len(clusts[i]) for i in clust_ids])
            limits = np.cumsum(limits)
            mask = np.zeros(limits[-1], dtype=np.int64)
            labels = np.zeros(limits[-1], dtype=np.int64)
            for i in range(len(clust_ids)):
                l, u = limits[i], limits[i + 1]
                mask[l:u] = clusts[clust_ids[i]]
                labels[l:u] = i

            # Run Delaunay triangulation in object mode because it relies on an
            # external package. Only way to speed this up would be to implement
            # Delaunay triangulation in Numba (tall order)
            with nb.objmode(tri="int32[:,:]"):
                # Run Delaunay triangulation in joggled mode, this
                # guarantees simplical faces (no ambiguities)
                tri = Delaunay(data[mask][:, COORD_COLS], qhull_options="QJ").simplices

            # Create an adjanceny matrix from the simplex list
            adj_mat = np.zeros((len(clust_ids), len(clust_ids)), dtype=np.bool_)
            for s in tri:
                for i in s:
                    for j in s:
                        if labels[j] > labels[i]:
                            adj_mat[labels[i], labels[j]] = True

            # Convert the adjancency matrix to a list of edges, store
            edges = np.where(adj_mat)
            edges = np.vstack((clust_ids[edges[0]], clust_ids[edges[1]]))

            edge_list.append(offset + edge_index.T)
            edge_counts[b] = edge_index.shape[-1]
            offset += c

        # Merge the blocks together
        offset = 0
        result = np.empty((2, num_edges), dtype=np.int64)
        for edges in edge_list:
            num_block = edges.shape[-1]
            result[:, offset : offset + num_block] = edges
            offset += num_block

        return result
