"""MST graph constructor for GNNs."""

import numba as nb
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from spine.math.linalg import submatrix

from .base import GraphBase

__all__ = ["MSTGraph"]


class MSTGraph(GraphBase):
    """Generates graphs based on the minimum-spanning tree (MST) of the input
    node locations.

    Makes an edge for each branch in the minimum-spanning tree.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "mst"

    def generate(self, clusts, dist_mat, **kwargs):
        """Generates an incidence matrix that connects nodes that share an
        edge in their corresponding Euclidean MST graph.

        Parameters
        ----------
        clusts : IndexBatch
            (C) Cluster indexes
        dist_mat : Union[np.ndarray, torch.Tensor]
            (C, C) Matrix of pair-wise distances between clusters in the batch
        **kwargs : dict, optional
            Unused graph generation arguments

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        """
        return self._generate(clusts.batch_ids, dist_mat, self.directed)

    @staticmethod
    @nb.njit(cache=True)
    def _generate(
        batch_ids: nb.int64[:], dist_mat: nb.float64[:, :], directed: bool = False
    ) -> nb.int64[:, :]:
        # For each batch, find the list of edges, append it
        edge_list = []
        num_edges = 0
        ret = np.empty((0, 2), dtype=np.int64)
        for b in np.unique(batch_ids):
            clust_ids = np.where(batch_ids == b)[0]
            if len(clust_ids) > 1:
                submat = np.triu(submatrix(dist_mat, clust_ids, clust_ids))
                # Suboptimal. Ideally want to reimplement in Numba, tall order.
                with nb.objmode(mst_mat="float32[:,:]"):
                    mst_mat = minimum_spanning_tree(submat)
                    mst_mat = mst_mat.toarray().astype(np.float32)
                edges = np.where(mst_mat > 0.0)
                edges = np.vstack((clust_ids[edges[0]], clust_ids[edges[1]])).T
                ret = np.vstack((ret, edges))

        return ret.T
