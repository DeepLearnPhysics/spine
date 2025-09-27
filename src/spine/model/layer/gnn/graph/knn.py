"""k Nearest-neighbor (kNN) graph constructor for GNNs."""

import numba as nb
import numpy as np

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

    def __init__(self, k, **kwargs):
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
        self.k = k

    def generate(self, clusts, dist_mat, **kwargs):
        """Generates an incidence matrix that connects nodes that share an
        edge in their corresponding kNN graph.

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
        return self._generate(clusts.batch_ids, self.k, dist_mat, self.directed)

    @staticmethod
    @nb.njit(cache=True)
    def _generate(
        batch_ids: nb.int64[:],
        k: nb.int64,
        dist_mat: nb.float64[:, :],
        directed: bool = False,
    ) -> nb.int64[:, :]:
        # Use the available distance matrix to build a kNN graph
        ret = np.empty((0, 2), dtype=np.int64)
        for b in np.unique(batch_ids):
            clust_ids = np.where(batch_ids == b)[0]
            if len(clust_ids) > 1:
                subk = min(k + 1, len(clust_ids))
                submat = submatrix(dist_mat, clust_ids, clust_ids)
                for i in range(len(submat)):
                    idxs = np.argsort(submat[i])[1:subk]
                    edges = np.empty((subk - 1, 2), dtype=np.int64)
                    for j, idx in enumerate(np.sort(idxs)):
                        edges[j] = [clust_ids[i], clust_ids[idx]]
                    if len(edges):
                        ret = np.vstack((ret, edges))

        return ret.T
