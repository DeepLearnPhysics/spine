"""MST graph constructor for GNNs."""

import numba as nb
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from spine.data import IndexBatch, TensorBatch
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

    def generate(
        self,
        *,
        data: TensorBatch,
        clusts: IndexBatch,
        dist_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates an incidence matrix that connects nodes that share an
        edge in their corresponding Euclidean MST graph.

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
            raise ValueError("MST graph construction requires `dist_mat`.")
        edge_index = self._generate(np.asarray(clusts.batch_ids), dist_mat)
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
        dist_mat: np.ndarray,
    ) -> np.ndarray:
        # For each batch, find the list of edges, append it
        ret = np.empty((0, 2), dtype=np.int64)
        for batch_id in np.unique(batch_ids):
            clust_ids = np.where(batch_ids == batch_id)[0]
            if len(clust_ids) > 1:
                submat = np.triu(submatrix(dist_mat, clust_ids, clust_ids))
                # SciPy interprets zero entries in a dense matrix as missing
                # edges. Preserve legitimate zero-distance connections with a
                # small positive weight. Use explicit scalar indexing because
                # Numba does not support two array indexes at once.
                for row in range(len(clust_ids)):
                    for column in range(row + 1, len(clust_ids)):
                        if submat[row, column] == 0.0:
                            submat[row, column] = 1.0e-6
                # Suboptimal. Ideally want to reimplement in Numba, tall order.
                with nb.objmode(mst_mat="float32[:,:]"):
                    mst_mat = minimum_spanning_tree(submat)
                    mst_mat = mst_mat.toarray().astype(np.float32)
                edges = np.where(mst_mat > 0.0)
                edges = np.vstack((clust_ids[edges[0]], clust_ids[edges[1]])).T
                ret = np.vstack((ret, edges))

        return ret.T
