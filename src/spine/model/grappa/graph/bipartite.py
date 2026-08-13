"""Bipartite graph constructor for GNNs."""

from typing import Any

import numba as nb
import numpy as np

from spine.cluster.label import get_cluster_label_batch
from spine.data import ClusterLabelBatch, IndexBatch

from .base import GraphBase

__all__ = ["BipartiteGraph"]


class BipartiteGraph(GraphBase):
    """Generates graphs that connect primary nodes to secondary nodes.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """

    # Name of the graph constructor (as specified in the configuration)
    name = "bipartite"

    def __init__(
        self,
        directed_to: str = "secondary",
        **kwargs: Any,
    ) -> None:
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
        if directed_to not in ("primary", "secondary"):
            raise ValueError("`directed_to` must be either 'primary' or 'secondary'.")
        self.directed_to = directed_to

    def generate(
        self,
        *,
        data: ClusterLabelBatch,
        clusts: IndexBatch,
        dist_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates an incidence matrix that connects nodes that share an
        edge in their corresponding kNN graph.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
        clusts : IndexBatch
            (C) Cluster indexes
        dist_mat : np.ndarray, optional
            Pairwise distance matrix, unused by this graph.

        Returns
        -------
        tuple of np.ndarray
            Edge index and per-entry edge counts.
        """
        # Get the primary status of each node
        if not isinstance(data, ClusterLabelBatch):
            raise TypeError("Bipartite graphs require structured cluster labels.")
        primaries = (
            get_cluster_label_batch(data, clusts, "interaction_primary")
            .numpy_tensor()
            .astype(bool, copy=False)
        )

        edge_index = self._generate(
            clusts.batch_ids, primaries, self.directed, self.directed_to
        )
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
        primaries: np.ndarray,
        directed: bool = True,
        directed_to: str = "secondary",
    ) -> np.ndarray:
        # Create the incidence matrix
        ret = np.empty((0, 2), dtype=np.int64)
        for primary_index in np.where(primaries)[0]:
            for secondary_index in np.where(~primaries)[0]:
                if batch_ids[primary_index] == batch_ids[secondary_index]:
                    ret = np.vstack(
                        (
                            ret,
                            np.array([[primary_index, secondary_index]]),
                        )
                    )

        # Handle directedness, by default graph is directed towards secondaries
        if directed:
            if directed_to == "primary":
                ret = ret[:, ::-1]
            elif directed_to != "secondary":
                raise ValueError("Graph orientation not recognized")

        return ret.T
