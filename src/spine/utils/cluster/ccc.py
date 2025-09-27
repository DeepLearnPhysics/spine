"""Connected component clustering module."""

import numpy as np
import torch

from spine.data import TensorBatch
from spine.math.graph import connected_components

from .orphan import OrphanAssigner

__all__ = ["ConnectedComponentClusterer"]


class ConnectedComponentClusterer:
    """Finds connected components based on graph edge assignments."""

    def __init__(self, min_size=0, orphan=None):
        """Initialize the connected component constructor.

        Parameters
        ----------
        min_size : int, default 0
            Minimum number of points in a label cluster to be considered
            already assigned. If below this value, it is considered an orphan.
        orphan : dict
            Orphan assigner configuration dictionary
        """
        # Store the parameters
        self.min_size = min_size

        # Initialize the orphan assigner
        self.orphan_assigner = None
        if orphan is not None:
            self.orphan_assigner = OrphanAssigner(**orphan)

    def __call__(
        self,
        node_coords,
        edge_index,
        edge_assn,
        node_clusts,
        edge_clusts,
        min_size=None,
    ):
        """Loop over the list of batch entries and semantic classes and
        label points use connected-component clustering.

        Parameters
        ----------
        node_coords : TensorBatch
            (N, 3) Set of point coordinates
        edge_index : TensorBatch
            (E, 2) Set of edge source and target indices
        edge_assn : TensorBatch
            (E) Boolean assignment for each edge (0 for off, 1 for on)
        node_clusts : List[List[List[int]]]
            (B, S) One list of node indices per (batch ID, shape) pair
        edge_clusts : List[List[List[int]]]
            (B, S) One list of edge indices per (batch ID, shape) pair
        min_size : int, optional
            Override the minimum cluster size set in the configuration

        Returns
        -------
        TensorBatch
            (N) Cluster assignments for each of the points in the input
        """
        # Cast the input to numpy, if necessary
        tensor_input = isinstance(node_coords.tensor, torch.Tensor)
        if tensor_input:
            device = node_coords.device
            node_clusts = node_clusts.to_numpy()
            edge_clusts = edge_clusts.to_numpy()
            node_coords = node_coords.to_numpy()
            edge_index = edge_index.to_numpy()
            edge_assn = edge_assn.to_numpy()

        # Loop over the unique entries in the batch
        node_pred_list = []
        for b in range(node_coords.batch_size):
            # Make predictions for one entry
            node_pred_b = self.fit_predict_entry(
                node_coords[b],
                edge_index[b],
                edge_assn[b],
                node_clusts[b],
                edge_clusts[b],
                min_size,
            )

            node_pred_list.append(node_pred_b)

        # Aggregate the node assignments into a batched object. If the input
        # is a torch tensor, convert the output
        node_pred = np.concatenate(node_pred_list)
        node_pred = TensorBatch(node_pred, counts=node_coords.counts)
        if tensor_input:
            node_pred = node_pred.to_tensor(dtype=torch.long, device=device)

        return node_pred

    def fit_predict_entry(
        self,
        node_coords,
        edge_index,
        edge_assn,
        node_clusts,
        edge_clusts,
        min_size=None,
    ):
        """Assign cluster labels to graph nodes based on edge assignments
        in one entry.

        Parameters
        ----------
        node_coords : np.ndarray
            (N, 3) Set of point coordinates
        edge_index : np.ndarray
            (E, 2) Set of edge source and target indices
        edge_assn : np.ndarray
            (E) Boolean assignment for each edge (0 for off, 1 for on)
        node_clusts : List[List[int]]
            (B, S) One list of node indices per (batch ID, shape) pair
        edge_clusts : List[List[int]]
            (B, S) One list of edge indices per (batch ID, shape) pair
        min_size : int, optional
            Override the minimum cluster size set in the configuration

        Returns
        -------
        np.ndarry
            (N) Cluster assignments for each of the points in the input
        """
        # Narrow down the input to this specific entry
        assert len(node_clusts) == len(edge_clusts), (
            "There should be one index per semantic class for both "
            "nodes and edges. Got different size arrays."
        )

        # Loop over the semantic classes
        offset = 0
        node_pred = -np.ones(len(node_coords), dtype=int)
        for nindex, eindex in zip(node_clusts, edge_clusts):
            # Get predictions for a specifc (batch ID, shape) pair
            node_pred_s = self.fit_predict_one(
                node_coords[nindex],
                edge_index[eindex],
                edge_assn[eindex],
                offset,
                min_size,
            )

            # Update the list of node assignments, offset the cluster
            # counter to make sure there is no overlap between shapes
            if len(nindex):
                node_pred[nindex] = node_pred_s
                offset = int(node_pred.max()) + 1

        return node_pred

    def fit_predict_one(
        self, node_coords, edge_index, edge_assn, offset, min_size=None
    ):
        """Assign cluster labels to graph nodes based on edge assignments
        in one graph, i.e. one (entry, shape) pair.

        Parameters
        ----------
        node_coords : np.ndarray
            (N, 3) Set of point coordinates
        edge_index : np.ndarray
            (E, 2) Set of edge source and target indices
        edge_assn : np.ndarray
            (E) Boolean assignment for each edge (0 for off, 1 for on)
        offset : int
            Offset to apply to assigned nodes
        min_size : int, optional
            Override the minimum cluster size set in the configuration

        Returns
        -------
        np.ndarry
            (N) Cluster assignments for each of the points in the input
        """
        # Narrow down the list of edges to those turned on
        assert edge_index.shape[1] == 2, "The edge index must be of shape (E, 2)."
        edges = edge_index[np.where(edge_assn)[0]]

        # Find connected components
        node_pred = connected_components(edges, len(node_coords))

        # If min_size is set, downselect the points considered labeled
        min_size = min_size if min_size is not None else self.min_size
        if min_size > 1:
            _, inverse, counts = np.unique(
                node_pred, return_inverse=True, return_counts=True
            )
            orphan_mask = np.where(counts[inverse] < min_size)[0]
            node_pred[orphan_mask] = -1

        # If requested, assign the orphans
        if self.orphan_assigner is not None:
            node_pred = self.orphan_assigner(node_coords, node_pred)

        # Offset the valid assignments
        node_pred[node_pred != -1] += offset

        return node_pred
