"""Connected component clustering module."""

import numpy as np
import scipy as sp
import torch

from .orphan import OrphanAssigner

__all__ = ['ConnectedComponentClusterer']


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

    def __call__(self, node_coords, edge_index, edge_assn,
                 node_clusts, edge_clusts):
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

        Returns
        -------
        TensorBatch
            (N) Cluster assignments for each of the points in the input
        """
        # Cast the input to numpy, if necessary
        tensor_input = isinstance(node_coords.tensor, torch.Tensor)
        if tensor_input:
            node_coords = node_coords.to_numpy()
            edge_index = edge_index.to_numpy()
            edge_assn = edge_assn.to_numpy()

        # Loop over the unique entries in the batch
        device = node_coords.device
        node_pred_list = []
        for b in range(node_coords.batch_size):
            # Narrow down the input to this specific entry
            node_clusts_b = node_clusts[b]
            edge_clusts_b = edge_clusts[b]
            assert len(node_clusts_b) == len(edge_clusts_b), (
                    "There should be one index per semantic class for both "
                    "nodes and edges. Got different size arrays.")

            node_coords_b = node_coords[b]
            edge_index_b = edge_index[b]
            edge_assn_b = edge_assn[b]

            # Loop over the semantic classes
            offset = 0
            node_pred_b = -np.ones(len(node_coords_b), dtype=int)
            for (nindex, eindex) in zip(node_clusts, edge_clusts):
                # Get predictions for a specifc (batch ID, shape) pair
                node_pred_b_s = self.fit_predict_one(
                        node_coords_b[nindex], edge_index_b[eindex],
                        edge_assn_b[eindex], offset)

                # Update the list of node assignments, offset the cluster
                # counter to make sure there is no overlap between shapes
                node_pred_b[nindex] = node_pred_b_s
                offset = int(node_pred_b.max()) + 1
            
            node_pred_list.append(node_pred_b)

        # Aggregate the node assignments into a batched object. If the input
        # is a torch tensor, convert the output
        node_pred = np.concatenate(node_pred_list)
        node_pred = TensorBatch(node_pred, counts=node_coords.counts)
        if is_tensor:
            node_pred = node_pred.to_tensor(
                    dtype=torch.long, device=node_coords.device)

        return node_pred

    def fit_predict_one(self, node_coords, edge_index, edge_assn, offset):
        """Assign cluster labels to graph nodes based on edge assignments.

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

        Returns
        -------
        np.ndarry
            (N) Cluster assignments for each of the points in the input
        """
        # Narrow down the list of edges to those turned on
        assert edge_index.shape[1] == 2, (
                "The edge index must be of shape (E, 2)")
        edges = edge_index[edge_assn == 1]

        # Convert the set of edges to a coordinate-format sparse adjacency matrix
        num_nodes = len(node_coords)
        edge_assn = np.ones(len(edges), dtype=int)
        adj = sp.coo_matrix((edge_assn, tuple(edges)), (num_nodes, num_nodes))

        # Find connected components, allow for unidirectional connections
        _, node_pred = sp.csgraph.connected_components(adj, connection='weak')
        node_pred = node_pred.astype(np.int64)

        # If min_size is set, downselect the points considered labeled
        if self.min_size > 1:
            _, counts, inverse = np.unique(
                    y, return_counts=True, return_inverse=True)
            orphan_mask = np.where(counts[inverse] < min_size)[0]
            node_pred[orphan_mask] = -1

        # If requested, assign the orphans
        if self.orphan_assigner is not None:
            node_pred = self.orphan_assigner(node_coords, node_pred)

        # Offset the valid assignments
        node_pred[node_pred != -1] += offset

        return node_pred
