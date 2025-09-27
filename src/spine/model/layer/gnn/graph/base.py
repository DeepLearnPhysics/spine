"""Base class for all graph construction classes."""

import inspect
from warnings import warn

import numpy as np

from spine.data import EdgeIndexBatch
from spine.utils.globals import COORD_COLS
from spine.utils.gnn.network import inter_cluster_distance


class GraphBase:
    """Parent class for all graph constructors."""

    # Name of the graph constructor (as specified in the configuration)
    name = None

    # List of recognized distance methods
    _dist_methods = ("voxel", "centroid")

    # List of recognized distance algorithms
    _dist_algorithms = ("brute", "iterative", "recursive")

    def __init__(
        self,
        directed=False,
        max_length=None,
        classes=None,
        max_count=None,
        dist_method="voxel",
        dist_algorithm="brute",
    ):
        """Initializes attributes shared accross all graph constructors.

        Parameters
        ----------
        directed : bool, default False
            If `True`, direct the edges from lower to higher rank only
        max_length : Union[float, List[float]], optional
            Length limitation to be applied to the edges. Can be:
            - Sclar: Constant threshold
            - Array: N*(N-1)/2 elements which correspond to the upper triangle
                     of an adjacency matrix providing cuts for each class pairs
        classes : Union[int, List[int]], optional
            List of classes that are involved in the graph
        max_count : int, optional
            Maximum number of edges that can be produced (memory limitation)
        dist_method : str, default 'voxel'
            Method used to compute inter-node distance ('voxel' or 'centroid')
        dist_algorithm : str, default 'brute'
            Algorithm used to comppute inter-node distance ('brute' or 'iterative')
        """
        # Check on enumarated strings
        assert dist_method in self._dist_methods, (
            f"Distance computation method not recognized: {dist_method}. "
            f"Must be one of {self._dist_methods}."
        )
        assert dist_algorithm in self._dist_algorithms, (
            f"Distance computation algorithm not recognized: {dist_algorithm}. "
            f"Must be one of {self._dist_algorithms}."
        )

        # Store attributes
        self.directed = directed
        self.max_count = max_count
        self.dist_centroid = dist_method == "centroid"
        self.dist_iterative = dist_algorithm != "brute"

        # Convert `max_length` to a matrix, if provided as a `triu`
        self.max_length = max_length
        if isinstance(max_length, list):
            assert classes is not None, (
                "If specifying the edge length cut per class, "
                "must provide the list of classes"
            )

            num_classes = np.max(classes) + 1
            assert len(max_length) == num_classes * (num_classes + 1) / 2, (
                "If provided as a list, the maximum edge length should be "
                "given for each upper triangular element of a matrix of "
                "size (num_classes*num_classes)."
            )

            max_length_mat = np.zeros((num_classes, num_classes), dtype=float)
            max_length_mat[np.triu_indices(num_classes)] = max_length
            max_length_mat += max_length_mat.T - np.diag(np.diag(max_length_mat))

            self.max_length = max_length_mat

        # Store whether the inter-cluster distance matrix must be evaluated
        self.compute_dist = max_length is not None or self.name in ("mst", "knn")

        # If this is a loop graph, simply set as undirected
        assert (
            self.name != "loop" or self.directed
        ), "For loop graphs, set as directed (no need for reciprocal)"

    def __call__(self, data, clusts, classes=None, groups=None):
        """Filters input to keep only what is needed to generate a graph.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
        clusts : IndexBatch
            (C) Cluster indexes
        classes : TensorBatch, optional
            (C) List of cluster semantic class used to define the max length
        groups : TensorBatch, optional
            (C) List of cluster groups which should not be mixed

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        """
        # Generate the inter-cluster distsnce matrix, if needed
        dist_mat, closest_index = None, None
        if self.compute_dist:
            dist_mat, closest_index = inter_cluster_distance(
                data.tensor[:, COORD_COLS],
                clusts.index_list,
                clusts.counts,
                centroid=self.dist_centroid,
                iterative=self.dist_iterative,
                return_index=True,
            )

        # Generate the edge index
        edge_index, edge_counts = self.generate(
            data=data, clusts=clusts, dist_mat=dist_mat
        )

        # Cut on the edge length, if specified
        if self.max_length is not None:
            assert (
                dist_mat is not None
            ), "Must provide `dist_mat` to restrict edge length."
            edge_index, edge_counts = self.restrict(
                edge_index, edge_counts, dist_mat, classes
            )

        # Disconnect nodes from separate groups, if specified
        if groups is not None:
            groups = groups.tensor
            mask = np.where(groups[edge_index[0]] == groups[edge_index[1]])[0]
            edge_index = edge_index[:, mask]
            edge_counts = self.update_counts(edge_counts, mask)

        # If the graph is directed, add reciprocal edges
        if not self.directed:
            full_index = np.empty((2, 2 * edge_index.shape[1]), dtype=np.int64)
            full_index[:, ::2] = edge_index
            full_index[:, 1::2] = np.flip(edge_index, axis=0)

            edge_index = full_index
            edge_counts = 2 * edge_counts

        # If there is a maximum count and the set of edges exceeds it, remove
        if self.max_count is not None and (edge_counts > self.max_count).any():
            batch_ids = np.repeat(np.arange(len(edge_counts)), edge_counts)
            mask = np.where(edge_counts[batch_ids] <= self.max_count)[0]
            edge_index = edge_index[:, mask]

            batch_mask = np.where(edge_counts > self.max_count)[0]
            edge_counts[batch_mask] = 0
            warn(
                f"Found too many edges in {len(batch_mask)} entry(ies) of the "
                "batch. There will be no aggregation predictions for such"
                "entries (all edges removed from the graph)."
            )

        # Get the offsets, initialize an EdgeIndexBatch obejct
        offsets = clusts.edges[:-1]
        edge_index = EdgeIndexBatch(edge_index, edge_counts, offsets, self.directed)

        return edge_index, dist_mat, closest_index

    def generate(self):
        """This function must be overridden in the constructor definition."""
        raise NotImplementedError("Must define the `generate` function")

    def restrict(self, edge_index, edge_counts, dist_mat, classes=None):
        """Function that restricts an incidence matrix of a graph
        to the edges below a certain length.

        If `classes` are specified, the maximum edge length must be provided
        for each possible combination of node classes.

        Parameters
        ----------
        edge_index : np.ndarray
            (2, E) Tensor of edges
        edge_counts : np.ndarray
            (B) : Number of edges in each entry of the batch
        dist_mat : np.ndarray
            (C, C) Tensor of pair-wise cluster distances
        classes : TensorBatch, optional
            (C) List of class for each cluster in the graph

        Returns
        -------
        np.ndarray
            (2,E) Restricted tensor of edges
        """
        # Restrict the input set of edges based on a edge length cut
        if classes is None or np.isscalar(self.max_length):
            # If classes are not provided, apply a static cut to all edges
            dists = dist_mat[(edge_index[0], edge_index[1])]
            mask = np.where(dists < self.max_length)[0]

        else:
            # If classes are provided, apply the cut based on the class
            dists = dist_mat[(edge_index[0], edge_index[1])]
            edge_classes = classes.tensor[edge_index]
            max_lengths = self.max_length[(edge_classes[0], edge_classes[1])]
            mask = np.where(dists < max_lengths)[0]

        # Update the number of edges in each entry of the batch
        edge_counts = self.update_counts(edge_counts, mask)

        return edge_index[:, mask], edge_counts

    def update_counts(self, counts, mask):
        """Updates the number of elements per entry in the batch, provided
        a mask which restricts the number of valid elements in the batch.

        Parameters
        ----------
        counts : np.ndarray
            (B) : Number of elements in each entry of the batch
        mask : np.ndarray
            Mask to apply to the list of elements

        Returns
        -------
        np.ndarray
            (B) Updated number of elements in each entry of the batch
        """
        # Get the batch ID of each elemnt in the input
        batch_size = len(counts)
        batch_ids = np.repeat(np.arange(batch_size), counts)[mask]

        # Get the new count list
        counts = np.zeros(batch_size, dtype=np.int64)
        if len(batch_ids):
            # Find the length of each batch ID in the input index
            uni, cnts = np.unique(batch_ids, return_counts=True)
            counts[uni.astype(int)] = cnts

        return counts
