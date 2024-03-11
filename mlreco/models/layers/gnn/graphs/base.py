"""Base class for all graph construction classes."""

import inspect
import numpy as np
from typing import List, Union

from mlreco.utils.globals import COORD_COLS
from mlreco.utils.data_structures import EdgeIndexBatch
from mlreco.utils.gnn.network import inter_cluster_distance


class GraphBase:
    """Parent class for all graph constructors."""

    def __init__(self, directed=False, max_length=None, classes=None,
                 max_count=None, dist_method='voxel', dist_algorithm='brute'):
        """Initializes attributes shared accross all graph constructors.
        
        Parameters
        ----------
        directed : bool, default False
            If `True`, direct the edges from lower to higher rank only
        max_length : Union[float, np.ndarray], optional
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
            Algorithm used to comppute inter-node distance
            ('brute' or 'recursive')
        """
        # Store attributes
        self.directed = directed
        self.max_count = max_count
        self.dist_method = dist_method
        self.dist_algorithm = dist_algorithm

        # Convert `max_length` to a matrix, if provided as a `triu`
        self.max_length = max_length
        if isinstance(max_length, list):
            assert classes is not None, (
                    "If specifying the edge length cut per class, "
                    "must provide the list of classes")

            num_classes = np.max(classes) + 1
            assert len(max_length) == num_classes*(num_classes + 1)/2, (
                    "If provided as a list, the maximum edge length should be "
                    "given for each upper triangular element of a matrix of "
                    "size num_classes*num_classes")

            max_length_mat = np.zeros((num_classes, num_classes), dtype=float)
            max_length_mat[np.triu_indices(num_classes)] = max_length
            max_length_mat += (max_length_mat.T - 
                               np.diag(np.diag(max_length_mat)))

            self.max_length = max_length

        # Store whether the inter-cluster distance matrix must be evaluated
        self.compute_dist = (max_length is not None or 
                             self.name in ['mst', 'knn'])

        # If this is a loop graph, simply set as undirected
        assert self.name != 'loop' or self.directed, (
                "For loop graphs, set as directed (no need for reciprocal)")

    def __call__(self, data, clusts, classes=None, groups=None):
        """Filters input to keep only what is needed to generate a graph.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
        clusts : IndexBatch
            (C) Cluster indexes
        classes : np.ndarray, optional
            (C) List of cluster semantic class used to define the max length
        groups : np.ndarray, optional
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
                    data.tensor[:, COORD_COLS], clusts.index_list,
                    clusts.batch_ids, method=self.dist_method,
                    algorithm=self.dist_algorithm, return_index=True)

        # Generate the edge index
        edge_index, edge_counts = self.generate(data=data, clusts=clusts)

        # Cut on the edge length, if specified
        if self.max_length is not None:
            assert dist_mat is not None, (
                    "Must provide `dist_mat` to restrict edge length.")
            edge_index = self.restrict(edge_index, dist_mat, classes)

        # Disconnect nodes from separate groups, if specified
        if groups is not None:
            index = np.where(groups[edge_index[0]] == groups[edge_index[1]])[0]
            edge_index = edge_index[:, index]

        # Get the offsets, initialize an EdgeIndexBatch obejct
        offsets = clusts.list_edges[:-1]
        edge_index = EdgeIndexBatch(
                edge_index, edge_counts, offsets, self.directed)

        return edge_index, dist_mat, closest_index

    def generate(self):
        """This function must be overridden in the constructor definition."""
        raise NotImplementedError("Must define the `generate` function")

    def restrict(edge_index, dist_mat, classes=None):
        """Function that restricts an incidence matrix of a graph
        to the edges below a certain length.

        If `classes` are specified, the maximum edge length must be provided
        for each possible combination of node classes.

        Parameters
        ----------
        edge_index : np.ndarray
            (2,E) Tensor of edges
        dist_mat : np.ndarray
            (C,C) Tensor of pair-wise cluster distances
        classes : np.ndarray, optional
            (C) List of class for each cluster in the graph

        Returns
        -------
        np.ndarray
            (2,E) Restricted tensor of edges
        """
        # Restrict the input set of edges based on a edge length cut
        if classes is None:
            # If classes are not provided, apply a static cut to all edges
            dists = dist_mat[(edge_index[0], edge_index[1])]

            return edge_index[dists < max_length]

        else:
            # If classes are provided, apply the cut based on the class
            dists = dist_mat[(edge_index[0], edge_index[1])]
            edge_classes = classes[edge_index]
            max_lengths = self.max_length[(edge_classes[0], edge_classes[1])]

            return edge_index[:, dists < max_lengths]


