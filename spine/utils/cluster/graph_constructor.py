"""Class and methods to convert the output of Graph-SPICE into a set of
pixel cluster assignments using a graph.
"""

import sys
from collections import defaultdict
from typing import Union, Callable, Tuple, List, Dict
from functools import partial

import numpy as np
import torch
from torch_cluster import knn_graph, radius_graph

from spine.data import TensorBatch, ObjectList
from spine.utils.globals import CLUST_COL, SHAPE_COL
from spine.utils.metrics import pur, eff, ari

from .ccc import ConnectedComponentClusterer

__all__ = ['ClusterGraphConstructor']


class ClusterGraphConstructor:
    """Manager class for handling per-batch, per-semantic type graph
    construction and node predictions in Graph-SPICE clustering.
    """

    def __init__(self, graph, classes, edge_threshold, kernel_fn=None,
                 min_size=0, invert=True, label_edges=False,
                 target_col=CLUST_COL, training=False, orphan=None):
        """Initialize the cluster graph constructor.

        Parameters
        ----------
        graph : dict
            Graph construction configuration dictionary
        classes : List[int]
            List of classes to construct clusters for
        edge_threshold : float
            Edge score below which it is disconnected (or above which it is,
            if the `inverted` parameter is turned on
        kernel_fn : callable, optional
            Kernel function computing edge scores from edge features
        min_size : int, default 0
            Minimum number of points below which pixels are considered orphans
            to be merged into touching larger clusters
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        label_edges : bool, default False
            If `True`, use cluster labels to label the edges as on or off
        target_col : int, default CLUST_COL
            Index of the column which specifies the label cluster ID for each point
        training : bool, default False
            If `True`, this constructor is being used at train time
        orphan : dict, optional
            Orphan clustering configuration dictionary

        Raises
        ------
        ValueError
            If the graph type is not supported.
        """
        # Store basic properties
        self.classes = classes
        self.min_size = min_size
        self.invert = invert
        self.label_edges = label_edges
        self.kernel_fn = kernel_fn
        self.target_col = target_col

        # At train time, some of the parameters must be set to special values
        self.threshold = edge_threshold if not training else 0.

        # Partially instantiate the graph constructor functions
        assert 'name' in graph, (
                "Must provide the graph constructor function name.")

        name = graph.pop('name')
        if name == 'knn':
            self.graph_fn = partial(knn_graph, **graph)
        elif name == 'knn_sklearn':
            self.graph_fn = partial(knn_sklearn, **graph)
        elif name == 'radius':
            self.graph_fn = partial(radius_graph, **graph)
        else:
            raise ValueError(
                    f"Requested graph construction mode ('{name}') is not "
                     "recognized. Must be one of 'knn', 'knn_sklearn', "
                     "or 'radius'.")

        # Initialize the cluster assignment class
        self.ccc = ConnectedComponentClusterer(min_size, orphan)

    def __call__(self, coords, features, seg_label, clust_label=None):
        """Constructs graphs for all the entries in a batch, one per shape.

        Parameters
        ----------
        coords : TensorBatch
            (N, 3) Point coordinates
        features : TensorBatch
            (N, N_f) Set of graph embeddings
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        """
        # If edge labeling is required, make sure clust_label is provided
        assert not self.label_edges or clust_label is not None, (
                "If edge labels are to be produced, must provide `clust_label`.")

        # Loop over the unique batch indices, build a list of graphs for each
        graph = defaultdict(list)
        edge_counts = []
        for b in range(coords.batch_size):
            # Build graphs (one per semantic type)
            clust_label_b = clust_label[b] if clust_label is not None else None
            graphs_b, edge_count = self.build_graph(
                coords[b], features[b], seg_label[b], clust_label_b)

            # Append the output
            edge_counts.append(edge_count)
            for key, value in graphs_b.items():
                graph[key].append(value)

        # Concatenate the outputs together
        is_tensor = isinstance(coords.tensor, torch.Tensor)
        cat = torch.cat if is_tensor else np.concatenate
        for key, value in graph.items():
            if key.startswith('edge') and not key.endswith('clusts'):
                # Turn edge attributes into tensor batches
                value = cat(value)
                graph[key] = TensorBatch(value, counts=edge_counts)

        # Add the node information to the graph
        graph['node_coords'] = coords
        graph['node_features'] = features
        graph['node_shapes'] = TensorBatch(
                seg_label.tensor[:, SHAPE_COL], seg_label.counts)

        return graph

    def build_graph(self, coords, features, seg_label, clust_label=None):
        """Construct a graph for a single batch id and semantic class that
        will be used for connected components clustering.

        Parameters
        ----------
        coords : torch.Tensor
            (N, 3) Tensor of point coordinates
        features : torch.Tensor
            (N, N_f) Graph embedding features to be used for edge prediction
        seg_clusts : List[List[int]]
            (S) One pixel index per semantic type
        seg_label : torch.Tensor
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : torch.Tensor, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of graph properties
        """
        # Loop over the semantic types, build a graph for each
        graph = defaultdict(list)
        edge_count = 0
        for s in self.classes:
            # Get the index of points which belong to this class
            seg_index = torch.where(seg_label[:, SHAPE_COL] == s)[0]
            graph['node_clusts'].append(seg_index)

            # If there are no points, append empty, proceed
            if not len(seg_index):
                graph['edge_clusts'].append(torch.empty(
                    0, dtype=torch.long, device=coords.device))
                graph['edge_index'].append(torch.empty(
                    (0, 2), dtype=torch.long, device=coords.device))
                graph['edge_shape'].append(torch.empty(
                    0, dtype=torch.long, device=coords.device))
                graph['edge_attr'].append(torch.empty(
                    0, dtype=features.dtype, device=features.device))
                graph['edge_label'].append(torch.empty(
                    0, dtype=torch.long, device=coords.device))
                continue

            # Make the graph edge index
            edge_index = self.graph_fn(coords[seg_index])
            graph['edge_clusts'].append(
                    edge_count + torch.arange(
                        edge_index.shape[1], device=coords.device))
            edge_count += edge_index.shape[1]

            # Produce edge predictions
            features_s = features[seg_index]
            edge_attr = self.kernel_fn(
                    features_s[edge_index[0]], features_s[edge_index[1]])

            # Append
            graph['edge_index'].append(edge_index.T)
            graph['edge_shape'].append(torch.full(
                (edge_index.shape[1],), s, dtype=torch.long,
                device=coords.device))
            graph['edge_attr'].append(edge_attr.flatten())

            if self.label_edges:
                node_label = clust_label[seg_index, self.target_col]
                edge_label = (
                        node_label[edge_index[0]] == node_label[edge_index[1]])
                graph['edge_label'].append(edge_label.long())

        # Concatenate the graph attributes
        for key, value in graph.items():
            if key != 'node_clusts' and key != 'edge_clusts':
                graph[key] = torch.cat(value)

        # Convert edge logits to sigmoid scores
        graph['edge_prob'] = torch.sigmoid(graph['edge_attr'])

        # Assign edge predictions based on the edge scores
        if self.invert:
            graph['edge_pred'] = (graph['edge_prob'] <= self.threshold).long()
        else:
            graph['edge_pred'] = (graph['edge_prob'] >= self.threshold).long()

        return graph, edge_count
        
    @staticmethod
    def fit_predict(graph, edge_mode='edge_pred'):
        """Perform connected components clustering on a batch.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Node assignments
        """
        # No gradients through this prediction
        with torch.no_grad():
            node_pred = self.ccc(
                    graph['node_coords'], graph['edge_index'], graph[edge_mode],
                    graph['node_clusts'], graph['edge_clusts'])

            graph['node_pred'] = node_pred

            return node_pred
    
    @staticmethod
    def evaluate(graph, mean=False):
        """Evaluate the clustering accuracy of a graph.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape
        mean : bool, default False
            If `True`, returns the batch-averaged metric values

        Returns
        -------
        dict
            Dictionary of accuracy metrics
        """
        # No gradients through this evaluation
        result = defaultdict(list)
        metrics = {'ari': ari, 'purity': pur, 'efficiency': eff}
        with torch.no_grad():
            # Loop over the batches
            for b in range(batch_size):
                # Get the node predictions and labels
                node_label_b = graph['node_label'][b]
                node_pred_b = graph['node_pred'][b]

                # Compute shape-agnostic metrics
                for m, metric in metrics.items():
                    result[m].append(metric(node_pred_b, node_label_b))

                # Loop over the semantic types
                for s, shape in self.classes:
                    # Narrow down the predictions and labels to this shape
                    node_index = graph['node_clusts'][b][s]
                    node_label_b_s = node_label_b[node_index]
                    node_pred_b_s = node_pred_b[node_index]

                    # If there are no points of this type, append default values
                    if not len(node_index):
                        for m in metrics:
                            result[f'{m}_{shape}'].append(1.)

                    # Otherwise, compute the metrics
                    for m, metric in metrics.items():
                        result[f'{m}_{shape}'].append(
                                metric(node_pred_b_s, node_label_b_s))

        # Compute batch averaged metrics, return
        if mean:
            for key, value in result.items():
                result[key] = np.mean(value)
            
        return result

    @staticmethod
    def get_entry(graph, batch_id, semantic_id):
        """Narrow down the graph to one specific (batch_id, shape) pair.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape
        batch_id : int
            Batch index
        semantic_id : int
            Semantic type

        Returns
        -------
        dict
            Dictionary of graph attributes for one (batch_id, shape) pair
        """
        # Loop over graph keys, narrow down wherever relevant
        single_graph = {}
        node_index = graph['node_clusts'][batch_id][semantic_id]
        edge_index = graph['edge_clusts'][batch_id][semantic_id]
        for key, value in graph.items():
            if key.endswith('clusts'):
                continue
            elif key.startswith('node'):
                single_graph[key] = graph[key][b][node_index]
            elif key.startswith('edge'):
                single_graph[key] = graph[key][b][edge_index]
            else:
                raise KeyError(f"Graph key not recognized: {key}")

        return single_graph
