"""Class and methods to convert the output of Graph-SPICE into a set of
pixel cluster assignments using a graph.
"""

import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch_cluster import knn_graph, radius_graph

from spine.data import IndexBatch, ObjectList, TensorBatch
from spine.utils.enums import enum_factory
from spine.utils.globals import CLUST_COL, SHAPE_COL
from spine.utils.gnn.cluster import form_clusters
from spine.utils.metrics import ari, eff, pur

from .ccc import ConnectedComponentClusterer

__all__ = ["ClusterGraphConstructor"]


class ClusterGraphConstructor:
    """Manager class for handling per-batch, per-semantic type graph
    construction and node predictions in Graph-SPICE clustering.
    """

    def __init__(
        self,
        graph,
        shapes,
        edge_threshold,
        kernel_fn=None,
        min_size=0,
        invert=True,
        label_edges=False,
        target_col=CLUST_COL,
        training=False,
        orphan=None,
    ):
        """Initialize the cluster graph constructor.

        Parameters
        ----------
        graph : dict
            Graph construction configuration dictionary
        shapes : List[str]
            List of shape names to construct clusters for
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
        # Parse the set of shapes to cluster
        self.shapes = enum_factory("shape", shapes)

        # Store other basic properties
        self.threshold = edge_threshold
        self.min_size = min_size
        self.invert = invert
        self.label_edges = label_edges
        self.kernel_fn = kernel_fn
        self.target_col = target_col

        # Partially instantiate the graph constructor functions
        assert "name" in graph, "Must provide the graph constructor function name."

        name = graph.pop("name")
        if name == "knn":
            self.graph_fn = partial(knn_graph, **graph)
        elif name == "radius":
            self.graph_fn = partial(radius_graph, **graph)
        else:
            raise ValueError(
                f"Requested graph construction mode ('{name}') is not "
                "recognized. Must be one of 'knn' or 'radius'"
            )

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
        assert (
            not self.label_edges or clust_label is not None
        ), "If edge labels are to be produced, must provide `clust_label`."

        # Loop over the unique batch indices, build a list of graphs for each
        graph = defaultdict(list)
        edge_offset = 0
        edge_counts, edge_offsets = [], []
        for b in range(coords.batch_size):
            # Build graphs (one per semantic type)
            clust_label_b = clust_label[b] if clust_label is not None else None
            graphs_b, edge_count = self.build_graph(
                coords[b], features[b], seg_label[b], clust_label_b
            )

            # Append the output
            edge_counts.append(edge_count)
            edge_offsets.append(edge_offset)
            for key, value in graphs_b.items():
                if key.endswith("clusts"):
                    is_edge = key.startswith("edge")
                    offset = edge_offset if is_edge else coords.edges[b]
                    for c in value:
                        graph[key].append(offset + c)

                else:
                    graph[key].append(value)

            # Increment the offset
            edge_offset += edge_count

        # Concatenate the graph attributes together
        is_tensor = isinstance(coords.tensor, torch.Tensor)
        cat = torch.cat if is_tensor else np.concatenate
        for key, value in graph.items():
            if key.endswith("clusts"):
                # Turn indexes into index batches
                counts = [len(self.shapes)] * coords.batch_size
                single_counts = [len(c) for c in value]
                is_edge = key.startswith("edge")
                offsets = edge_offsets if is_edge else coords.edges[:-1]
                graph[key] = IndexBatch(value, offsets, counts, single_counts)

            else:
                # Turn edge index/attributes into tensor batches
                value = cat(value)
                graph[key] = TensorBatch(value, edge_counts)

        # Add the input node information to the graph
        graph["node_coords"] = coords
        graph["node_features"] = features
        graph["node_shapes"] = TensorBatch(
            seg_label.tensor[:, SHAPE_COL], seg_label.counts
        )

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
        for s in self.shapes:
            # Get the index of points which belong to this class
            seg_index = torch.where(seg_label[:, SHAPE_COL] == s)[0]
            graph["node_clusts"].append(seg_index)

            # If there are no points, append empty, proceed
            if not len(seg_index):
                graph["edge_clusts"].append(
                    torch.empty(0, dtype=torch.long, device=coords.device)
                )
                graph["edge_index"].append(
                    torch.empty((0, 2), dtype=torch.long, device=coords.device)
                )
                graph["edge_shape"].append(
                    torch.empty(0, dtype=torch.long, device=coords.device)
                )
                graph["edge_attr"].append(
                    torch.empty(0, dtype=features.dtype, device=features.device)
                )
                if self.label_edges:
                    graph["edge_label"].append(
                        torch.empty(0, dtype=torch.long, device=coords.device)
                    )
                continue

            # Make the graph edge index
            edge_index = self.graph_fn(coords[seg_index])
            graph["edge_clusts"].append(
                edge_count + torch.arange(edge_index.shape[1], device=coords.device)
            )
            edge_count += edge_index.shape[1]

            # Produce edge predictions
            features_s = features[seg_index]
            edge_attr = self.kernel_fn(
                features_s[edge_index[0]], features_s[edge_index[1]]
            )

            # Append
            graph["edge_index"].append(edge_index.T)
            graph["edge_shape"].append(
                torch.full(
                    (edge_index.shape[1],), s, dtype=torch.long, device=coords.device
                )
            )
            graph["edge_attr"].append(edge_attr.flatten())

            if self.label_edges:
                node_label = clust_label[seg_index, self.target_col]
                edge_label = node_label[edge_index[0]] == node_label[edge_index[1]]
                graph["edge_label"].append(edge_label.long())

        # Concatenate the graph attributes
        for key, value in graph.items():
            if key != "node_clusts" and key != "edge_clusts":
                graph[key] = torch.cat(value)

        # Convert edge logits to sigmoid scores
        graph["edge_prob"] = torch.sigmoid(graph["edge_attr"])

        return graph, edge_count

    def fit_predict(self, graph, edge_mode="edge_pred", threshold=None, min_size=None):
        """Perform connected components clustering on a batch.

        Parameters
        ----------
        graph : dict
            Dictionary of graph attributes organized by batch and shape
        edge_mode : str, default 'edge_pred'
            Attribute of the graph used to get the edge status
        threshold : float, optional
            Override the edge score threshold set in the configuration
        min_size : int, optional
            Override the minimum cluster size set in the configuration

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Node assignments
        """
        # No gradients through this prediction
        with torch.no_grad():
            # Assign edge predictions based on the edge scores
            threshold = threshold if threshold is not None else self.threshold
            if self.invert:
                edge_pred = (graph["edge_prob"].tensor <= threshold).long()
            else:
                edge_pred = (graph["edge_prob"].tensor >= threshold).long()

            graph["edge_pred"] = TensorBatch(edge_pred, graph["edge_prob"].counts)

            # Assign each node to a cluster
            node_pred = self.ccc(
                graph["node_coords"],
                graph["edge_index"],
                graph[edge_mode],
                graph["node_clusts"],
                graph["edge_clusts"],
            )

            graph["node_pred"] = node_pred

            # Loop over entries in the batch, build fragments
            node_clusts = graph["node_clusts"]
            clusts, counts, single_counts, shapes = [], [], [], []
            for b in range(node_pred.batch_size):
                # Loop over shapes in the entry
                counts_b = 0
                for s, shape in enumerate(self.shapes):
                    # Get the list of clusters for this (entry, shape) pair
                    index_b_s = node_clusts[b][s]
                    clusts_b_s, counts_b_s = form_clusters(
                        node_pred[b][index_b_s, None], column=0
                    )

                    # Offset the cluster indexes appropriately, append
                    for i, c in enumerate(clusts_b_s):
                        clusts_b_s[i] = node_pred.edges[b] + index_b_s[c]

                    # Append
                    clusts.extend(clusts_b_s)
                    single_counts.extend(counts_b_s)
                    shapes.append(shape * np.ones(len(clusts_b_s), dtype=int))
                    counts_b += len(clusts_b_s)

                counts.append(counts_b)

            # Make an IndexBatch out of the list
            clusts = IndexBatch(clusts, node_pred.edges[:-1], counts, single_counts)
            clust_shapes = TensorBatch(np.concatenate(shapes), counts)

            # Return
            return clusts, clust_shapes

    def evaluate(self, graph, mean=False):
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
        metrics = {"ari": ari, "purity": pur, "efficiency": eff}
        with torch.no_grad():
            # Loop over the batches
            for b in range(batch_size):
                # Get the node predictions and labels
                node_label_b = graph["node_label"][b]
                node_pred_b = graph["node_pred"][b]

                # Compute shape-agnostic metrics
                for m, metric in metrics.items():
                    result[m].append(metric(node_pred_b, node_label_b))

                # Loop over the semantic types
                for s, shape in self.shapes:
                    # Narrow down the predictions and labels to this shape
                    node_index = graph["node_clusts"][b][s]
                    node_label_b_s = node_label_b[node_index]
                    node_pred_b_s = node_pred_b[node_index]

                    # If there are no points of this type, append default values
                    if not len(node_index):
                        for m in metrics:
                            result[f"{m}_{shape}"].append(1.0)

                    # Otherwise, compute the metrics
                    for m, metric in metrics.items():
                        result[f"{m}_{shape}"].append(
                            metric(node_pred_b_s, node_label_b_s)
                        )

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
        node_index = graph["node_clusts"][batch_id][semantic_id]
        edge_index = graph["edge_clusts"][batch_id][semantic_id]
        for key, value in graph.items():
            if key.endswith("clusts"):
                continue
            elif key.startswith("node"):
                single_graph[key] = graph[key][b][node_index]
            elif key.startswith("edge"):
                single_graph[key] = graph[key][b][edge_index]
            else:
                raise KeyError(f"Graph key not recognized: {key}")

        return single_graph
