"""Supervi dense clustering model and its loss."""

import MinkowskiEngine as ME
import numpy as np
import torch

from spine.data import IndexBatch, TensorBatch
from spine.utils.cluster.graph import ClusterGraphConstructor
from spine.utils.enums import enum_factory
from spine.utils.globals import (
    COORD_COLS,
    DELTA_SHP,
    MICHL_SHP,
    SHAPE_COL,
    SHOWR_SHP,
    TRACK_SHP,
)

from .layer.cluster import kernel_factory, loss_factory
from .layer.cluster.graph_spice_embedder import GraphSPICEEmbedder

__all__ = ["GraphSPICE", "GraphSPICELoss"]


class GraphSPICE(torch.nn.Module):
    """Graph Scalable Proposal-free Instance Clustering Engine (Graph-SPICE).

    Graph-SPICE has two components:
    1. Voxel embedder: UNet-type CNN architecture used for feature
       extraction and feature embeddings.
    2. Edge probability kernel function: A kernel function (any callable
       that takes two node attribute vectors to give a edge proability score).

    Prediction is done in two steps:
    1. A neighbor graph (ex. KNN, Radius) is constructed to compute
       edge probabilities between neighboring edges;
    2. Edges with low probability scores are dropped;
    3. The voxels are clustered through connected component clustering.

    A typical configuration is broken down into multiple components:

    .. code-block:: yaml

        model:
          name: graph_spice
          modules:
            graph_spice:
              <Basic parameters>
              embedder:
                <Feature embedding configuration block>
              kernel:
                <Edge kernel function configuration block>
              constructor:
                <Graph construction base parameters>
                graph:
                  <Graph configuration block>
                orphan:
                  <Orphan assignment configuration block>

    See configuration file(s) prefixed with `graph_spice` under the `config`
    directory for detailed examples of working configurations.
    """

    MODULES = ["constructor", "embedder", "kernel"]

    def __init__(self, graph_spice, graph_spice_loss=None):
        """Initialize the Graph-SPICE model.

        Parameters
        ----------
        graph_spice : dict
            Graph-SPICE configuration dictionary
        graph_spice_loss : dict, optional
            Graph-SPICE loss configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the model configuration
        self.process_model_config(**graph_spice)

    def process_model_config(
        self,
        embedder,
        kernel,
        constructor,
        shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
        use_raw_features=False,
        invert=True,
        make_clusters=False,
    ):
        """Initialize the underlying modules.

        Parameters
        ----------
        embedder : dict
            Pixel embedding configuration
        kernel : dict
            Edge kernel configuration
        constructor : dict
            Edge index construction configuration
        shapes : List[str]
            List of shape names to construct clusters for
        use_raw_features : bool, default True
            Use the list of embedder features as is, without the output layers
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        make_clusters : bool, default False
            If `True`, builds a list of cluster indexes
        """
        # Initialize the embedder
        self.embedder = GraphSPICEEmbedder(
            **embedder, use_raw_features=use_raw_features
        )

        # Initialize the kernel function (must be owned here to be loaded)
        self.kernel_fn = kernel_factory(kernel)

        # Initialize the graph constructor
        self.constructor = ClusterGraphConstructor(
            **constructor,
            kernel_fn=self.kernel_fn,
            shapes=shapes,
            invert=invert,
            training=self.training,
        )

        # Parse the set of shapes to cluster
        self.shapes = enum_factory("shape", shapes)

        # Store model parameters
        self.use_raw_features = use_raw_features
        self.invert = invert
        self.make_clusters = make_clusters

    def filter_class(self, data, seg_label, clust_label=None):
        """Filter the list of pixels to those in the list of requested shapes.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels

        Parameters
        ----------
        data : TensorBatch
            (M, 1+ + D + Nf) restricted tensor of voxel/value pairs
        seg_label : TensorBatch
            (M, 1 + D + 1) restricted tensor of segmentation labels
        clust_label : TensorBatch
            (M, 1 + D + N_c) Restricted tnesor of cluster labels
        index : torch.Tensor
            (M) Index to narrow down the original tensor
        counts : torch.Tensor
            (B) Number of restricted points in each batch entry
        """
        # Convert shapes to a torch tensor for easy comparison
        shapes = torch.tensor(self.shapes, device=data.device)

        # Create an index of the valid input rows
        mask = (seg_label.tensor[:, SHAPE_COL] == shapes.view(-1, 1)).any(dim=0)
        index = torch.where(mask)[0]

        # Restrict the input
        offsets = data.edges[:-1]
        data = TensorBatch(
            data.tensor[index], batch_size=data.batch_size, has_batch_col=True
        )

        # Restrict the label tensors
        assert seg_label.shape[0] == mask.shape[0], (
            "The segmentation label tensor is of the wrong shape: "
            f"{seg_label.shape[0]} != {mask.shape[0]}"
        )
        seg_label = TensorBatch(seg_label.tensor[index], data.counts)

        if clust_label is not None:
            assert clust_label.shape[0] == mask.shape[0], (
                "The cluster label tensor is of the wrong shape: "
                f"{clust_label.shape[0]} != {mask.shape[0]}"
            )
            clust_label = TensorBatch(clust_label.tensor[index], data.counts)

        # Store the index as an IndexBatch
        index = IndexBatch(index, offsets, data.counts)

        return data, seg_label, clust_label, index

    def forward(self, data, seg_label, clust_label=None):
        """Run a batch of data through the forward function.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels

        Returns
        -------
        dict
            Dictionary of outputs
        """
        # Filter the input down to the requested shapes
        data, seg_label, clust_label, index = self.filter_class(
            data, seg_label, clust_label
        )

        # Embed the input pixels into a feature space used for graph clustering
        result = self.embedder(data)

        # Store the index and the counts to not have to recompute them later
        result["filter_index"] = index

        # Build the graph on the pixel set
        coords = result["coordinates"]
        if self.use_raw_features:
            features = result["features"]
        else:
            features = result["hypergraph_features"]

        coords = TensorBatch(coords.data[:, coords.coord_cols], coords.counts)
        graph = self.constructor(coords, features, seg_label, clust_label)

        # If requested, convert edge predictions to node predictions
        if self.make_clusters:
            clusts, clust_shapes = self.constructor.fit_predict(graph)
            result["clusts"] = clusts
            result["clust_shapes"] = clust_shapes

        # Save the graph dictionary
        result.update(graph)

        return result


class GraphSPICELoss(torch.nn.Module):
    """Loss function for Graph-SPICE.

    For use in config:

    ..  code-block:: yaml

        model:
          name: graph_spice
          modules:
            graph_spice_loss:
              <Basic parameters>
              edge_loss:
                <Edge loss configuration block>

    See configuration files prefixed with `graph_spice` under the `config`
    directory for detailed examples of working configurations.

    See Also
    --------
    :class:`GraphSPICE`
    """

    def __init__(self, graph_spice, graph_spice_loss=None):
        """Intialize the Graph-SPICE loss.

        Parameters
        ----------
        graph_spice : dict
            Graph-SPICE configuration dictionary
        graph_spice_loss : dict
            Graph-SPICE loss configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Process the loss configuration
        self.process_loss_config(**graph_spice_loss)

        # Process the main mode configuration for its crucial elements
        self.process_model_config(**graph_spice)

    def process_loss_config(self, evaluate_clustering_metrics=False, **loss):
        """Process the loss configuration

        Parameters
        ----------
        evaluate_clustering_metrics : bool, default False
            If `True`, evaluates the clustering accuracy directly, rather than
            simply reporting an edge assignment acurracy
        **loss : dict
            Loss configurationd dictionary
        """
        # Store basic parameters
        self.evaluate_clustering_metrics = evaluate_clustering_metrics

        # Initialize the loss function
        self.loss_fn = loss_factory(loss)

    def process_model_config(
        self,
        constructor,
        shapes=[SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
        invert=True,
        **kwargs,
    ):
        """Process the model configuration

        Parameters
        ----------
        constructor : dict, optional
            Edge index construction configuration
        shapes : List[int], default [0, 1, 2, 3]
            List of semantic shapes to run DBSCAN on
        invert : bool, default True
            Invert the edge scores so that 0 is on an 1 is off
        """
        # Initialize the graph constructor (used to produce node assignments)
        if self.evaluate_clustering_metrics:
            self.constructor = ClusterGraphConstructor(
                **constructor, shapes=shapes, invert=invert
            )

    def filter_class(self, seg_label, clust_label, filter_index):
        """Filter the list of pixels to those in the list of requested shapes.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        filter_index : IndexBatch
            (M) Index to narrow down the original tensor

        Parameters
        ----------
        seg_label : TensorBatch
            (M, 1 + D + 1) restricted tensor of segmentation labels
        clust_label : TensorBatch
            (M, 1 + D + N_c) Restricted tnesor of cluster labels
        """
        seg_label = TensorBatch(
            seg_label.tensor[filter_index.index], filter_index.counts
        )
        clust_label = TensorBatch(
            clust_label.tensor[filter_index.index], filter_index.counts
        )

        return seg_label, clust_label

    def forward(self, seg_label, clust_label, filter_index, **output):
        """Run a batch of data through the loss function.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labelresul
            - N_c is is the number of cluster labels
        filter_index : IndexBatch
            (M) Index to narrow down the original tensor
        **output : dict
            Output of the Graph-SPICE model

        Returns
        -------
        dict
            Dictionary of outputs
        """
        # Narrow down the labels to those corresponding to the relevant shapes
        seg_label, clust_label = self.filter_class(seg_label, clust_label, filter_index)

        # Pass the output through the loss function
        result = self.loss_fn(seg_label=seg_label, clust_label=clust_label, **output)

        # If requested, compute clustering metrics
        if self.evaluate_clustering_metrics:
            # Assign cluster IDs to each of the input points, if not yet done
            if "node_pred" not in output:
                self.constructor.fit_predict(output)

            # Evaluate clustering metrics
            metrics = self.constructor.evaluate(output, mean=True)

            # Append metrics to the result dictionary
            result.update(metrics)

        return result
