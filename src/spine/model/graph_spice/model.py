"""Supervised dense clustering model and its loss."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import torch

from spine.constants.factory import enum_factory
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.graph_spice.constructor import ClusterGraphConstructor

from ..registry import ModelSpec
from .embedder import GraphSPICEEmbedder
from .factories import kernel_factory, loss_factory

__all__ = ["GraphSPICE", "GraphSPICELoss"]

GraphSPICEOutput = dict[str, TensorBatch | IndexBatch]


class GraphSPICE(torch.nn.Module):
    """Graph Scalable Proposal-free Instance Clustering Engine (Graph-SPICE).

    Graph-SPICE has two main components:

    - A voxel embedder, implemented as a UNet-like CNN for feature extraction
      and embeddings
    - An edge probability kernel that maps pairs of node attribute vectors to
      edge scores

    Prediction proceeds in three stages:

    - A neighbor graph such as KNN or a radius graph is constructed
    - Edge probabilities are evaluated and low-probability edges are dropped
    - Voxels are clustered through connected-component clustering

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

    def __init__(self, graph_spice: dict[str, Any]) -> None:
        """Initialize the Graph-SPICE model.

        Parameters
        ----------
        graph_spice : dict
            Graph-SPICE configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Declare attributes populated by the configuration helper.
        self.embedder: GraphSPICEEmbedder
        self.kernel_fn: torch.nn.Module
        self.constructor: ClusterGraphConstructor
        self.shapes: list[int]
        self.use_raw_features: bool
        self.invert: bool
        self.make_clusters: bool

        # Initialize the model configuration
        self.process_model_config(**graph_spice)

    def process_model_config(
        self,
        embedder: dict[str, Any],
        kernel: dict[str, Any],
        constructor: dict[str, Any],
        shapes: list[str] | tuple[str, ...] = (
            "shower",
            "track",
            "michel",
            "delta",
        ),
        use_raw_features: bool = False,
        invert: bool = True,
        make_clusters: bool = False,
    ) -> None:
        """Initialize the underlying modules.

        Parameters
        ----------
        embedder : dict
            Pixel embedding configuration
        kernel : dict
            Edge kernel configuration
        constructor : dict
            Edge index construction configuration
        shapes : sequence of str
            List of shape names to construct clusters for
        use_raw_features : bool, default False
            Use the list of embedder features as is, without the output layers
        invert : bool, default True
            Invert the edge scores so that 0 is on and 1 is off
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
            **deepcopy(constructor),
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

    def filter_class(
        self,
        data: TensorBatch,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch | None = None,
    ) -> tuple[TensorBatch, TensorBatch, ClusterLabelBatch | None, IndexBatch]:
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
        clust_label : ClusterLabelBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels

        Returns
        -------
        data : TensorBatch
            (M, 1+ + D + Nf) restricted tensor of voxel/value pairs
        seg_label : TensorBatch
            (M, 1 + D + 1) restricted tensor of segmentation labels
        clust_label : ClusterLabelBatch
            (M, 1 + D + N_c) Restricted tensor of cluster labels
        index : torch.Tensor
            (M) Index to narrow down the original tensor
        """
        # Validate that the input tensors describe the same voxels before using
        # label-derived indexes to narrow the data tensor.
        if data.shape[0] != seg_label.shape[0]:
            raise ValueError(
                "The data and segmentation label tensors must have matching "
                f"row counts, got {data.shape[0]} and {seg_label.shape[0]}."
            )
        if clust_label is not None and data.shape[0] != clust_label.data.shape[0]:
            raise ValueError(
                "The data and cluster label tensors must have matching row counts, "
                f"got {data.shape[0]} and {clust_label.data.shape[0]}."
            )

        # Convert shapes to a tensor for easy comparison.
        seg_label_tensor = seg_label.values.torch_tensor()
        shapes = torch.as_tensor(
            self.shapes,
            dtype=seg_label_tensor.dtype,
            device=seg_label_tensor.device,
        )

        # Create an index of the valid input rows
        mask = (seg_label_tensor == shapes.view(-1, 1)).any(dim=0)
        index = torch.where(mask)[0]

        # Restrict the input
        spans = data.counts
        data = data.select(mask)

        # Restrict the label tensors
        seg_label = seg_label.select(mask)

        if clust_label is not None:
            clust_label = clust_label.select(index, data.counts)

        # Store the index as an IndexBatch
        index = IndexBatch(index, spans, data.counts)

        return data, seg_label, clust_label, index

    def forward(
        self,
        data: TensorBatch,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch | None = None,
    ) -> GraphSPICEOutput:
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
        clust_label : ClusterLabelBatch, optional
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
        embedder_result = self.embedder(data)

        # Store the index and the counts to not have to recompute them later
        result: GraphSPICEOutput = {**embedder_result, "filter_index": index}

        # Build the graph on the pixel set
        coordinate_batch = embedder_result["coordinates"]
        if self.use_raw_features:
            features = embedder_result["features"]
        else:
            features = embedder_result["hypergraph_features"]

        coord_cols = coordinate_batch.coord_cols
        if coord_cols is None:
            raise RuntimeError(
                "GraphSPICE coordinates do not define coordinate columns."
            )
        coords = TensorBatch(
            coordinate_batch.torch_tensor()[:, coord_cols],
            coordinate_batch.counts,
        )
        cluster_ids = None
        if clust_label is not None:
            cluster_ids = clust_label.voxel_field(self.constructor.target_col)
        graph = cast(
            GraphSPICEOutput,
            self.constructor(coords, features, seg_label, cluster_ids),
        )

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
              name: edge
              loss: <Binary loss configuration>

    See configuration files prefixed with `graph_spice` under the `config`
    directory for detailed examples of working configurations.

    See Also
    --------
    :class:`GraphSPICE`
    """

    def __init__(
        self,
        graph_spice: dict[str, Any],
        graph_spice_loss: dict[str, Any],
    ) -> None:
        """Initialize the Graph-SPICE loss.

        Parameters
        ----------
        graph_spice : dict
            Graph-SPICE configuration dictionary
        graph_spice_loss : dict
            Graph-SPICE loss configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Declare attributes populated by the configuration helpers.
        self.evaluate_clustering_metrics: bool
        self.loss_fn: torch.nn.Module
        self.constructor: ClusterGraphConstructor | None = None
        self.target_col: str

        # Process the loss configuration
        self.process_loss_config(**graph_spice_loss)

        # Process the main mode configuration for its crucial elements
        self.process_model_config(**graph_spice)

    def process_loss_config(
        self, evaluate_clustering_metrics: bool = False, **loss: Any
    ) -> None:
        """Process the loss configuration.

        Parameters
        ----------
        evaluate_clustering_metrics : bool, default False
            If `True`, evaluates the clustering accuracy directly, rather than
            simply reporting an edge assignment accuracy
        **loss : dict, optional
            Loss configuration dictionary
        """
        # Store basic parameters
        self.evaluate_clustering_metrics = evaluate_clustering_metrics

        # Initialize the loss function
        self.loss_fn = loss_factory(loss)

    def process_model_config(
        self,
        constructor: dict[str, Any],
        shapes: list[str] | tuple[str, ...] = (
            "shower",
            "track",
            "michel",
            "delta",
        ),
        invert: bool = True,
        **_kwargs: Any,
    ) -> None:
        """Process the model configuration.

        Parameters
        ----------
        constructor : dict
            Edge index construction configuration
        shapes : sequence of str, default ("shower", "track", "michel", "delta")
            Semantic shapes to cluster
        invert : bool, default True
            Invert the edge scores so that 0 is on and 1 is off
        **_kwargs : dict, optional
            Other model parameters not needed by the loss
        """
        # Initialize the graph constructor (used to produce node assignments)
        self.target_col = constructor.get("target_col", "cluster")
        if self.evaluate_clustering_metrics:
            self.constructor = ClusterGraphConstructor(
                **deepcopy(constructor), shapes=shapes, invert=invert
            )

    @staticmethod
    def filter_class(
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch,
        filter_index: IndexBatch,
    ) -> tuple[TensorBatch, ClusterLabelBatch]:
        """Filter the list of pixels to those in the list of requested shapes.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        filter_index : IndexBatch
            (M) Index to narrow down the original tensor

        Returns
        -------
        seg_label : TensorBatch
            (M, 1 + D + 1) restricted tensor of segmentation labels
        clust_label : ClusterLabelBatch
            (M, 1 + D + N_c) Restricted tensor of cluster labels
        """
        index = filter_index.index
        mask = torch.zeros(
            len(seg_label.tensor), dtype=torch.bool, device=seg_label.device
        )
        mask[index] = True
        seg_label = seg_label.select(mask)
        clust_label = clust_label.select(index, filter_index.counts)

        return seg_label, clust_label

    def get_edge_labels(
        self,
        clust_label: ClusterLabelBatch,
        edge_index: TensorBatch,
        node_clusts: IndexBatch,
        edge_clusts: IndexBatch,
    ) -> TensorBatch:
        """Build binary edge labels from the target node assignments.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Cluster labels narrowed to the nodes used by Graph-SPICE.
        edge_index : TensorBatch
            ``(E, 2)`` local endpoint indexes for every semantic subgraph.
        node_clusts : IndexBatch
            Node indexes grouped by batch entry and semantic class.
        edge_clusts : IndexBatch
            Edge indexes grouped by batch entry and semantic class.

        Returns
        -------
        TensorBatch
            ``(E,)`` labels which are one when both endpoints have the same
            target cluster ID.
        """
        if not node_clusts.is_list or not edge_clusts.is_list:
            raise TypeError("Graph node and edge groups must be index lists.")
        if len(node_clusts.index_list) != len(edge_clusts.index_list):
            raise ValueError(
                "Graph node and edge groups must contain the same number of "
                "semantic subgraphs."
            )

        cluster_ids = clust_label.voxel_field(self.target_col).torch_tensor()
        edges = edge_index.torch_tensor().long()
        labels = torch.empty(
            edge_index.shape[0],
            dtype=torch.long,
            device=edges.device,
        )
        labeled = torch.zeros(
            edge_index.shape[0],
            dtype=torch.bool,
            device=edges.device,
        )

        for node_index, edge_group in zip(
            node_clusts.index_list,
            edge_clusts.index_list,
        ):
            if not isinstance(node_index, torch.Tensor) or not isinstance(
                edge_group, torch.Tensor
            ):
                raise TypeError("Graph index groups must be PyTorch tensors.")
            if len(edge_group) == 0:
                continue

            node_index = cast(torch.Tensor, node_index).long()
            edge_group = cast(torch.Tensor, edge_group).long()
            local_edges = edges[edge_group]
            if torch.any(local_edges < 0) or torch.any(local_edges >= len(node_index)):
                raise IndexError(
                    "Graph edge endpoints fall outside their semantic node group."
                )

            node_cluster_ids = cluster_ids[node_index]
            labels[edge_group] = (
                node_cluster_ids[local_edges[:, 0]]
                == node_cluster_ids[local_edges[:, 1]]
            ).long()
            labeled[edge_group] = True

        if not bool(torch.all(labeled)):
            raise ValueError("Graph edge groups do not cover every edge.")

        return TensorBatch(labels, edge_index.counts)

    def forward(
        self,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch,
        filter_index: IndexBatch,
        **output: Any,
    ) -> dict[str, Any]:
        """Run a batch of data through the loss function.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels
            - 1 is the segmentation label
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_c) Tensor of cluster labels
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
        seg_label, clust_label = self.filter_class(
            seg_label,
            clust_label,
            filter_index,
        )

        # New configurations keep truth labels out of the network. Derive the
        # supervision targets here unless a legacy network invocation already
        # produced them.
        if "edge_label" not in output:
            output["edge_label"] = self.get_edge_labels(
                clust_label,
                output["edge_index"],
                output["node_clusts"],
                output["edge_clusts"],
            )

        # Pass the output through the loss function
        result = self.loss_fn(seg_label=seg_label, clust_label=clust_label, **output)

        # If requested, compute clustering metrics
        if self.evaluate_clustering_metrics:
            constructor = self.constructor
            assert constructor is not None

            # Assign cluster IDs to each of the input points, if not yet done
            if "node_pred" not in output:
                constructor.fit_predict(output)

            # Evaluate clustering metrics
            metrics = constructor.evaluate(output, mean=True)

            # Append metrics to the result dictionary
            result.update(metrics)

        return result


MODEL_SPEC = ModelSpec("graph_spice", GraphSPICE, GraphSPICELoss)
