"""GrapPA: Graph Neural Network for Particle Aggregation.

This module implements the GrapPA (Graph Particle Aggregation) architecture,
a graph neural network designed for clustering and grouping particle instances.

GrapPA learns to aggregate fragment-level features into particle-level clusters
through message passing and edge classification, enabling particle instance
segmentation and identification.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch

from spine.cluster.formation import form_clusters_batch
from spine.cluster.label import (
    get_cluster_label_batch,
    get_cluster_primary_label_batch,
)
from spine.constants import LOWES_SHP, TRACK_SHP
from spine.constants.factory import enum_factory
from spine.data import ClusterLabelBatch, EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.common.dbscan import DBSCAN
from spine.model.common.factories import final_factory
from spine.model.common.quality import ClusterOverlapCache
from spine.model.grappa.evaluation import (
    node_assignment_batch,
    node_assignment_score_batch,
)

from ..registry import ModelSpec
from .factories import (
    FeatureEncoder,
    GNNModel,
    edge_encoder_factory,
    edge_loss_factory,
    global_encoder_factory,
    global_loss_factory,
    gnn_model_factory,
    graph_factory,
    node_encoder_factory,
    node_loss_factory,
)
from .graph.base import GraphBase

__all__ = ["GrapPA", "GrapPALoss"]


class GrapPA(torch.nn.Module):
    """Graph Particle Aggregator (GrapPA) model.

    This class mostly acts as a wrapper that will hand the graph data
    to the underlying graph neural network (GNN).

    When trained standalone, this model must be provided with a cluster
    label tensor, allowing it to build a set of input clusters based on the
    label boundaries of the clusters and their semantic types.

    Typical configuration can look like this:

    .. code-block:: yaml

        model:
          name: grappa
          modules:
            grappa:
              nodes:
                <dictionary of arguments to specify the input type>
              graph:
                name: <name of the input graph type>
                <dictionary of arguments to specify the graph>
              node_encoder:
                name: <name of the type of node encoder>
                <dictionary of arguments to specify the node encoder>
              edge_encoder:
                name: <name of the type of edge encoder>
                <dictionary of arguments to specify the edge encoder>
              global_encoder:
                name: <name of the type of global encoder>
                <dictionary of arguments to specify the global encoder>
              gnn_model:
                name: <name of the type of backbone GNN feature extractor>
                <dictionary of arguments to specify the GNN>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.

    See Also
    --------
    :class:`GrapPALoss`
    """

    def __init__(self, grappa: dict[str, Any]) -> None:
        """Initialize the GrapPA model.

        Parameters
        ----------
        grappa : dict
            Model configuration
        """
        # Initialize the parent class
        super().__init__()

        # Declare configuration-owned attributes up front. Besides making the
        # runtime state explicit, this keeps static analyzers from treating
        # assignments in the configuration helpers as ad hoc attributes.
        self.out_types = ("node", "edge", "global")
        self.gnn: GNNModel
        self.node_source: str
        self.node_type: list[int]
        self.node_min_size: int
        self.make_groups: bool
        self.grouping_method: str
        self.grouping_through_track: bool
        self.graph_constructor: GraphBase | None = None
        self.node_encoder: FeatureEncoder | None = None
        self.edge_encoder: FeatureEncoder | None = None
        self.global_encoder: FeatureEncoder | None = None
        self.dbscan: DBSCAN | None = None
        self.return_features = False
        self.node_pred_keys: list[str] = []
        self.edge_pred_keys: list[str] = []
        self.global_pred_keys: list[str] = []

        # Process the model configuration
        self.process_model_config(**grappa)

    def process_model_config(
        self,
        gnn_model: dict[str, Any],
        nodes: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
        node_encoder: dict[str, Any] | None = None,
        edge_encoder: dict[str, Any] | None = None,
        global_encoder: dict[str, Any] | None = None,
        dbscan: dict[str, Any] | None = None,
        return_features: bool = False,
    ) -> None:
        """Process the top-level configuration block.

        This dispatches each block to its own configuration processor.

        Parameters
        ----------
        gnn_model : dict
            Underlying graph neural network configuration
        nodes : dict, optional
            Input node configuration
        graph : dict, optional
            Input graph configuration
        node_encoder : dict, optional
            Node encoder configuration
        edge_encoder : dict, optional
            Edge encoder configuration
        global_encoder : dict, optional
            Global encoder configuration
        dbscan : dict, optional
            DBSCAN fragmentation configuration
        return_features : bool, default False
            If `True`, the model will return the node/edge/global features
        """
        if nodes is None and (graph is not None or dbscan is not None):
            raise ValueError(
                "Must provide a `nodes` configuration when using GrapPA to "
                "build clusters/graphs on the fly."
            )

        # Construct the underlying graph neural network
        self.process_gnn_config(**gnn_model)

        # Process the node configuration
        self.process_node_config(**(nodes or {}))

        # Process the graph configuration
        if graph is not None:
            self.graph_constructor = graph_factory(graph, self.node_type)

        # Process the encoder configurations
        if node_encoder is not None:
            self.node_encoder = node_encoder_factory(node_encoder)

        # Initialize edge encoder
        if edge_encoder is not None:
            self.edge_encoder = edge_encoder_factory(edge_encoder)

        # Initialize the global encoder
        if global_encoder is not None:
            self.global_encoder = global_encoder_factory(global_encoder)

        # Fail at construction time when a configured encoder cannot feed the
        # corresponding GNN input, rather than surfacing a matrix-shape error
        # in the first forward pass.
        self._validate_encoder_size("node", self.node_encoder)
        self._validate_encoder_size("edge", self.edge_encoder)
        self._validate_encoder_size("global", self.global_encoder)

        # Process the dbscan fragmenter configuration, if provided
        if dbscan is not None:
            self.process_dbscan_config(**dbscan)

        if self.make_groups and not self.edge_pred_keys:
            raise ValueError("Building groups requires an edge prediction head.")

        # Store whether to return the features
        self.return_features = return_features

    def process_node_config(
        self,
        source: str = "cluster",
        shapes: Sequence[int | str] | None = None,
        min_size: int = -1,
        make_groups: bool = False,
        grouping_method: str = "score",
        grouping_through_track: bool = False,
    ) -> None:
        """Process the node parameters of the model.

        Parameters
        ----------
        source : str, default 'cluster'
            Column name in the label tensor which contains the input cluster IDs
        shapes : sequence of int or str, optional
            Semantic types to include as nodes. If omitted, include every
            non-low-energy class.
        min_size : int, default -1
            Minimum number of voxels in a cluster to be included in the input
        make_groups : bool, default False
            Use edge predictions to build node groups
        grouping_method : str, default 'score'
            Algorithm used to build a node partition
        grouping_through_track : bool, default False
            If `True`, shower objects can only be connected to one track object
        """
        # Parse the node source
        source_aliases = {
            "clust": "cluster",
            "part": "particle",
            "inter": "interaction",
        }
        self.node_source = source_aliases.get(source, source)
        if self.node_source == "voxel" and min_size not in (-1, 1):
            raise ValueError("Voxel nodes are singletons; `min_size` must be -1 or 1.")

        # Interpret node type as list of shapes to cluster
        if shapes is None:
            self.node_type = list(range(LOWES_SHP))
        else:
            if isinstance(shapes, (str, bytes)) or np.isscalar(shapes):
                raise ValueError("Semantic classes should be provided as a list.")
            self.node_type = [
                enum_factory("shape", shape) if isinstance(shape, str) else int(shape)
                for shape in shapes
            ]

        if grouping_method not in ("score", "threshold"):
            raise ValueError(
                f"Grouping method not recognized: {grouping_method}. "
                "Must be one of ('score', 'threshold')."
            )
        if grouping_through_track and grouping_method != "score":
            raise ValueError(
                "Track-restricted grouping is only supported by the "
                "`score` grouping method."
            )

        # Store the node parameters
        self.node_min_size = min_size
        self.make_groups = make_groups
        self.grouping_method = grouping_method
        self.grouping_through_track = grouping_through_track

    def _validate_encoder_size(
        self,
        prefix: str,
        encoder: FeatureEncoder | None,
    ) -> None:
        """Check that a configured encoder matches its GNN input width.

        Parameters
        ----------
        prefix : str
            Feature family, one of ``"node"``, ``"edge"`` or ``"global"``.
        encoder : FeatureEncoder, optional
            Configured encoder for that feature family.

        Raises
        ------
        ValueError
            If the encoder output width differs from the GNN input width.
        """
        if encoder is None:
            return

        expected_size = getattr(self.gnn, f"{prefix}_feats")
        if encoder.feature_size != expected_size:
            raise ValueError(
                f"The {prefix} encoder produces {encoder.feature_size} features, "
                f"but the GNN expects {expected_size}."
            )

    def process_gnn_config(
        self,
        node_pred: int | dict[str, Any] | None = None,
        edge_pred: int | dict[str, Any] | None = None,
        global_pred: int | dict[str, Any] | None = None,
        **gnn_model: Any,
    ) -> None:
        """Process the GNN backbone structure and the output layers.

        Parameters
        ----------
        node_pred : Union[int, dict], optional
            Number of node predictions. If there are multiple node predictions,
            provide a (key, value) pair for each type of prediction
        edge_pred : Union[int, dict], optional
            Number of edge predictions. If there are multiple edge predictions,
            provide a (key, value) pair for each type of prediction
        global_pred : Union[int, dict], optional
            Number of global predictions. If there are multiple global predictions,
            provide a (key, value) pair for each type of prediction
        **gnn_model : dict
            Parameters used to initialize the GNN backbone.
        """
        # Initialize the GNN backbone
        self.gnn = gnn_model_factory(
            gnn_model,
            node_pred is not None,
            edge_pred is not None,
            global_pred is not None,
        )

        self.process_final_config(node_pred, "node")
        self.process_final_config(edge_pred, "edge")
        self.process_final_config(global_pred, "global")

    def process_final_config(
        self, final: int | dict[str, Any] | None, prefix: str
    ) -> None:
        """Process a final layer configuration.

        Parameters
        ----------
        final : Union[int, dict]
            Final layer configuration
        prefix : str
            Name of the final layer
        """
        # If the final layer is not specified, nothing to do here
        if final is None:
            setattr(self, f"{prefix}_pred_keys", [])
            return

        # If the final layer is specified as a number, use linear layer
        if isinstance(final, int):
            final = {"name": "linear", "out_channels": final}

        # Process the configuration dictionary otherwise
        out_keys = []
        in_channels = getattr(self.gnn, f"{prefix}_feature_size")
        if "name" in final:
            # Initialize a single final layer (single prediction of this type)
            out_key = f"{prefix}_pred"
            out_keys.append(out_key)
            setattr(self, out_key, final_factory(in_channels, **final))

        else:
            # Otherwise, initialize one final layer per prediction type
            for key, cfg in final.items():
                # If the final layer is specified as a number, use linear layer
                out_key = f"{prefix}_{key}_pred"
                out_keys.append(out_key)
                if isinstance(cfg, int):
                    cfg = {"name": "linear", "out_channels": cfg}
                setattr(self, out_key, final_factory(in_channels, **cfg))

        setattr(self, f"{prefix}_pred_keys", out_keys)

    def process_dbscan_config(
        self,
        shapes: Sequence[int | str] | None = None,
        min_size: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Process the DBSCAN fragmenter configuration.

        Parameters
        ----------
        shapes : sequence of int or str, optional
            This should not be specified (fetched from the node configuration)
        min_size : int or sequence of int, optional
            This should not be specified (fetched from the node configuration)
        **kwargs : dict, optional
            Rest of the DBSCAN configuration
        """
        # Make sure the basic parameters are not specified twice
        if shapes is not None or min_size is not None:
            raise ValueError(
                "Do not specify 'shapes' or 'min_size' in the "
                "`dbscan` block, it is shared with the `node` block"
            )

        # Initialize DBSCAN fragmenter
        self.dbscan = DBSCAN(
            shapes=self.node_type, min_size=self.node_min_size, **kwargs
        )

    def forward(
        self,
        data: ClusterLabelBatch | TensorBatch,
        coord_label: TensorBatch | None = None,
        clusts: IndexBatch | None = None,
        edge_index: EdgeIndexBatch | None = None,
        node_features: TensorBatch | None = None,
        edge_features: TensorBatch | None = None,
        global_features: TensorBatch | None = None,
        shapes: TensorBatch | None = None,
        groups: TensorBatch | None = None,
        points: TensorBatch | None = None,
        extra: TensorBatch | None = None,
    ) -> dict[str, Any]:
        """Prepare particle clusters and feed them to the GNN model.

        Parameters
        ----------
        data : TensorBatch
            Tensor of voxel/value pairs with shape `(N, 1 + D + N_f)`, where
            `N` is the total number of voxels, the leading column stores the
            batch ID, `D` is the image dimensionality and `N_f` is the number
            of features. When `clusts` is not provided, the features must also
            contain the labels needed to build clusters on the fly.
        coord_label : TensorBatch, optional
            (P, 1 + 2*D + 2) Tensor of label points (start/end/time/shape)
        clusts : IndexBatch, optional
            (C) List of indexes corresponding to each cluster
        edge_index : EdgeIndexBatch, optional
            (E, 2) Incidence matrix. If not provided, it will be built based on
            the cluster indexes and the graph configuration
        node_features : TensorBatch, optional
            (C, N_c,f) Node features. If omitted, build them with the configured
            node encoder.
        edge_features : TensorBatch, optional
            (C, N_e,f) Edge features. If not provided, they will be built based on
            the cluster indexes and the edge encoder configuration
        global_features : TensorBatch, optional
            (C, N_g,f) Global features. If not provided, they will be built based on
            the cluster indexes and the global encoder configuration
        shapes : TensorBatch, optional
            (C) List of cluster semantic class used to define the max length
        groups : TensorBatch, optional
            (C) List of node groups, one per cluster. If specified, removes
            connections between nodes that belong to different groups.
        points : TensorBatch, optional
            (C, 3/6) Tensor of start (and end) points
        extra : TensorBatch, optional
            (C, N_f) Batch of features to append to the existing node features

        Returns
        -------
        clusts : IndexBatch
            (C, N_c, N_{c,i}) Cluster indexes
        edge_index : EdgeIndexBatch
            (E, 2) Incidence matrix
        node_features : TensorBatch
            (C, N_c,f) Node features
        edge_features : TensorBatch
            (C, N_e,f) Node features
        global_features : TensorBatch
            (C, N_g,f) Global features
        node_pred : TensorBatch
            (C, N_n) Node predictions (logits)
        edge_pred : TensorBatch
            (C, N_e) Edge predictions (logits)
        global_pred : TensorBatch
            (C, N_e) Global predictions (logits)
        """
        result: dict[str, Any] = {}
        voxel_data = (
            data.to_tensor_batch() if isinstance(data, ClusterLabelBatch) else data
        )

        # Encode the node boundaries as clusters if they are not provided directly
        if clusts is None:
            if not isinstance(data, ClusterLabelBatch) and self.node_source != "voxel":
                raise TypeError(
                    "Building clusters requires structured cluster labels. "
                    "Tensor input must be paired with explicit `clusts`."
                )
            clusts = self._make_clusters(data, coord_label=coord_label)
        result["clusts"] = clusts

        # If needed, infer per-cluster shapes once and reuse them downstream
        shapes = self._get_shapes(data, clusts, shapes)

        # Build the graph if it is not provided directly
        closest_index = None
        if edge_index is None:
            if self.graph_constructor is None:
                raise ValueError(
                    "Must provide edge_index or graph configuration to build it."
                )
            edge_index, closest_index = self._make_edge_index(
                data, clusts, shapes=shapes, groups=groups
            )
        result["edge_index"] = edge_index

        # Fetch the node features
        if node_features is None:
            if self.node_encoder is None:
                raise ValueError(
                    "Must provide node_features or node encoder configuration to build them."
                )
            encoded_nodes = self.node_encoder(
                data, clusts, coord_label=coord_label, points=points, extra=extra
            )

            if isinstance(encoded_nodes, tuple):
                # If the output of the node encoder is a tuple, separate points
                if len(encoded_nodes) != 2:
                    raise TypeError(
                        "Node encoders must return TensorBatch or a pair of "
                        "TensorBatch objects."
                    )
                node_features, encoded_points = cast(
                    tuple[TensorBatch, TensorBatch], encoded_nodes
                )
                point_tensor = encoded_points.torch_tensor()
                if point_tensor.shape[1] != 6:
                    raise ValueError(
                        "Endpoint-producing node encoders must return six "
                        "coordinates per cluster."
                    )
                start_points, end_points = point_tensor.split(3, dim=1)

                result["start_points"] = TensorBatch(
                    start_points,
                    encoded_points.counts,
                    coord_cols=np.array([0, 1, 2]),
                )
                result["end_points"] = TensorBatch(
                    end_points,
                    encoded_points.counts,
                    coord_cols=np.array([0, 1, 2]),
                )
                points = encoded_points
            else:
                node_features = cast(TensorBatch, encoded_nodes)

        if self.return_features:
            result["node_features"] = node_features

        # Fetch the edge features
        if edge_features is None and self.edge_encoder is not None:
            edge_features = cast(
                TensorBatch,
                self.edge_encoder(
                    voxel_data, clusts, edge_index, closest_index=closest_index
                ),
            )

        if self.return_features and edge_features is not None:
            result["edge_features"] = edge_features

        # Fetch the global_features
        if global_features is None and self.global_encoder is not None:
            global_features = cast(TensorBatch, self.global_encoder(voxel_data, clusts))

        if global_features is not None and self.return_features:
            result["global_features"] = global_features

        # Bring graph indexes to the feature device. Graph construction remains
        # CPU-based, while the message-passing network operates on Torch tensors.
        data_tensor = voxel_data.torch_tensor()
        index = torch.as_tensor(
            edge_index.index,
            dtype=torch.long,
            device=data_tensor.device,
        )
        node_batch_ids = torch.as_tensor(
            clusts.batch_ids,
            dtype=torch.long,
            device=data_tensor.device,
        )

        # Pass through the model, update results
        out = self.gnn(
            node_features,
            index,
            edge_features,
            global_features,
            node_batch_ids,
        )

        # Loop over the necessary node/edge/global predictions, store output
        for output_type in self.out_types:
            prediction_keys = getattr(self, f"{output_type}_pred_keys")
            feature_key = f"{output_type}_features"
            if prediction_keys and feature_key not in out:
                raise RuntimeError(
                    f"The GNN did not produce `{feature_key}` for its configured "
                    "prediction head."
                )
            for key in prediction_keys:
                result[key] = cast(TensorBatch, getattr(self, key)(out[feature_key]))

        # If requested, build node groups from edge predictions
        if self.make_groups:
            self._make_groups(result, clusts, edge_index, shapes=shapes)

        return result

    def _make_clusters(
        self,
        data: ClusterLabelBatch | TensorBatch,
        coord_label: TensorBatch | None = None,
    ) -> IndexBatch:
        """Build node clusters from labels or configured DBSCAN fragmentation.

        Parameters
        ----------
        data : ClusterLabelBatch or TensorBatch
            Structured labels used to build clusters, or a plain tensor batch
            when each selected voxel is represented as an individual node.
        coord_label : TensorBatch, optional
            (P, 1 + 2*D + 2) Tensor of label points

        Returns
        -------
        clusts : IndexBatch
            (C, N_c, N_{c,i}) Cluster indexes

        Raises
        ------
        TypeError
            If cluster or DBSCAN fragmentation is requested with a plain
            ``TensorBatch``, which does not carry semantic or instance labels.
        """
        if self.node_source == "voxel":
            # Represent each selected voxel as a singleton graph node. With
            # structured labels, retain the usual semantic-shape selection;
            # plain tensors intentionally include every input row.
            tensor_data = (
                data.to_tensor_batch() if isinstance(data, ClusterLabelBatch) else data
            )
            selected = np.arange(len(tensor_data.data), dtype=np.int64)
            if isinstance(data, ClusterLabelBatch):
                shape_values = data.shapes.to_numpy().data
                selected = selected[np.isin(shape_values, self.node_type)]

            clusters = [np.asarray([index], dtype=np.int64) for index in selected]
            batch_ids = tensor_data.batch_ids
            if not isinstance(batch_ids, np.ndarray):
                batch_ids = batch_ids.detach().cpu().numpy()
            batch_ids = batch_ids[selected]
            counts = np.bincount(
                batch_ids.astype(np.int64),
                minlength=tensor_data.batch_size,
            )
            single_counts = np.ones(len(clusters), dtype=np.int64)
            return IndexBatch(
                clusters,
                spans=tensor_data.counts,
                counts=counts,
                single_counts=single_counts,
                default=np.empty(0, dtype=np.int64),
            )

        # All non-voxel fragmentation paths require the named semantic and
        # instance fields provided by structured cluster labels. Besides making
        # that runtime contract explicit, this narrows the type below.
        if not isinstance(data, ClusterLabelBatch):
            raise TypeError("Label clustering requires structured labels.")

        if self.dbscan is not None:
            # Use the DBSCAN fragmenter to build the clusters
            seg_label = data.shapes
            clusts, _ = self.dbscan(data.to_tensor_batch(), seg_label, coord_label)
        else:
            # Use the label tensor to build the clusters
            clusts = form_clusters_batch(
                data.to_numpy(),
                self.node_min_size,
                self.node_source,
                shapes=self.node_type,
            )

        return clusts

    def _get_shapes(
        self,
        data: ClusterLabelBatch | TensorBatch,
        clusts: IndexBatch,
        shapes: TensorBatch | None = None,
    ) -> TensorBatch | None:
        """Return per-cluster semantic labels if the graph logic needs them.

        Parameters
        ----------
        data : TensorBatch
            Tensor of voxel/value pairs with shape `(N, 1 + D + N_f)`.
        clusts : IndexBatch
            (C) List of indexes corresponding to each cluster
        shapes : TensorBatch, optional
            (C) Explicit semantic label per cluster

        Returns
        -------
        TensorBatch or None
            Cluster semantic labels, or `None` if they are not needed and were
            not provided.
        """
        if shapes is not None:
            return shapes

        class_dependent_edges = self.graph_constructor is not None and isinstance(
            self.graph_constructor.max_length, np.ndarray
        )
        track_restricted_grouping = self.make_groups and self.grouping_through_track
        if not class_dependent_edges and not track_restricted_grouping:
            return None

        if not isinstance(data, ClusterLabelBatch):
            raise TypeError(
                "Deriving semantic node labels requires structured cluster labels "
                "or an explicit `shapes` tensor."
            )
        data_np = data.to_numpy()
        if self.node_source == "group":
            shapes = get_cluster_primary_label_batch(data_np, clusts, "shape")
        else:
            shapes = get_cluster_label_batch(data_np, clusts, "shape")

        shape_values = shapes.numpy_tensor().astype(np.int64, copy=False)
        return TensorBatch(shape_values, shapes.counts)

    def _make_edge_index(
        self,
        data: ClusterLabelBatch | TensorBatch,
        clusts: IndexBatch,
        shapes: TensorBatch | None = None,
        groups: TensorBatch | None = None,
    ) -> tuple[EdgeIndexBatch, np.ndarray | None]:
        """Make the edge index based on the cluster indexes and the graph configuration.

        Parameters
        ----------
        data : TensorBatch
            Tensor of voxel/value pairs with shape `(N, 1 + D + N_f)`, where
            `N` is the total number of voxels, the leading column stores the
            batch ID, `D` is the image dimensionality and `N_f` is the number
            of features. The features must also contain the labels needed to
            build clusters on the fly.
        clusts : IndexBatch
            (C) List of indexes corresponding to each cluster
        shapes : TensorBatch, optional
            (C) List of cluster semantic class used to define the max length
        groups : TensorBatch, optional
            (C) List of node groups, one per cluster. If specified, removes
            connections between nodes that belong to different groups.

        Returns
        -------
        edge_index : EdgeIndexBatch
            (E, 2) Incidence matrix
        closest_index : np.ndarray
            (C, C) Closest voxel-pair index for each pair of clusters.
        """
        # Check that the graph constructor is defined
        if self.graph_constructor is None:
            raise ValueError(
                "Must provide graph configuration to build edge index from clusters."
            )

        # Bring data to numpy for the graph construction
        data_np = data.to_numpy()
        shapes_np = None if shapes is None else shapes.to_numpy()
        groups_np = None if groups is None else groups.to_numpy()

        # Initialize the input graph
        edge_index, _, closest_index = self.graph_constructor(
            data_np, clusts, shapes_np, groups_np
        )

        return edge_index, closest_index

    def _make_groups(
        self,
        result: dict[str, Any],
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        shapes: TensorBatch | None = None,
    ) -> None:
        """Make node groups based on edge predictions.

        Parameters
        ----------
        result : dict
            Model outputs containing the configured edge-prediction heads.
        clusts : IndexBatch
            (C) List of indexes corresponding to each cluster
        edge_index : EdgeIndexBatch
            (E, 2) Incidence matrix
        shapes : TensorBatch, optional
            (C) List of cluster semantic class used to restrict track association
        """
        # Fetch the list of edge prediction keys
        edge_pred_keys = [key for key in self.edge_pred_keys if key in result]
        if not edge_pred_keys:
            raise ValueError(
                "Must provide edge predictions to build node groups. "
                "Edge predictions should be stored under keys with the format "
                "`edge{key}_pred`, where `key` is the name of the prediction head."
            )

        # Loop over the edge predictions, build node groups based on each of them
        edge_index_np = edge_index.to_numpy()
        clusts_np = clusts.to_numpy()
        for key in edge_pred_keys:
            edge_pred = result[key].to_numpy()
            prefix = "group" + key.replace("edge", "").replace("_pred", "")
            if self.grouping_method == "threshold":
                result[f"{prefix}_pred"] = node_assignment_batch(
                    edge_index_np, edge_pred, clusts_np
                )

            elif self.grouping_method == "score":
                if not self.grouping_through_track:
                    result[f"{prefix}_pred"] = node_assignment_score_batch(
                        edge_index_np, edge_pred, clusts_np
                    )
                else:
                    if shapes is None:
                        raise ValueError(
                            "Must provide shapes to restrict track association."
                        )
                    shapes_np = shapes.to_numpy()
                    track_node = TensorBatch(
                        shapes_np.numpy_tensor() == TRACK_SHP,
                        counts=shapes_np.counts,
                    )
                    result[f"{prefix}_pred"] = node_assignment_score_batch(
                        edge_index_np,
                        edge_pred,
                        clusts_np,
                        track_node,
                    )

            else:
                raise RuntimeError(
                    f"Unexpected grouping method: {self.grouping_method}."
                )


class GrapPALoss(torch.nn.Module):
    """Takes the output of the GrapPA and computes the total loss.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: <name of the node loss>
                <dictionary of arguments to pass to the loss>
              edge_loss:
                name: <name of the edge loss>
                <dictionary of arguments to pass to the loss>
              global_loss:
                name: <name of the global loss>
                <dictionary of arguments to pass to the loss>

    Each specific loss block can also contain multiple losses by
    providing a name key in a loss block nested below it. Each loss name of a
    specific type should be provided with a corresponding output from GrapPA.

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    def __init__(
        self,
        grappa_loss: dict[str, Any],
        grappa: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GrapPA loss function.

        Parameters
        ----------
        grappa_loss : dict
            Loss configuration
        grappa : dict, optional
            Model configuration supplied through the manager's shared
            model/loss contract. Individual objectives currently infer their
            required heads from the model output.
        """
        # Initialize the parent class
        super().__init__()

        self.out_types = ("node", "edge", "global")
        self.node_loss_keys: list[str] = []
        self.edge_loss_keys: list[str] = []
        self.global_loss_keys: list[str] = []

        # Process the loss configuration
        self.process_loss_config(**grappa_loss)

    def process_loss_config(
        self,
        node_loss: dict[str, Any] | None = None,
        edge_loss: dict[str, Any] | None = None,
        global_loss: dict[str, Any] | None = None,
    ) -> None:
        """Process the loss configuration.

        Parameters
        ----------
        node_loss : Union[dict, Dict[dict]], optional
            Node loss configuration
        edge_loss : Union[dict, Dict[dict]], optional
            Edge loss configuration
        global_loss : Union[dict, Dict[dict]], optional
            Global loss configuration
        """
        # Check that there is at least one loss to apply
        if node_loss is None and edge_loss is None and global_loss is None:
            raise ValueError(
                "Must provide at least one of `node_loss`, `edge_loss` or "
                "`global_loss` to the GrapPA loss function."
            )

        # Initialize the node/edge/global losses
        self.process_single_loss_config("node", node_loss, node_loss_factory)
        self.process_single_loss_config("edge", edge_loss, edge_loss_factory)
        self.process_single_loss_config("global", global_loss, global_loss_factory)

    def process_single_loss_config(
        self,
        prefix: str,
        loss: dict[str, Any] | None,
        constructor: Callable[[dict[str, Any]], torch.nn.Module],
    ) -> None:
        """Process a loss configuration.

        Parameters
        ----------
        prefix : str
            Name of the output type to apply the loss to
        loss : dict, optional
            Loss configuration
        constructor : callable
            Loss constructor function
        """
        # If the loss is not specified, nothing to do here
        if loss is None:
            setattr(self, f"{prefix}_loss_keys", [])
            return

        # Process the configuration dictionary otherwise
        loss_keys = []
        if "name" in loss:
            # Initialize a single loss
            loss_key = f"{prefix}_loss"
            loss_keys.append(loss_key)
            setattr(self, loss_key, constructor(loss))

        else:
            # Otherwise, initialize one loss per prediction type
            for key, cfg in loss.items():
                loss_key = f"{prefix}_{key}_loss"
                loss_keys.append(loss_key)
                setattr(self, loss_key, constructor(cfg))

        setattr(self, f"{prefix}_loss_keys", loss_keys)

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        coord_label: TensorBatch | None = None,
        graph_label: EdgeIndexBatch | None = None,
        iteration: int | None = None,
        **output: Any,
    ) -> dict[str, Any]:
        """Apply the node/edge/global losses to the logits from GrapPA.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is is the number of cluster labels
        coord_label : TensorBatch, optional
            (P, 1 + D + 8) Tensor of start/end point labels for each
            true particle in the image
        graph_label : EdgeIndexBatch, optional
            (2, E) Tensor of edges that correspond to physical
            connections between true particle in the image
        iteration : int, optional
            Iteration index
        **output : dict
            Output of the GrapPA model
        """
        # Loop and apply the losses
        result: dict[str, Any] = {}

        # Objectives often match the same clusters to the same truth field.
        # Cache those geometrical overlaps for the duration of this forward.
        overlap_cache: ClusterOverlapCache = {}
        num_losses = 0
        total_loss: torch.Tensor | None = None
        total_accuracy = 0.0
        for t in self.out_types:
            loss_keys = getattr(self, f"{t}_loss_keys")
            for key in loss_keys:
                # Route each configured loss to the prediction head with the
                # matching name. Single heads naturally map, e.g.
                # ``edge_loss`` -> ``edge_pred``.
                prediction_key = key.removesuffix("_loss") + "_pred"
                if prediction_key not in output:
                    raise KeyError(
                        f"Loss `{key}` requires model output `{prediction_key}`."
                    )
                extra = {}
                generic_prediction_key = f"{t}_pred"
                if prediction_key != generic_prediction_key:
                    extra[generic_prediction_key] = output[prediction_key]

                # Compute the loss
                out = getattr(self, key)(
                    clust_label=clust_label,
                    coord_label=coord_label,
                    true_edge_index=graph_label,
                    iteration=iteration,
                    overlap_cache=overlap_cache,
                    **output,
                    **extra,
                )

                # Increment the loss and accuracy
                loss_value = out["loss"]
                if not isinstance(loss_value, torch.Tensor):
                    raise TypeError(f"Loss `{key}` did not return a torch.Tensor.")
                total_loss = (
                    loss_value if total_loss is None else total_loss + loss_value
                )
                total_accuracy += float(out["accuracy"])
                num_losses += 1

                # Update the result dictionary
                prefix = "_".join(key.split("_")[:-1])
                for k, v in out.items():
                    result[f"{prefix}_{k}"] = v

        # Append the total loss and total accuracy
        assert total_loss is not None
        assert num_losses > 0
        result["loss"] = total_loss / num_losses
        result["accuracy"] = total_accuracy / num_losses

        return result


MODEL_SPEC = ModelSpec("grappa", GrapPA, GrapPALoss)
