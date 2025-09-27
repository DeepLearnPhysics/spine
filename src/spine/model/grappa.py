import random
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.enums import enum_factory
from spine.utils.globals import (
    BATCH_COL,
    CLUST_COL,
    COORD_COLS,
    GROUP_COL,
    LOWES_SHP,
    SHAPE_COL,
    TRACK_SHP,
)
from spine.utils.gnn.cluster import (
    form_clusters_batch,
    get_cluster_label_batch,
    get_cluster_primary_label_batch,
)
from spine.utils.gnn.evaluation import (
    node_assignment_batch,
    node_assignment_score_batch,
)

from .layer.common.dbscan import DBSCAN
from .layer.factories import final_factory
from .layer.gnn.factories import *

__all__ = ["GrapPA", "GrapPALoss"]


class GrapPA(torch.nn.Module):
    """Graph Particle Aggregator (GrapPA) model.

    This class mostly acts as a wrapper that will hand the graph data
    to the underlying graph neural network (GNN).

    When trained standalone, this model must be provided with a cluster
    label tensor, allowing it to build a set of intput clusters based on the
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

    # TODO: update
    MODULES = [
        ("grappa", ["base", "dbscan", "node_encoder", "edge_encoder", "gnn_model"]),
        "grappa_loss",
    ]

    def __init__(self, grappa, grappa_loss=None):
        """Initialize the GrapPA model.

        Parameters
        ----------
        grappa : dict
            Model configuration
        grappa_loss : dict, optional
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the model configuration
        self.process_model_config(**grappa)

    def process_model_config(
        self,
        nodes,
        graph,
        gnn_model,
        node_encoder,
        edge_encoder=None,
        global_encoder=None,
        dbscan=None,
    ):
        """Process the top-level configuration block.

        This dispatches each block to its own configuration processor.

        Parameters
        ----------
        nodes : dict
            Input node configuration
        graph : dict
            Input graph configuration
        gnn_model : dict
            Underlying graph neural network configuration
        node_encoder : dict
            Node encoder configuration
        edge_encoder : dict
            Edge encoder configuration
        global_encoder : dict, optional
            Global encoder configuration
        dbscan : dict, optional
            DBSCAN fragmentation configuration
        """
        # Store the output types of GNNs
        self.out_types = ["node", "edge", "global"]

        # Process the node configuration
        self.process_node_config(**nodes)

        # Construct the underlying graph neural network
        self.process_gnn_config(**gnn_model)

        # Process the graph configuration
        self.graph_constructor = graph_factory(graph, self.node_type)

        # Process the encoder configurations
        self.node_encoder = node_encoder_factory(node_encoder)

        # Initialize edge encoder
        self.edge_encoder = None
        if edge_encoder is not None:
            self.edge_encoder = edge_encoder_factory(edge_encoder)

        # Initialize the global encoder
        self.global_encoder = None
        if global_encoder is not None:
            self.global_encoder = global_encoder_factory(global_encoder)

        # Process the dbscan fragmenter configuration, if provided
        self.dbscan = None
        if dbscan is not None:
            self.process_dbscan_config(dbscan)

    def process_node_config(
        self,
        source="cluster",
        shapes=None,
        min_size=-1,
        make_groups=False,
        grouping_method="score",
        grouping_through_track=False,
    ):
        """Process the node parameters of the model.

        Parameters
        ----------
        source : str, default 'cluster'
            Column name in the label tensor which contains the input cluster IDs
        shapes : int, optional
            Type of nodes to include in the input. If not specified, include
            all types
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
        self.node_source = enum_factory("cluster", source)

        # Interpret node type as list of shapes to cluster
        if shapes is None:
            self.node_type = list(np.arange(LOWES_SHP))
        else:
            assert not np.isscalar(
                shapes
            ), "Semantic classes should be provided as a list."
            self.node_type = enum_factory("shape", shapes)

        # Store the node parameters
        self.node_min_size = min_size
        self.make_groups = make_groups
        self.grouping_method = grouping_method
        self.grouping_through_track = grouping_through_track

    def process_gnn_config(
        self, node_pred=None, edge_pred=None, global_pred=None, **gnn_model
    ):
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
            Number of edge predictions. If there are multiple edge predictions,
            provide a (key, value) pair for each type of prediction
        **gnn_model, dict
            Paramters to initialize the GNN backbone
        """
        # Initialize the GNN backbone
        self.gnn = gnn_model_factory(
            gnn_model,
            node_pred is not None,
            edge_pred is not None,
            global_pred is not None,
        )

        # Initialize output layers based on the configuration
        self.process_final_config(node_pred, "node")
        self.process_final_config(edge_pred, "edge")
        self.process_final_config(global_pred, "global")

    def process_final_config(self, final, prefix):
        """Process a final layer configuration.

        Parameters
        ----------
        final : Union[int, dict]
            Final layer configuration
        prefix : dict
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

    def process_dbscan_config(shapes=None, min_size=None, **kwargs):
        """Process the DBSCAN fragmenter configuration.

        Parameters
        ----------
        shapes : Union[int, list], optional
            This should not be specified (fetched from the node configuration)
        min_size : Union[int, list], optional
            This should not be specified (fetched from the node configuration)
        **kwargs : dict, optional
            Rest of the DBSCAN configuration
        """
        # Make sure the basic parameters are not specified twice
        assert shapes is not None and min_size is not None, (
            "Do not specify 'shapes' or 'min_size' in the "
            "`dbscan` block, it is shared with the `node` block"
        )

        # Initialize DBSCAN fragmenter
        self.dbscan = DBSCAN(shapes=self.node_type, min_size=self.min_size, **kwargs)

    def forward(
        self,
        data,
        coord_label=None,
        clusts=None,
        shapes=None,
        groups=None,
        points=None,
        extra=None,
    ):
        """Prepares particle clusters and feed them to the GNN model.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is 1 (charge/energy) if the clusters (`clusts`) are provided,
              or it needs to contain cluster labels to build them on the fly
        coord_label : TensorBatch, optional
            (P, 1 + 2*D + 2) Tensor of label points (start/end/time/shape)
        clusts : IndexBatch, optional
            (C) List of indexes corresponding to each cluster
        shapes : TensorBatch, optional
            (C) List of cluster semantic class used to define the max length
        groups : TensorBatch, optional
            (C) List of node groups, one per cluster. If specified, will
                remove connections between nodes of a separate group.
        points : TensorBatch, optional
            (C, 3/6) Tensor of start (and end) points
        extra : TensorBatch, optional
            (C, N_f) Batch of features to append to the existing node features

        Returns
        -------
        clusts : IndexBatch
            (C, N_c, N_{c,i}) Cluster indexes
        edge_index : TensorBatch
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
        # Cast the labels to numpy for the functions run on CPU
        result = {}
        data_np = data.to_numpy()

        # If not provided, form the clusters: a list of list of voxel indices,
        # one list per cluster matching the list of requested class
        if clusts is None:
            if self.dbscan is not None:
                # Use the DBSCAN fragmenter to build the clusters
                seg_label = TensorBatch(data.tensor[:, SHAPE_COL], data.counts)
                clusts, _ = self.dbscan(data, seg_label, coord_label)
            else:
                # Use the label tensor to build the clusters
                clusts = form_clusters_batch(
                    data_np, self.node_min_size, self.node_source, shapes=self.node_type
                )

        result["clusts"] = clusts

        # If the graph edge length cut is class-specific, get the class labels
        if shapes is None and hasattr(self.graph_constructor.max_length, "__len__"):
            if self.node_source == GROUP_COL:
                # For groups, use primary shape to handle Michel/Delta properly
                shapes = get_cluster_primary_label_batch(data_np, clusts, SHAPE_COL)
            else:
                # Just use the shape of the cluster itself otherwise
                shapes = get_cluster_label_batch(data_np, clusts, SHAPE_COL)

            shapes.data = shapes.data.astype(np.int64)

        # Initialize the input graph
        edge_index, dist_mat, closest_index = self.graph_constructor(
            data_np, clusts, shapes, groups
        )

        result["edge_index"] = edge_index

        # Fetch the node features
        node_features = self.node_encoder(
            data, clusts, coord_label=coord_label, points=points, extra=extra
        )

        if isinstance(node_features, tuple):
            # If the output of the node encoder is a tuple, separate points
            node_features, points = node_features
            start_points, end_points = points.tensor.split(3, dim=1)

            result["start_points"] = TensorBatch(
                start_points, points.counts, coord_cols=np.array([0, 1, 2])
            )
            result["end_points"] = TensorBatch(
                end_points, points.counts, coord_cols=np.array([0, 1, 2])
            )

        # Fetch the edge features
        edge_features = self.edge_encoder(
            data, clusts, edge_index, closest_index=closest_index
        )

        # Feath the global_features
        global_features = None
        if self.global_encoder is not None:
            global_features = self.global_encoder(data, clusts)

        # Bring edge_index and batch_ids to device
        # TODO: try to keep everything (apart from clusts?) on GPU?
        index = torch.tensor(edge_index.index, device=data.tensor.device)
        xbatch = torch.tensor(clusts.batch_ids, device=data.tensor.device)

        # Pass through the model, update results
        out = self.gnn(node_features, index, edge_features, global_features, xbatch)

        # Loop over the necessary node/edge/global predictions, store output
        for t in self.out_types:
            for key in getattr(self, f"{t}_pred_keys"):
                result[key] = getattr(self, key)(out[f"{t}_features"])

        # If requested, build node groups from edge predictions
        if self.make_groups:
            assert (
                "edge_pred" in result
            ), "Must provide edge predictions to build node groups."
            edge_pred = result["edge_pred"].to_numpy()
            if self.grouping_method == "threshold":
                result["group_pred"] = node_assignment_batch(
                    edge_index, edge_pred, clusts
                )
            elif self.grouping_method == "score":
                if not self.grouping_through_track:
                    result["group_pred"] = node_assignment_score_batch(
                        edge_index, edge_pred, clusts
                    )
                else:
                    assert (
                        shapes is not None
                    ), "Must provide shapes to restrict track association."
                    track_node = TensorBatch(
                        shapes.data == TRACK_SHP, counts=shapes.counts
                    )
                    result["group_pred"] = node_assignment_score_batch(
                        edge_index, edge_pred, clusts, track_node
                    )

            else:
                raise ValueError(
                    "Group prediction algorithm not " "recognized:",
                    self.grouping_method,
                )

        return result


class GrapPALoss(torch.nn.modules.loss._Loss):
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

    Each of the specific loss blocks can also contain multiple losses by
    providing a name key in a loss block nested below it. Each loss name of a
    specific type should be provided with a corresponding output from GRaPA.

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    def __init__(self, grappa_loss, grappa=None):
        """Initialize the GrapPA loss function.

        Parameters
        ----------
        grappa_loss : dict
            Loss configuration
        grappa : dict, optional
            Model configuration
        """
        # Initialize the parent class
        super().__init__()

        # Process the loss configuration
        self.process_loss_config(**grappa_loss)

    def process_loss_config(self, node_loss=None, edge_loss=None, global_loss=None):
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
        self.out_types = ["node", "edge", "global"]
        assert (
            node_loss is not None or edge_loss is not None or global_lsos is not None
        ), (
            "Must provide either a `node_loss`, `edge_loss` or "
            "`global_loss` to the GrapPA loss function."
        )

        # Initialize the node/edge/global losses
        self.process_single_loss_config("node", node_loss, node_loss_factory)
        self.process_single_loss_config("edge", edge_loss, edge_loss_factory)
        self.process_single_loss_config("global", global_loss, global_loss_factory)

    def process_single_loss_config(self, prefix, loss, constructor):
        """Process a loss configuration.

        Parameters
        ----------
        prefix : dict
            Name of the output type to apply the loss to
        loss : Union[int, dict]
            Loss configuration
        constructor : object
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
            # Otherwise, initialzie one loss per prediction type
            for key, cfg in loss.items():
                loss_key = f"{prefix}_{key}_loss"
                loss_keys.append(loss_key)
                setattr(self, loss_key, constructor(cfg))

        setattr(self, f"{prefix}_loss_keys", loss_keys)

    def forward(
        self, clust_label, coord_label=None, graph_label=None, iteration=None, **output
    ):
        """Apply the node/edge/global losses to the logits from GrapPA.

        Parameters
        ----------
        clust_label : TensorBatch
            (N, 1 + D + N_f) Tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is is the number of cluster labels
        coord_label : TensorBatch, optional
            (P, 1 + D + 8) Tensor of start/end point labels for each
            true particle in the image
        graph_label : EdgeIndexTensor, optional
            (2, E) Tensor of edges that correspond to physical
            connections between true particle in the image
        iteration : int, optional
            Iteration index
        **output : dict
            Output of the GrapPA model
        """
        # Loop and apply the losses
        result = {}
        num_losses = 0
        loss, accuracy = 0.0, 0.0
        for t in self.out_types:
            loss_keys = getattr(self, f"{t}_loss_keys")
            for key in loss_keys:
                # If the number of loss keys is > 1 for this type of
                # prediction, must rename the prediction appropriately
                extra = {}
                if len(loss_keys) > 1:
                    extra[f"{t}_pred"] = output[key.replace("loss", "pred")]

                # Compute the loss
                out = getattr(self, key)(
                    clust_label=clust_label,
                    coord_label=coord_label,
                    graph_label=graph_label,
                    iteration=iteration,
                    **output,
                    **extra,
                )

                # Increment the loss and accuracy
                loss += out["loss"]
                accuracy += out["accuracy"]
                num_losses += 1

                # Update the result dictionary
                prefix = "_".join(key.split("_")[:-1])
                for k, v in out.items():
                    result[f"{prefix}_{k}"] = v

        # Append the total loss and total accuracy
        result["loss"] = torch.sum(loss) / num_losses
        result["accuracy"] = np.sum(accuracy) / num_losses

        return result
