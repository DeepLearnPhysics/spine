"""Module that defines a vertex identification loss using node predictions."""

from warnings import warn

import numpy as np
import torch

from spine.data import Meta, TensorBatch
from spine.model.layer.factories import loss_fn_factory
from spine.utils.geo import Geometry
from spine.utils.globals import PRINT_COL, VTX_COLS
from spine.utils.gnn.cluster import get_cluster_label_batch

from .node_class import NodeClassLoss

__all__ = ["NodeVertexLoss"]


class NodeVertexLoss(torch.nn.Module):
    """Loss used to predict the position of the vertex within each interaction.

    This loss formulates the problem as a node problem:
    - Predict which nodes are primary nodes (originate from the vertex);
    - Primary nodes predict the vertex position;
    - The positions predicted by each primary particle are aggregated
      downstream to form a vertex prediction for each interaction.

    This loss expects 5 outputs per node:
    - 2 for the primary identification
    - 3 for the position regression

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: vertex
                <dictionary of arguments to pass to the loss>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    # Name of the loss (as specified in the configuration)
    name = "vertex"

    def __init__(
        self,
        balance_primary_loss=False,
        primary_loss="ce",
        regression_loss="mse",
        only_contained=True,
        normalize_positions=False,
        use_anchor_points=False,
        return_vertex_labels=False,
        detector=None,
        geometry_file=None,
    ):
        """Initialize the vertex regression loss function.

        Parameters
        ----------
        balance_primary_loss : bool, default `False`
            Whether to weight the primary loss to account for class imbalance
        primary_loss : str, default `'ce'`
            Name of the loss function used to predict interaction primaries
        regression_loss : str, default `'mse'`
            Name of the loss function used to predict the vertex position
        only_contained : bool, default `True`
            Only considers label vertices contained in the active volume
        normalize_positions : bool, default `False`
            Normalize the target position between 0 and 1
        use_anchor_points : bool, default `False`
            Predict positions w.r.t. to the particle end points
        return_vertex_labels : bool, default `False`
            If `True`, return the list vertex labels (one per particle)
        detector : str, optional
            Name of a recognized detector to the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        """
        # Initialize the parent class
        super().__init__()

        # Initialize basic parameters
        self.balance_primary_loss = balance_primary_loss
        self.only_contained = only_contained
        self.normalize_positions = normalize_positions
        self.use_anchor_points = use_anchor_points
        self.return_vertex_labels = return_vertex_labels

        # Initialize the primary identification loss
        self.primary_loss = NodeClassLoss(
            target="inter_primary", balance_loss=balance_primary_loss, loss=primary_loss
        )

        # Initialize the regression loss
        self.reg_loss_fn = loss_fn_factory(regression_loss, reduction="sum")

        # If containment is requested, intialize geometry
        if self.only_contained:
            self.geo = Geometry(detector, geometry_file)
            self.geo.define_containment_volumes(margin=0.0, mode="module")

    def forward(
        self,
        clust_label,
        clusts,
        node_pred,
        meta=None,
        start_points=None,
        end_points=None,
        **kwargs,
    ):
        """Applies the node type loss to a batch of data.

        Parameters
        ----------
        clust_label : TensorBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        node_pred : TensorBatch
            (C, 2) Node prediction logits (binary output)
        meta : List[Meta], optional
            Image metadata information
        start_points : TensorBatch, optional
            (C, 3) Node start positions
        end_points : TensorBatch, optional
            (C, 3) Node end positions
        **kwargs : dict, optional
            Other labels/outputs of the model which are not relevant here

        Returns
        -------
        loss : torch.Tensor
            Value of the loss
        accuracy : float
            Value of the node-wise classification accuracy
        count : int
            Number of nodes the loss was applied to
        primary_loss : torch.Tensor
            Value of the primary classification loss
        primary_accuracy : float
            Value of the primary classification accuracy
        reg_loss : torch.Tensor
            Value of the vertex regression loss
        reg_accuracy : float
            Value of the vertex regression accuracy
        """
        # Ensure that the predictions are of the expected shape, split them
        assert node_pred.shape[1] == 5, (
            "The output used for vertex prediction should contain 5 "
            "features, 2 used for primary prediction and 3 for regression."
        )

        primary_pred, vertex_pred = torch.tensor_split(node_pred.tensor, [2], dim=1)

        primary_pred = TensorBatch(primary_pred, node_pred.counts)
        vertex_pred = TensorBatch(vertex_pred, node_pred.counts)

        # Compute the primary identification loss
        result_primary = self.primary_loss(clust_label, clusts, primary_pred)

        # If containment or normalization are requested, ensure meta is provided
        if self.only_contained or self.normalize_positions:
            assert meta is not None, (
                "Must provide `meta` to check containement or normalize "
                "vertex positions."
            )

        # Get the interaction primary labels and the vertex position labels
        # TODO: could modify `get_cluster_label_batch` to accept a column list
        primary_ids = get_cluster_label_batch(clust_label, clusts, column=PRINT_COL)

        vertex_labels = np.empty((len(clusts.index_list), 3), dtype=primary_ids.dtype)
        for i, col in enumerate(VTX_COLS):
            vertex_labels[:, i] = get_cluster_label_batch(
                clust_label, clusts, column=col
            ).tensor
        vertex_labels = TensorBatch(vertex_labels, primary_ids.counts)

        # Create a mask for valid nodes (-1 indicates invalid labels,
        # 0 indicates a secondary)
        valid_mask = primary_ids.tensor > 0

        # If requested, check that the vertexes are contained
        if self.only_contained:
            contain_mask = np.empty(len(clusts.index_list), dtype=bool)
            for b in range(vertex_labels.batch_size):
                lower, upper = vertex_labels.edges[b], vertex_labels.edges[b + 1]
                points = meta[b].to_cm(vertex_labels[b])
                contain_mask[lower:upper] = self.geo.check_containment(
                    points, summarize=False
                )

            valid_mask &= contain_mask

        # If requested, normalize the target positions to the detector size
        if self.normalize_positions:
            ranges = (meta[0].upper - meta[0].lower) / meta[0].size
            vertex_labels = TensorBatch(
                vertex_labels.tensor / ranges, vertex_labels.counts
            )

        # If requested, anchor predicted positions to the closest particle point
        if self.use_anchor_points:
            # Check that we have particle end poins
            assert (
                start_points is not None and end_points is not None
            ), "Must provided particle end poins to anchor predictions."

            # Get the particle end points, scale if necessary
            points = torch.hstack((start_points.tensor, end_points.tensor)).view(
                -1, 2, 3
            )
            if self.normalize_positions:
                ranges = (meta[0].upper - meta[0].lower) / meta[0].size
                points = points / torch.tensor(ranges, device=points.device)

            # Get the closest particle end point for each prediction
            dist_to_anchor = torch.norm(
                vertex_pred.tensor.view(-1, 1, 3) - points, dim=2
            )
            min_index = torch.argmin(dist_to_anchor, dim=1)
            range_index = torch.arange(len(points), device=points.device).long()
            anchors = points[range_index, min_index, :]

            # Update the predictions so that the offset w.r.t. to anchor
            # points is predicted instead of the raw position
            vertex_pred = TensorBatch(anchors + vertex_pred.tensor, vertex_pred.counts)

        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        vertex_assn = vertex_labels.to_tensor(device=node_pred.device)
        vertex_assn = vertex_assn.tensor[valid_index]
        vertex_pred = vertex_pred.tensor[valid_index]

        # Compute the regression loss
        reg_loss = self.reg_loss_fn(vertex_pred, vertex_assn)
        if len(valid_index):
            reg_loss /= len(valid_index)

        # Compute accuracy of assignment (average distance)
        # TODO: Come up with a better implementation (between 0 and 1?)
        reg_acc = 1.0
        if len(valid_index):
            dists = torch.norm(vertex_pred - vertex_assn)
            reg_acc = float(torch.mean(dists))

        # Build the result dictionary
        result = {
            "loss": (reg_loss + result_primary["loss"]) / 2,
            "accuracy": (reg_acc + result_primary["accuracy"]) / 2,
            "reg_accuracy": reg_acc,
            "reg_loss": reg_loss,
            "reg_count": len(valid_index),
            **{f"primary_{k}": v for k, v in result_primary.items()},
        }

        if self.return_vertex_labels:
            result["labels"] = vertex_labels

        return result
