"""Module that defines a generic node classification loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from spine.constants import TRACK_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.utils.gnn.cluster import get_cluster_label_batch

__all__ = ["NodeOrientLoss"]


class NodeOrientLoss(torch.nn.Module):
    """Loss to learn how to point a track node in the right direction.

    Takes the 2-channel node output of the GNN and optimizes node-wise scores
    such that the score corresponding to the correct orientation is maximized.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: orient
                <dictionary of arguments to pass to the loss>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    # Name of the loss (as specified in the configuration)
    name = "orient"

    # Alternative allowed names of the loss
    aliases = ("orientation",)

    def __init__(self, loss: str | dict[str, Any] = "ce") -> None:
        """Initialize the node orientation loss function.

        Parameters
        ----------
        loss : str, default 'ce'
            Name of the loss function to apply
        """
        # Initialize the parent class
        super().__init__()

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        coord_label: TensorBatch,
        clusts: IndexBatch,
        node_pred: TensorBatch,
        start_points: TensorBatch,
        end_points: TensorBatch,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | float | int]:
        """Applies the node orientation loss to a batch of data.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        coord_label : TensorBatch, optional
            (P, 1 + D + 8) Tensor of start/end point labels for each
            true particle in the image
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        node_pred : TensorBatch
            (C, 2) Node prediction logits (binary output)
        start_points : TensorBatch
            (C, 3) Start point features associated with each node
        end_points : TensorBatch
            (C, 3) End point features associated with each node
        **kwargs : dict, optional
            Other labels/outputs of the model which are not relevant here

        Returns
        -------
        loss : torch.Tensor
            Value of the loss
        accuracy : float
            Value of the node-wise orientation accuracy
        count : int
            Number of nodes the loss was applied to
        """
        # Fetch the true particle associations and the shape
        part_ids = get_cluster_label_batch(clust_label, clusts, column="particle")
        part_ids_array = part_ids.numpy_tensor()
        global_part_ids = np.empty_like(part_ids_array, dtype=np.int64)
        for batch_id in range(part_ids.batch_size):
            shift_part_ids = part_ids[batch_id].copy()
            valid_index = np.where(shift_part_ids > -1)[0]
            shift_part_ids[valid_index] += int(coord_label.edges[batch_id])
            lower = part_ids.edges[batch_id]
            upper = part_ids.edges[batch_id + 1]
            global_part_ids[lower:upper] = shift_part_ids

        # Restrict the loss to matched track clusters
        shapes = get_cluster_label_batch(clust_label, clusts, column="shape")
        valid_index = np.where(
            (global_part_ids > -1) & (shapes.numpy_tensor() == TRACK_SHP)
        )[0]

        # Fetch the true directions from the particle associations
        index = global_part_ids[valid_index]
        true_starts = coord_label.coordinates("start").torch_tensor()[index]
        true_ends = coord_label.coordinates("end").torch_tensor()[index]
        true_dirs = true_ends - true_starts

        # Restrict the start/end points, compute the vector
        start_tensor = start_points.torch_tensor()[valid_index]
        end_tensor = end_points.torch_tensor()[valid_index]
        feat_dirs = end_tensor - start_tensor

        # For each node, check whether the vector that joins the start to end
        # point node features are aligned with the ground truth
        node_assn = torch.sign(torch.sum(true_dirs * feat_dirs, dim=1)).long()
        node_assn = (node_assn + 1) // 2

        # Compute the loss
        node_pred_tensor = node_pred.torch_tensor()[valid_index]
        loss = self.loss_fn(node_pred_tensor, node_assn, reduction="sum")
        if len(valid_index) > 0:
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned nodes)
        acc = 1.0
        if len(valid_index) > 0:
            acc = float(torch.sum(torch.argmax(node_pred_tensor, dim=1) == node_assn))
            acc /= len(valid_index)

        return {"accuracy": acc, "loss": loss, "count": len(valid_index)}
