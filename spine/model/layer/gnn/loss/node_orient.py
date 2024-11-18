"""Module that defines a generic node classification loss."""

import torch
import numpy as np
from warnings import warn

from spine.model.layer.factories import loss_fn_factory

from spine.data import IndexBatch
from spine.utils.globals import (
        PART_COL, SHAPE_COL, COORD_START_COLS, COORD_END_COLS, TRACK_SHP)
from spine.utils.gnn.cluster import get_cluster_label_batch

__all__ = ['NodeOrientLoss']


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
    name = 'orient'

    # Alternative allowed names of the loss
    aliases = ('orientation',)

    def __init__(self, loss='ce'):
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

    def forward(self, clust_label, coord_label, clusts, node_pred,
                start_points, end_points, **kwargs):
        """Applies the node orientation loss to a batch of data.

        Parameters
        ----------
        clust_label : TensorBatch
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
        part_ids = get_cluster_label_batch(clust_label, clusts, column=PART_COL)
        global_part_ids = np.empty_like(part_ids.tensor, dtype=np.int64)
        for b in range(part_ids.batch_size):
            shift_part_ids = part_ids[b]
            valid_index = np.where(shift_part_ids > -1)[0]
            shift_part_ids[valid_index] += coord_label.edges[b].item()
            lower, upper = part_ids.edges[b], part_ids.edges[b+1]
            global_part_ids[lower:upper] = shift_part_ids

        # Restrict the loss to matched track clusters
        shapes = get_cluster_label_batch(clust_label, clusts, column=SHAPE_COL)
        valid_index = np.where(
                (global_part_ids > -1) & (shapes.tensor == TRACK_SHP))[0]

        # Fetch the true directions from the particle associations
        all_cols = np.concatenate((COORD_START_COLS, COORD_END_COLS))
        index = global_part_ids[valid_index]
        true_starts, true_ends = (
                coord_label.tensor[index][:, all_cols].split(3, dim=1))
        true_dirs = true_ends - true_starts

        # Restrict the start/end points, compute the vector
        start_points = start_points.tensor[valid_index]
        end_points = end_points.tensor[valid_index]
        feat_dirs = end_points - start_points

        # For each node, check whether the vector that joins the start to end
        # point node features are aligned with the ground truth
        node_assn = torch.sign(torch.sum(true_dirs*feat_dirs, dim=1)).long()
        node_assn = (node_assn + 1)//2

        # Compute the loss
        node_pred = node_pred.tensor[valid_index]
        loss = self.loss_fn(node_pred, node_assn, reduction='mean')

        # Compute accuracy of assignment (fraction of correctly assigned nodes)
        acc = 1.
        if len(valid_index):
            acc = float(torch.sum(torch.argmax(node_pred, dim=1) == node_assn))
            acc /= len(valid_index)

        return {
            'accuracy': acc,
            'loss': loss,
            'count': len(valid_index)
        }
