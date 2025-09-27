"""Module that defines an edge classification loss (ON vs OFF)."""

import numpy as np
import torch

from spine.model.layer.factories import loss_fn_factory
from spine.utils.enums import enum_factory
from spine.utils.globals import CLUST_COL, GROUP_COL, PART_COL, PRGRP_COL
from spine.utils.gnn.cluster import get_cluster_label_batch
from spine.utils.gnn.evaluation import (
    edge_assignment_batch,
    edge_assignment_from_graph_batch,
    edge_purity_mask_batch,
)
from spine.utils.weighting import get_class_weights

__all__ = ["EdgeChannelLoss"]


class EdgeChannelLoss(torch.nn.Module):
    """Takes the two-channel edge output of the GNN and optimizes edge-wise
    scores such that edges that connect nodes that belong to common instance
    are given a high score.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              edge_loss:
                name: channel
                <dictionary of arguments to pass to the loss>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    # Name of the GNN loss (as specified in the configuration)
    name = "channel"

    def __init__(
        self, target, mode="group", loss="ce", balance_loss=False, high_purity=False
    ):
        """Initialize the primary identification loss function.

        Parameters
        ----------
        target : str
            Column name in the label tensor specifying the aggregation target
        mode : str, default 'group'
            Loss mode, one of 'group', 'forest' or 'particle_forest'
            - 'group' turns every edge that connect two nodes that belong to
              the same group (same target value) on
            - 'forest' ensures that at least one path in the graph connects two
              nodes, if they belong to the same group
            - 'particle_forest' only turns on edges that join two particles
              have a parentage relationship in the true particle tree
        loss : Union[str, dict], default 'ce'
            Name of the loss function to apply
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        high_purity : bool, default False
            Only apply loss to nodes which belong to a sensible group, i.e.
            one with exactly one shower primary in it (not 0, not > 1)
        """
        # Initialize the parent class
        super().__init__()

        # Parse the aggregation target
        self.target = enum_factory("cluster", target)

        # Initialize basic parameters
        self.mode = mode
        self.balance_loss = balance_loss
        self.high_purity = high_purity

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(
        self, clust_label, clusts, edge_index, edge_pred, true_edge_index=None, **kwargs
    ):
        """Applies the edge channel loss to a batch of data.

        Parameters
        ----------
        clust_label : TensorBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        edge_index : EdgeIndexBatch
            (2, E) Sparse ncidence matrix between clusters
        edge_pred : TensorBatch
            (E, 2) Edge prediction logits (binary output)
        true_edge_index : EdgeIndexBatch
            (2, E') True reference sparse incidence matrix
        **kwargs : dict, optional
            Other labels/outputs of the model which are not relevant here

        Returns
        -------
        loss : torch.Tensor
            Value of the loss
        accuracy : float
            Value of the edge-wise classification accuracy
        count : int
            Number of edges the loss was applied to
        """
        # Start to build a mask of valid edges. Check that the group ID
        # of both clusters edge joins has a valid group ID
        group_ids = get_cluster_label_batch(clust_label, clusts, self.target)
        valid_mask = np.all(group_ids.tensor[edge_index.index] > -1, axis=0)

        # If requested, check that each group contains a single shower primary
        if self.high_purity:
            assert self.target == GROUP_COL, (
                "The `high_purity` flag is only valid when " "building shower groups."
            )

            part_ids = get_cluster_label_batch(clust_label, clusts, PART_COL)
            prim_ids = get_cluster_label_batch(clust_label, clusts, PRGRP_COL)
            valid_mask &= edge_purity_mask_batch(
                edge_index, part_ids, group_ids, prim_ids
            )

        # Compute the edge assignments
        if self.mode == "group":
            # If an edge joins two nodes in the same group, label it as on
            edge_assn = edge_assignment_batch(edge_index, group_ids)

        elif self.mode == "forest":
            # For each group, find the most likely spanning tree, label the
            # edges in the tree as 1. For all other edges, apply loss only if
            # in separate groups. If undirected, also assign symmetric path
            edge_assn, valid_mask_mst = edge_assignment_forest_batch(
                edge_index, edge_pred.to_numpy(), group_ids
            )
            valid_mask &= valid_mask_mst

        elif self.mode == "particle_forest":
            # If an edge matches a parentage relation, mark it as on
            assert true_edge_index is not None, (
                "Must provide true `true_edge_index` object when using "
                "the `particle_forest` truth mode"
            )

            part_ids = get_cluster_label_batch(clust_label, clusts, PART_COL)
            edge_assn = edge_assignement_from_graph_batch(
                edge_index, true_edge_index, part_ids
            )

        else:
            raise ValueError("Loss mode not recognized:", self.mode)

        # Apply the mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        edge_assn = edge_assn.to_tensor(dtype=torch.long, device=edge_pred.device)
        edge_pred = edge_pred.tensor[valid_index]
        edge_assn = edge_assn.tensor[valid_index]

        # Compute the loss. Balance classes if requested
        weights = None
        if self.balance_loss:
            weights = get_class_weights(edge_assn, num_classes=2)

        loss = self.loss_fn(edge_pred, edge_assn, weight=weights, reduction="sum")
        if len(valid_index):
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned edges)
        acc = 1.0
        if len(valid_index):
            acc = float(torch.sum(torch.argmax(edge_pred, dim=1) == edge_assn))
            acc /= len(valid_index)

        return {"accuracy": acc, "loss": loss, "count": len(valid_index)}
