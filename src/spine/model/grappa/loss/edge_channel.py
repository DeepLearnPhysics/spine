"""Module that defines an edge classification loss (ON vs OFF)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from spine.cluster.label import get_cluster_label_batch
from spine.data import ClusterLabelBatch, EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.model.common.quality import (
    ClusterOverlapCache,
    ClusterQualityFilter,
)
from spine.model.common.weighting import get_class_weights
from spine.model.grappa.evaluation import (
    edge_assignment_batch,
    edge_assignment_forest_batch,
    edge_assignment_from_graph_batch,
    edge_purity_mask_batch,
)

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
        self,
        target: str,
        mode: str = "group",
        loss: str | dict[str, Any] = "ce",
        balance_loss: bool = False,
        high_purity: bool = False,
        *,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str | None = None,
    ) -> None:
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
        min_iou : float or sequence of float, optional
            Minimum IoU required of both endpoint clusters. A sequence provides
            separate requirements for OFF and ON edges, in that order.
        min_purity : float or sequence of float, optional
            Minimum purity required of both endpoint clusters, shared or
            specified for the OFF and ON edge classes.
        min_efficiency : float or sequence of float, optional
            Minimum efficiency required of both endpoint clusters, shared or
            specified for the OFF and ON edge classes.
        match_target : str, optional
            Truth-instance field used to evaluate endpoint quality. Defaults
            to the edge aggregation ``target``.
        """
        # Initialize the parent class
        super().__init__()

        # Parse the aggregation target
        self.target = target

        # Initialize basic parameters
        self.mode = mode
        self.balance_loss = balance_loss
        self.high_purity = high_purity

        # Apply the same overlap policy to both endpoints of a supervised edge
        self.quality_filter = ClusterQualityFilter(
            min_iou,
            min_purity,
            min_efficiency,
            match_target=match_target or target,
            num_classes=2,
        )

        if self.high_purity and self.target != "group":
            raise ValueError(
                "The `high_purity` flag is only valid when building shower groups."
            )
        if self.mode == "forest" and self.quality_filter.class_dependent:
            raise ValueError(
                "Forest edge losses require scalar overlap thresholds because "
                "the edge classes are defined by the target spanning tree."
            )

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        edge_pred: TensorBatch,
        true_edge_index: EdgeIndexBatch | None = None,
        overlap_cache: ClusterOverlapCache | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | float | int]:
        """Applies the edge channel loss to a batch of data.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        edge_index : EdgeIndexBatch
            (2, E) Sparse incidence matrix between clusters
        edge_pred : TensorBatch
            (E, 2) Edge prediction logits (binary output)
        true_edge_index : EdgeIndexBatch
            (2, E') True reference sparse incidence matrix
        overlap_cache : ClusterOverlapCache, optional
            Cluster overlaps shared by the objectives in one GrapPA forward.
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
        count_rejected : int, optional
            Number of otherwise valid edges removed because at least one
            endpoint failed overlap quality. Only returned when thresholds are
            configured.
        """
        # Start to build a mask of valid edges. Check that the group ID
        # of both clusters edge joins has a valid group ID
        group_ids = get_cluster_label_batch(clust_label, clusts, self.target)
        valid_mask = np.all(
            group_ids.numpy_tensor()[edge_index.index] > -1,
            axis=0,
        )

        # If requested, check that each group contains a single shower primary
        if self.high_purity:
            part_ids = get_cluster_label_batch(clust_label, clusts, "particle")
            prim_ids = get_cluster_label_batch(clust_label, clusts, "group_primary")
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
            forest_group_ids = group_ids
            if self.quality_filter.active:
                # Give each rejected node a unique group before constructing
                # the forest. This prevents a valid target path from using an
                # endpoint that will subsequently be removed from the loss.
                node_quality_mask = self.quality_filter.node_mask(
                    clust_label,
                    clusts,
                    cache=overlap_cache,
                )
                forest_ids = group_ids.numpy_tensor().copy()
                invalid_index = np.where(~node_quality_mask)[0]
                next_id = int(np.max(forest_ids, initial=-1)) + 1
                forest_ids[invalid_index] = next_id + np.arange(len(invalid_index))
                forest_group_ids = TensorBatch(forest_ids, group_ids.counts)

            edge_assn, valid_mask_mst = edge_assignment_forest_batch(
                edge_index,
                edge_pred.to_numpy(),
                forest_group_ids,
            )
            valid_mask &= valid_mask_mst.numpy_tensor()

        elif self.mode == "particle_forest":
            # If an edge matches a parentage relation, mark it as on
            if true_edge_index is None:
                raise ValueError(
                    "Must provide true `true_edge_index` object when using "
                    "the `particle_forest` truth mode"
                )

            part_ids = get_cluster_label_batch(clust_label, clusts, "particle")
            edge_assn = edge_assignment_from_graph_batch(
                edge_index, true_edge_index, part_ids
            )

        else:
            raise ValueError(f"Loss mode not recognized: {self.mode}")

        # An edge is trustworthy only when both endpoint clusters satisfy the
        # requested policy. Class-dependent thresholds use the ON/OFF target.
        count_rejected = 0
        if self.quality_filter.active:
            edge_quality_mask = self.quality_filter.edge_mask(
                clust_label,
                clusts,
                edge_index,
                edge_assn.numpy_tensor(),
                overlap_cache,
            )
            count_rejected = int(np.count_nonzero(valid_mask & ~edge_quality_mask))
            valid_mask &= edge_quality_mask

        # Apply the mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        edge_assn = edge_assn.to_tensor(dtype=torch.long, device=edge_pred.device)
        edge_pred_tensor = edge_pred.torch_tensor()[valid_index]
        edge_assn_tensor = edge_assn.torch_tensor()[valid_index]

        # Compute the loss. Balance classes if requested
        weights = None
        if self.balance_loss:
            weights = get_class_weights(edge_assn_tensor, num_classes=2)

        loss = self.loss_fn(
            edge_pred_tensor,
            edge_assn_tensor,
            weight=weights,
            reduction="sum",
        )
        if len(valid_index) > 0:
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned edges)
        acc = 1.0
        if len(valid_index) > 0:
            acc = float(
                torch.sum(torch.argmax(edge_pred_tensor, dim=1) == edge_assn_tensor)
            )
            acc /= len(valid_index)

        result = {"accuracy": acc, "loss": loss, "count": len(valid_index)}
        if self.quality_filter.active:
            result["count_rejected"] = count_rejected

        return result
