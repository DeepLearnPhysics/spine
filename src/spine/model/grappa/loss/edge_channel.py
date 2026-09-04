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
from spine.model.grappa.augment import EdgeSelection
from spine.model.grappa.evaluation import (
    edge_assignment_batch,
    edge_assignment_forest_batch,
    edge_assignment_from_graph_batch,
    edge_purity_mask_batch,
)

from .target import (
    prepare_cached_target,
    prepare_cached_validity,
    target_tensor,
    validity_batch,
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
        edge_index: EdgeIndexBatch,
        edge_pred: TensorBatch,
        clust_label: ClusterLabelBatch | None = None,
        clusts: IndexBatch | None = None,
        true_edge_index: EdgeIndexBatch | None = None,
        overlap_cache: ClusterOverlapCache | None = None,
        labels: TensorBatch | None = None,
        valid_mask: TensorBatch | None = None,
        edge_keep: TensorBatch | None = None,
        return_target: bool = False,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | TensorBatch | float | int]:
        """Applies the edge channel loss to a batch of data.

        Parameters
        ----------
        edge_index : EdgeIndexBatch
            (2, E) Sparse incidence matrix between clusters
        edge_pred : TensorBatch
            (E, 2) Edge prediction logits (binary output)
        clust_label : ClusterLabelBatch, optional
            (N, 1 + D + N_f) Cluster labels used to construct live edge
            targets. May be omitted with cached supervision.
        clusts : IndexBatch, optional
            (C) Cluster-to-voxel index used to construct live targets. May be
            omitted when ``labels`` and ``valid_mask`` are supplied.
        true_edge_index : EdgeIndexBatch, optional
            (2, E') True reference sparse incidence matrix
        overlap_cache : ClusterOverlapCache, optional
            Cluster overlaps shared by the objectives in one GrapPA forward.
        labels : TensorBatch, optional
            Cached supervision supplied together with ``valid_mask``. This is
            an edge-aligned binary target in ``group`` and ``particle_forest``
            modes. In ``forest`` mode, it contains stable node group IDs from
            which the current prediction-dependent tree is rebuilt.
        valid_mask : TensorBatch, optional
            Cached one-dimensional static edge validity mask.
        edge_keep : TensorBatch, optional
            Training-time selection aligned with the original cached graph.
            Edge-aligned cached products are filtered before validation; the
            node-aligned target used in ``forest`` mode is preserved.
        return_target : bool, default False
            If `True`, return stable supervision which can be reused safely in
            a later training iteration.
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
        # Validate that cached labels and validity mask are provided together.
        if (labels is None) != (valid_mask is None):
            raise ValueError(
                "Cached labels and validity mask must be provided together."
            )

        # Reuse cached static supervision or derive it from structured truth.
        count_rejected = 0
        if labels is not None:
            assert valid_mask is not None
            # Cached supervision describes the graph before augmentation.
            # Forest targets are node-aligned, while every validity mask and
            # all other target modes are edge-aligned.
            if edge_keep is not None:
                edge_selection = EdgeSelection(edge_keep)
                valid_mask = edge_selection.filter_tensor(valid_mask)
                if self.mode != "forest":
                    labels = edge_selection.filter_tensor(labels)
            if self.mode == "forest":
                forest_group_ids = labels
                static_valid = self._prepare_cached_forest_target(
                    forest_group_ids,
                    valid_mask,
                    edge_index,
                    edge_pred,
                )
                edge_assn, forest_valid = edge_assignment_forest_batch(
                    edge_index,
                    edge_pred.to_numpy(),
                    forest_group_ids,
                )
                valid_array = static_valid & forest_valid.numpy_tensor()
                cache_target = forest_group_ids
            else:
                edge_assn = labels
                valid_array = prepare_cached_target(
                    labels, valid_mask, edge_pred, "edge"
                )
                static_valid = valid_array
                cache_target = edge_assn
        else:
            if clust_label is None or clusts is None:
                raise ValueError(
                    "Edge classification requires either cached supervision or "
                    "both structured cluster labels and clusters."
                )
            (
                edge_assn,
                valid_array,
                cache_target,
                static_valid,
                count_rejected,
            ) = self._build_target(
                clust_label,
                clusts,
                edge_index,
                edge_pred,
                true_edge_index,
                overlap_cache,
            )

        # Persist the stable mask, not a forest selection tied to old logits.
        if valid_mask is None:
            valid_mask = validity_batch(static_valid, edge_assn)
        valid_index = np.where(valid_array)[0]
        edge_pred_tensor = edge_pred.torch_tensor()[valid_index]
        edge_assn_tensor = target_tensor(edge_assn, edge_pred, dtype=torch.long)[
            valid_index
        ]

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
        if labels is None and self.quality_filter.active:
            result["count_rejected"] = count_rejected
        # Expose only the stable inputs needed to rebuild later supervision.
        if return_target:
            result["target"] = cache_target
            result["valid"] = valid_mask

        return result

    def _build_target(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        edge_pred: TensorBatch,
        true_edge_index: EdgeIndexBatch | None,
        overlap_cache: ClusterOverlapCache | None,
    ) -> tuple[TensorBatch, np.ndarray, TensorBatch, np.ndarray, int]:
        """Build edge-aligned supervision from structured truth.

        Static truth, purity and overlap decisions are separated from the
        prediction-dependent spanning-tree selection. In ``forest`` mode the
        cache target is therefore node-aligned group IDs, while the labels
        consumed by the current loss remain edge aligned.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Voxel-level truth used to label clusters and their relationships.
        clusts : IndexBatch
            Cluster-to-voxel index for the predicted graph nodes.
        edge_index : EdgeIndexBatch
            Graph edges whose order defines the supervision axis.
        edge_pred : TensorBatch
            Edge logits, used by forest mode to select its target tree.
        true_edge_index : EdgeIndexBatch, optional
            Reference particle graph required by ``particle_forest`` mode.
        overlap_cache : ClusterOverlapCache, optional
            Precomputed cluster overlaps shared across GrapPA objectives.

        Returns
        -------
        TensorBatch
            Binary labels aligned with the predicted edges.
        np.ndarray
            Effective Boolean mask selecting edges for the current loss.
        TensorBatch
            Stable cache target. This is node-aligned in ``forest`` mode and
            otherwise identical to the edge labels.
        np.ndarray
            Static edge-validity mask suitable for caching.
        int
            Number of otherwise eligible edges rejected by overlap quality.
        """
        # Build the static supervision and validity mask from structured truth.
        group_ids = get_cluster_label_batch(clust_label, clusts, self.target)
        static_valid = np.all(
            group_ids.numpy_tensor()[edge_index.index] > -1,
            axis=0,
        )

        # Optionally require each shower group to contain one true primary.
        if self.high_purity:
            part_ids = get_cluster_label_batch(clust_label, clusts, "particle")
            prim_ids = get_cluster_label_batch(clust_label, clusts, "group_primary")
            static_valid &= edge_purity_mask_batch(
                edge_index, part_ids, group_ids, prim_ids
            )

        # Construct the exact binary supervision requested by the loss mode.
        dynamic_valid = np.ones_like(static_valid)
        if self.mode == "group":
            edge_assn = edge_assignment_batch(edge_index, group_ids)
            cache_target = edge_assn

        elif self.mode == "forest":
            forest_group_ids = group_ids
            if self.quality_filter.active:
                # Prevent a target path from traversing a rejected endpoint.
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
            dynamic_valid = valid_mask_mst.numpy_tensor()
            cache_target = forest_group_ids

        elif self.mode == "particle_forest":
            if true_edge_index is None:
                raise ValueError(
                    "Must provide true `true_edge_index` object when using "
                    "the `particle_forest` truth mode"
                )
            part_ids = get_cluster_label_batch(clust_label, clusts, "particle")
            edge_assn = edge_assignment_from_graph_batch(
                edge_index, true_edge_index, part_ids
            )
            cache_target = edge_assn

        else:
            raise ValueError(f"Loss mode not recognized: {self.mode}")

        # Apply endpoint-quality requirements after constructing edge classes.
        count_rejected = 0
        if self.quality_filter.active:
            quality_classes = (
                None if self.mode == "forest" else edge_assn.numpy_tensor()
            )
            edge_quality_mask = self.quality_filter.edge_mask(
                clust_label,
                clusts,
                edge_index,
                quality_classes,
                overlap_cache,
            )
            count_rejected = int(np.count_nonzero(static_valid & ~edge_quality_mask))
            static_valid &= edge_quality_mask

        # The spanning tree is selected from the current edge logits. All other
        # modes have no dynamic validity component.
        valid_mask = static_valid & dynamic_valid

        return edge_assn, valid_mask, cache_target, static_valid, count_rejected

    @staticmethod
    def _prepare_cached_forest_target(
        group_ids: TensorBatch,
        valid_mask: TensorBatch,
        edge_index: EdgeIndexBatch,
        edge_pred: TensorBatch,
    ) -> np.ndarray:
        """Validate cached forest primitives on their distinct axes.

        Forest group IDs align with graph nodes, while static validity aligns
        with edge predictions. Keeping both partitions explicit prevents a
        cached tree selected from old logits from masquerading as stable truth.

        Parameters
        ----------
        group_ids : TensorBatch
            Cached node group IDs used to rebuild the target spanning forest.
        valid_mask : TensorBatch
            Cached static edge-validity mask.
        edge_index : EdgeIndexBatch
            Current graph whose node spans define the target partitioning.
        edge_pred : TensorBatch
            Current edge logits and their event partitioning.

        Returns
        -------
        np.ndarray
            Boolean static edge-validity mask on CPU.
        """
        if not isinstance(group_ids, TensorBatch):
            raise TypeError("Cached forest group labels must be TensorBatch.")

        group_counts = group_ids.counts
        if not isinstance(group_counts, np.ndarray):
            group_counts = group_counts.detach().cpu().numpy()
        node_spans = edge_index.spans
        if not isinstance(node_spans, np.ndarray):
            node_spans = node_spans.detach().cpu().numpy()
        if group_ids.shape[0] != int(np.sum(node_spans)) or not np.array_equal(
            group_counts, node_spans
        ):
            raise ValueError("Cached forest group labels must align with graph nodes.")

        return prepare_cached_validity(valid_mask, edge_pred, "edge")
