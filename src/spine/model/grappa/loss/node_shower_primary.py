"""Module that defines an EM shower primary identification loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from spine.cluster.label import (
    get_cluster_closest_primary_label_batch,
    get_cluster_label_batch,
)
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.model.common.quality import (
    ClusterOverlapCache,
    ClusterQualityFilter,
)
from spine.model.common.weighting import get_class_weights
from spine.model.grappa.evaluation import node_purity_mask_batch

from .target import prepare_cached_target, target_tensor, validity_batch

__all__ = ["NodeShowerPrimaryLoss"]


class NodeShowerPrimaryLoss(torch.nn.Module):
    """Loss used to train the EM shower primary identification.

    Takes the two-channel node output of the GNN and optimizes node-wise scores
    such that nodes that initiate a particle cascade are given a high score
    (exclusively relevant for showers for now).

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: shower_primary
                <dictionary of arguments to pass to the loss>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    # Name of the loss (as specified in the configuration)
    name = "shower_primary"

    def __init__(
        self,
        loss: str | dict[str, Any] = "ce",
        balance_loss: bool = False,
        high_purity: bool = False,
        use_closest: bool = False,
        use_group_pred: bool = False,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str = "group",
    ) -> None:
        """Initialize the EM shower primary identification loss function.

        Parameters
        ----------
        loss : str, default 'ce'
            Name of the loss function to apply
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        high_purity : bool, default False
            Only apply loss to nodes which belong to a sensible group, i.e.
            one with exactly one primary in it (not 0, not > 1)
        use_closest : bool, default False
            For each group, pick the fragment which is closest to the start
            point of the shower as the primary (more robust to fragment breaks)
        use_group_pred : bool, default False
            Use predicted group to check for high purity
        min_iou : float or sequence of float, optional
            Minimum truth-instance IoU, shared or specified for secondary and
            primary shower fragments.
        min_purity : float or sequence of float, optional
            Minimum predicted-cluster purity, shared or specified for secondary
            and primary shower fragments.
        min_efficiency : float or sequence of float, optional
            Minimum truth-instance efficiency, shared or specified for
            secondary and primary shower fragments.
        match_target : str, default 'group'
            Truth-instance field used to evaluate overlap quality.
        """
        # Initialize the parent class
        super().__init__()

        # Initialize basic parameters
        self.balance_loss = balance_loss
        self.high_purity = high_purity
        self.use_closest = use_closest
        self.use_group_pred = use_group_pred

        # The binary quality classes follow the secondary/primary target IDs
        self.quality_filter = ClusterQualityFilter(
            min_iou,
            min_purity,
            min_efficiency,
            match_target=match_target,
            num_classes=2,
        )

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(
        self,
        node_pred: TensorBatch,
        clust_label: ClusterLabelBatch | None = None,
        clusts: IndexBatch | None = None,
        coord_label: TensorBatch | None = None,
        group_pred: TensorBatch | None = None,
        overlap_cache: ClusterOverlapCache | None = None,
        labels: TensorBatch | None = None,
        valid_mask: TensorBatch | None = None,
        return_target: bool = False,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | TensorBatch | float | int]:
        """Applies the shower primary loss to a batch of data.

        Parameters
        ----------
        node_pred : TensorBatch
            (C, 2) Node prediction logits (binary output)
        clust_label : ClusterLabelBatch, optional
            (N, 1 + D + N_f) Cluster labels used to construct live primary
            targets. May be omitted with cached supervision.
        clusts : IndexBatch, optional
            (C) Cluster-to-voxel index used to construct live targets. May be
            omitted when ``labels`` and ``valid_mask`` are supplied.
        coord_label : TensorBatch, optional
            (P, 1 + D + 8) Label start, end, time and shape for each point
        group_pred : TensorBatch, optional
            (C) Predicted group to which each node belongs to
        overlap_cache : ClusterOverlapCache, optional
            Cluster overlaps shared by the objectives in one GrapPA forward.
        labels : TensorBatch, optional
            Cached primary labels aligned with ``node_pred``. Must be supplied
            together with ``valid_mask``.
        valid_mask : TensorBatch, optional
            Cached one-dimensional static node validity mask. Predicted-group
            purity is reapplied from the current ``group_pred`` every time.
        return_target : bool, default False
            If `True`, return the primary labels and static validity mask which
            can be reused safely in a later training iteration.
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
        count_rejected : int, optional
            Number of otherwise valid shower fragments removed by overlap
            quality. Only returned when thresholds are configured.
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
            primary_ids = labels
            static_valid = prepare_cached_target(labels, valid_mask, node_pred, "node")
        else:
            if clust_label is None or clusts is None:
                raise ValueError(
                    "Shower-primary loss requires either cached supervision or "
                    "both structured cluster labels and clusters."
                )
            primary_ids, static_valid, count_rejected = self._build_target(
                clust_label,
                clusts,
                coord_label,
                overlap_cache,
            )

        # Preserve only iteration-independent validity in the cache product.
        if valid_mask is None:
            valid_mask = validity_batch(static_valid, primary_ids)

        # Predicted grouping changes during training and must be reapplied to
        # both freshly built and cached targets on every forward pass.
        valid_array = static_valid.copy()
        if self.high_purity and self.use_group_pred:
            if group_pred is None:
                raise ValueError("If using group predictions, must provide them.")
            valid_array &= node_purity_mask_batch(group_pred, primary_ids)

        valid_index = np.where(valid_array)[0]
        node_assn_tensor = target_tensor(primary_ids, node_pred, dtype=torch.long)[
            valid_index
        ]
        node_pred_tensor = node_pred.torch_tensor()[valid_index]

        # Compute the loss. Balance classes if requested
        weights = None
        if self.balance_loss:
            weights = get_class_weights(node_assn_tensor, num_classes=2)

        loss = self.loss_fn(
            node_pred_tensor,
            node_assn_tensor,
            weight=weights,
            reduction="sum",
        )
        if len(valid_index) > 0:
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned nodes)
        acc = 1.0
        if len(valid_index) > 0:
            acc = float(
                torch.sum(torch.argmax(node_pred_tensor, dim=1) == node_assn_tensor)
            )
            acc /= len(valid_index)

        result = {"accuracy": acc, "loss": loss, "count": len(valid_index)}
        if labels is None and self.quality_filter.active:
            result["count_rejected"] = count_rejected
        # Expose only the stable supervision needed by a later iteration.
        if return_target:
            result["target"] = primary_ids
            result["valid"] = valid_mask

        return result

    def _build_target(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        coord_label: TensorBatch | None,
        overlap_cache: ClusterOverlapCache | None,
    ) -> tuple[TensorBatch, np.ndarray, int]:
        """Build node-aligned shower-primary supervision from truth.

        The helper applies closest-fragment, truth-group purity and overlap
        rules. Predicted-group purity is intentionally deferred to
        :meth:`forward` because it changes between training iterations.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Voxel-level labels used to identify primary shower fragments.
        clusts : IndexBatch
            Cluster-to-voxel index whose ordering defines the node axis.
        coord_label : TensorBatch, optional
            Particle creation points required by ``use_closest``.
        overlap_cache : ClusterOverlapCache, optional
            Precomputed cluster overlaps shared across GrapPA objectives.

        Returns
        -------
        TensorBatch
            Binary primary labels aligned with graph nodes.
        np.ndarray
            Static Boolean mask suitable for caching. Predicted-group purity
            is not represented in this mask.
        int
            Number of otherwise eligible nodes rejected by overlap quality.
        """
        # Build the static supervision and validity mask from structured truth.
        primary_ids = get_cluster_label_batch(
            clust_label, clusts, column="group_primary"
        )
        valid_array = primary_ids.to_numpy().data > -1

        # Optionally identify the fragment nearest the shower creation point.
        if self.use_closest:
            if coord_label is None:
                raise ValueError(
                    "To use the shower fragment closest to the shower creation "
                    "point as the primary fragment, must provide `coord_label`."
                )
            primary_ids = get_cluster_closest_primary_label_batch(
                clust_label, coord_label, clusts, primary_ids
            )
            valid_array &= primary_ids.numpy_tensor() > -1

        # Truth-group purity is static; predicted-group purity is applied later.
        if self.high_purity and not self.use_group_pred:
            group_ids = get_cluster_label_batch(clust_label, clusts, column="group")
            valid_array &= node_purity_mask_batch(group_ids, primary_ids)

        # Exclude unstable targets according to the overlap-quality policy.
        count_rejected = 0
        if self.quality_filter.active:
            quality_mask = self.quality_filter.node_mask(
                clust_label,
                clusts,
                primary_ids.numpy_tensor(),
                overlap_cache,
            )
            count_rejected = int(np.count_nonzero(valid_array & ~quality_mask))
            valid_array &= quality_mask

        return primary_ids, valid_array, count_rejected
