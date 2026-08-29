"""Module that defines a generic node classification loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from spine.cluster.label import get_cluster_label_batch
from spine.constants import TRACK_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.model.common.quality import (
    ClusterOverlapCache,
    ClusterQualityFilter,
)

from .target import prepare_cached_target, target_tensor, validity_batch

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

    def __init__(
        self,
        loss: str | dict[str, Any] = "ce",
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str = "particle",
    ) -> None:
        """Initialize the node orientation loss function.

        Parameters
        ----------
        loss : str, default 'ce'
            Name of the loss function to apply
        min_iou : float or sequence of float, optional
            Minimum truth-particle IoU, shared or specified for the backward
            and forward orientation targets.
        min_purity : float or sequence of float, optional
            Minimum predicted-cluster purity, shared or specified for the two
            orientation targets.
        min_efficiency : float or sequence of float, optional
            Minimum truth-particle efficiency, shared or specified for the two
            orientation targets.
        match_target : str, default 'particle'
            Truth-instance field used to evaluate overlap quality.
        """
        # Initialize the parent class
        super().__init__()

        # Configure optional cluster-quality filtering
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
        clust_label: ClusterLabelBatch | None,
        coord_label: TensorBatch | None,
        clusts: IndexBatch,
        node_pred: TensorBatch,
        start_points: TensorBatch | None,
        end_points: TensorBatch | None,
        overlap_cache: ClusterOverlapCache | None = None,
        labels: TensorBatch | None = None,
        valid_mask: TensorBatch | None = None,
        return_target: bool = False,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | TensorBatch | float | int]:
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
        overlap_cache : ClusterOverlapCache, optional
            Cluster overlaps shared by the objectives in one GrapPA forward.
        labels : TensorBatch, optional
            Cached orientation labels aligned with ``node_pred``. Must be
            supplied together with ``valid_mask``.
        valid_mask : TensorBatch, optional
            Cached one-dimensional node validity mask.
        return_target : bool, default False
            If `True`, return the exact labels and mask consumed by the loss.
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
        count_rejected : int, optional
            Number of otherwise valid track nodes removed by overlap quality.
            Only returned when thresholds are configured.
        """
        # Validate that cached labels and validity mask are provided together.
        if (labels is None) != (valid_mask is None):
            raise ValueError(
                "Cached labels and validity mask must be provided together."
            )

        # Reuse exact cached supervision or derive it from structured truth.
        count_rejected = 0
        if labels is not None:
            assert valid_mask is not None
            node_assn = labels
            valid_array = prepare_cached_target(labels, valid_mask, node_pred, "node")
        else:
            if any(
                value is None
                for value in (clust_label, coord_label, start_points, end_points)
            ):
                raise ValueError(
                    "Orientation loss requires either cached supervision or all "
                    "structured labels and endpoint inputs."
                )
            assert clust_label is not None
            assert coord_label is not None
            assert start_points is not None
            assert end_points is not None
            node_assn, valid_array, count_rejected = self._build_target(
                clust_label,
                coord_label,
                clusts,
                start_points,
                end_points,
                overlap_cache,
            )

        # From here on, cached and live targets follow the same loss path.
        if valid_mask is None:
            valid_mask = validity_batch(valid_array, node_assn)
        valid_index = np.where(valid_array)[0]
        node_assn_tensor = target_tensor(node_assn, node_pred, dtype=torch.long)[
            valid_index
        ]
        node_pred_tensor = node_pred.torch_tensor()[valid_index]
        loss = self.loss_fn(node_pred_tensor, node_assn_tensor, reduction="sum")
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
        # Expose the exact supervision consumed above for later caching.
        if return_target:
            result["target"] = node_assn
            result["valid"] = valid_mask

        return result

    def _build_target(
        self,
        clust_label: ClusterLabelBatch,
        coord_label: TensorBatch,
        clusts: IndexBatch,
        start_points: TensorBatch,
        end_points: TensorBatch,
        overlap_cache: ClusterOverlapCache | None,
    ) -> tuple[TensorBatch, np.ndarray, int]:
        """Build node-aligned orientation supervision from structured truth.

        Orientation is defined only for matched track clusters. This method
        maps batch-local particle IDs into the global coordinate-label index,
        compares truth and feature endpoint directions, and expands the binary
        result back onto the complete node axis before applying quality cuts.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Voxel-level particle associations and semantic shapes.
        coord_label : TensorBatch
            Truth start and end coordinates for particles in each batch item.
        clusts : IndexBatch
            Cluster-to-voxel index whose ordering defines the node axis.
        start_points : TensorBatch
            Predicted or reconstructed start point for each graph node.
        end_points : TensorBatch
            Predicted or reconstructed end point for each graph node.
        overlap_cache : ClusterOverlapCache, optional
            Precomputed cluster overlaps shared across GrapPA objectives.

        Returns
        -------
        TensorBatch
            Binary orientation labels aligned with all graph nodes; ineligible
            nodes carry the standard ``-1`` sentinel.
        np.ndarray
            Boolean mask selecting nodes eligible for the loss.
        int
            Number of otherwise eligible nodes rejected by overlap quality.
        """
        # Fetch the true particle associations and convert them to global IDs.
        part_ids = get_cluster_label_batch(clust_label, clusts, column="particle")
        part_ids_array = part_ids.numpy_tensor()
        global_part_ids = np.empty_like(part_ids_array, dtype=np.int64)
        for batch_id in range(part_ids.batch_size):
            shift_part_ids = part_ids[batch_id].copy()
            batch_valid_index = np.where(shift_part_ids > -1)[0]
            shift_part_ids[batch_valid_index] += int(coord_label.edges[batch_id])
            lower = part_ids.edges[batch_id]
            upper = part_ids.edges[batch_id + 1]
            global_part_ids[lower:upper] = shift_part_ids

        # Restrict the loss to matched track clusters.
        shapes = get_cluster_label_batch(clust_label, clusts, column="shape")
        valid_index = np.where(
            (global_part_ids > -1) & (shapes.numpy_tensor() == TRACK_SHP)
        )[0]

        # Compare true and feature endpoint directions for eligible clusters.
        index = global_part_ids[valid_index]
        true_starts = coord_label.coordinates("start").torch_tensor()[index]
        true_ends = coord_label.coordinates("end").torch_tensor()[index]
        true_dirs = true_ends - true_starts
        start_tensor = start_points.torch_tensor()[valid_index]
        end_tensor = end_points.torch_tensor()[valid_index]
        feat_dirs = end_tensor - start_tensor

        compact_assn = torch.sign(torch.sum(true_dirs * feat_dirs, dim=1)).long()
        compact_assn = (compact_assn + 1) // 2
        target_array = np.full(len(clusts.index_list), -1, dtype=np.int64)
        target_array[valid_index] = compact_assn.detach().cpu().numpy()
        node_assn = TensorBatch(target_array, clusts.counts)
        valid_array = target_array > -1

        # Class-dependent thresholds refer to the resulting orientation label.
        count_rejected = 0
        if self.quality_filter.active:
            quality_mask = self.quality_filter.node_mask(
                clust_label,
                clusts,
                target_array,
                overlap_cache,
            )
            count_rejected = int(np.count_nonzero(valid_array & ~quality_mask))
            valid_array &= quality_mask

        return node_assn, valid_array, count_rejected
