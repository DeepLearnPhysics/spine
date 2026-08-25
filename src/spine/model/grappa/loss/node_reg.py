"""Module that defines a generic node classification loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from spine.cluster.label import get_cluster_label_batch
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.model.common.quality import (
    ClusterOverlapCache,
    ClusterQualityFilter,
)

__all__ = ["NodeRegressionLoss"]


class NodeRegressionLoss(torch.nn.Module):
    """Generic loss used to train node regression.

    Takes the C-channel node output of the GNN and optimizes node-wise values
    such that it matches the label values as closely as possible.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: reg
                <dictionary of arguments to pass to the loss>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    # Name of the loss (as specified in the configuration)
    name = "reg"

    # Alternative allowed names of the loss
    aliases = ("regression",)

    def __init__(
        self,
        target: str,
        loss: str | dict[str, Any] = "mse",
        *,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str = "group",
        quality_target: str = "pid",
        quality_num_classes: int | None = None,
    ) -> None:
        """Initialize the node regression loss function.

        Parameters
        ----------
        target : str
            Column(s) in the label tensor specifying the regression target(s)
        loss : str, default 'mse'
            Name of the loss function to apply
        min_iou : float or sequence of float, optional
            Minimum truth-instance IoU, shared or selected from the class given
            by ``quality_target``.
        min_purity : float or sequence of float, optional
            Minimum predicted-cluster purity, shared or selected from the class
            given by ``quality_target``.
        min_efficiency : float or sequence of float, optional
            Minimum truth-instance efficiency, shared or selected from the
            class given by ``quality_target``.
        match_target : str, default 'group'
            Truth-instance field used to evaluate overlap quality.
        quality_target : str, default 'pid'
            Categorical field used to select class-dependent thresholds.
        quality_num_classes : int, optional
            Number of values represented by ``quality_target``. Required when
            any quality threshold is a sequence.
        """
        # Initialize the parent class
        super().__init__()

        # Parse the regression target
        self.target = target
        self.quality_target = quality_target

        # Regression width is unrelated to the categorical quality policy, so
        # class-dependent thresholds use an explicit target and class count.
        self.quality_filter = ClusterQualityFilter(
            min_iou,
            min_purity,
            min_efficiency,
            match_target=match_target,
            num_classes=quality_num_classes,
            require_num_classes=True,
        )

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, reduction="sum")

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        node_pred: TensorBatch,
        overlap_cache: ClusterOverlapCache | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | float | int]:
        """Applies the node regression loss to a batch of data.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        node_pred : TensorBatch
            (C, N_d) Node prediction
        overlap_cache : ClusterOverlapCache, optional
            Cluster overlaps shared by the objectives in one GrapPA forward.
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
            Number of otherwise valid nodes removed by overlap-quality
            filtering. Only returned when thresholds are configured.
        """
        # Get the regression labels
        node_assn = get_cluster_label_batch(clust_label, clusts, column=self.target)

        # Vector targets are eligible only when every component is available.
        node_assn_array = node_assn.numpy_tensor()
        base_valid_mask = node_assn_array > -1
        if base_valid_mask.ndim > 1:
            base_valid_mask = np.all(base_valid_mask, axis=1)
        valid_mask = base_valid_mask.copy()
        count_rejected = 0

        # Reject regression targets attached to poor truth-instance matches.
        if self.quality_filter.active:
            classes = None
            if self.quality_filter.class_dependent:
                # Reduce the categorical quality field on the same clusters as
                # the continuous regression target.
                classes = get_cluster_label_batch(
                    clust_label,
                    clusts,
                    column=self.quality_target,
                ).numpy_tensor()
            quality_mask = self.quality_filter.node_mask(
                clust_label,
                clusts,
                classes,
                overlap_cache,
            )
            count_rejected = int(np.count_nonzero(base_valid_mask & ~quality_mask))
            valid_mask &= quality_mask

            # Mark rejected targets with the ordinary invalid-label sentinel.
            node_assn_array = node_assn.numpy_tensor().copy()
            node_assn_array[~quality_mask] = -1
            node_assn = TensorBatch(node_assn_array, node_assn.counts)

        # Combine ordinary target validity with the optional quality policy.
        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        node_assn = node_assn.to_tensor(device=node_pred.device)
        node_assn_tensor = node_assn.torch_tensor()[valid_index]
        node_pred_tensor = node_pred.torch_tensor()[valid_index]

        # Scalar labels are stored as ``(N,)`` while model predictions use
        # ``(N, 1)``. Align equivalent shapes explicitly so that PyTorch does
        # not broadcast the two node axes into an ``(N, N)`` loss matrix.
        if node_assn_tensor.shape != node_pred_tensor.shape:
            if node_assn_tensor.numel() != node_pred_tensor.numel():
                raise ValueError(
                    "Node regression labels and predictions contain "
                    "incompatible numbers of values: "
                    f"{tuple(node_assn_tensor.shape)} and "
                    f"{tuple(node_pred_tensor.shape)}."
                )
            node_assn_tensor = node_assn_tensor.view_as(node_pred_tensor)

        # Compute the loss
        loss = self.loss_fn(node_pred_tensor, node_assn_tensor)
        if len(valid_index) > 0:
            loss /= len(valid_index)

        # Report the spread of the fractional residual as the regression
        # metric. Clamp zero-valued targets to keep the metric finite.
        acc = 1.0
        if len(valid_index) > 0:
            denominator = torch.clamp(torch.abs(node_assn_tensor), min=1e-12)
            rel_res = (
                node_pred_tensor.view_as(node_assn_tensor) - node_assn_tensor
            ) / denominator
            acc = float(torch.std(rel_res, correction=0))

        result = {"accuracy": acc, "loss": loss, "count": len(valid_index)}
        if self.quality_filter.active:
            result["count_rejected"] = count_rejected

        return result
