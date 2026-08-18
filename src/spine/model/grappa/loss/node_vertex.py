"""Module that defines a vertex identification loss using node predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from spine.cluster.label import get_cluster_label_batch
from spine.data import ClusterLabelBatch, IndexBatch, Meta, TensorBatch
from spine.geo import GeoManager
from spine.model.common.factories import loss_fn_factory
from spine.model.grappa.vertex import (
    decode_vertex_positions,
    vertex_position_scales,
)

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
        balance_primary_loss: bool = False,
        primary_loss: str | dict[str, Any] = "ce",
        regression_loss: str | dict[str, Any] = "mse",
        only_contained: bool = True,
        normalize_positions: bool = False,
        use_anchor_points: bool = False,
        return_vertex_labels: bool = False,
    ) -> None:
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
        self.geo = GeoManager.get_instance() if self.only_contained else None
        self.cont_def = None
        if self.geo is not None:
            self.cont_def = self.geo.define_containment_volumes(
                margin=0.0, mode="module"
            )

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        node_pred: TensorBatch,
        meta: Sequence[Meta] | None = None,
        start_points: TensorBatch | None = None,
        end_points: TensorBatch | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | TensorBatch | float | int]:
        """Applies the node type loss to a batch of data.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
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
        if node_pred.shape[1] != 5:
            raise ValueError(
                "The output used for vertex prediction should contain 5 "
                "features, 2 used for primary prediction and 3 for regression."
            )

        primary_pred, vertex_pred = torch.tensor_split(
            node_pred.torch_tensor(),
            [2],
            dim=1,
        )

        primary_pred = TensorBatch(primary_pred, node_pred.counts)
        vertex_pred = TensorBatch(vertex_pred, node_pred.counts)

        # Compute the primary identification loss
        result_primary = self.primary_loss(clust_label, clusts, primary_pred)

        # If containment or normalization are requested, ensure meta is provided
        if self.only_contained or self.normalize_positions:
            if meta is None:
                raise ValueError(
                    "Must provide `meta` to check containment or normalize "
                    "vertex positions."
                )

        # Get interaction-primary and three-dimensional vertex labels.
        primary_ids = get_cluster_label_batch(
            clust_label, clusts, column="interaction_primary"
        )
        vertex_labels = get_cluster_label_batch(clust_label, clusts, column="vertex")

        # Create a mask for valid nodes (-1 indicates invalid labels,
        # 0 indicates a secondary)
        valid_mask = primary_ids.numpy_tensor() > 0

        # If requested, check that the vertexes are contained
        if self.only_contained:
            assert meta is not None
            assert self.geo is not None
            assert self.cont_def is not None
            contain_mask = np.empty(len(clusts.index_list), dtype=bool)
            for batch_id in range(vertex_labels.batch_size):
                lower = vertex_labels.edges[batch_id]
                upper = vertex_labels.edges[batch_id + 1]
                points = meta[batch_id].to_cm(vertex_labels[batch_id])
                contain_mask[lower:upper] = self.geo.check_containment(
                    self.cont_def, points, summarize=False
                )

            valid_mask &= contain_mask

        # If requested, normalize the target positions to the detector size
        position_scales = None
        if self.normalize_positions:
            assert meta is not None
            position_scales = vertex_position_scales(vertex_pred, meta)
            vertex_labels = TensorBatch(
                vertex_labels.numpy_tensor() / position_scales.detach().cpu().numpy(),
                vertex_labels.counts,
            )

        # Apply the same position transformation used by full-chain inference.
        vertex_pred = decode_vertex_positions(
            vertex_pred,
            start_points=start_points,
            end_points=end_points,
            meta=meta,
            normalize_positions=self.normalize_positions,
            use_anchor_points=self.use_anchor_points,
            position_scales=position_scales,
        )

        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        vertex_assn = vertex_labels.to_tensor(device=node_pred.device)
        vertex_assn_tensor = vertex_assn.torch_tensor()[valid_index]
        vertex_pred_tensor = vertex_pred.torch_tensor()[valid_index]

        # Compute the regression loss
        reg_loss = self.reg_loss_fn(vertex_pred_tensor, vertex_assn_tensor)
        if len(valid_index) > 0:
            reg_loss /= len(valid_index)

        # Report mean Euclidean vertex error as the regression metric.
        reg_acc = 1.0
        if len(valid_index) > 0:
            dists = torch.norm(
                vertex_pred_tensor - vertex_assn_tensor,
                dim=1,
            )
            reg_acc = float(torch.mean(dists))

        # Build the result dictionary
        result = {
            "loss": (reg_loss + result_primary["loss"]) / 2,
            "accuracy": (reg_acc + result_primary["accuracy"]) / 2,
            "reg_accuracy": reg_acc,
            "reg_loss": reg_loss,
            "reg_count": len(valid_index),
            **{f"primary_{key}": value for key, value in result_primary.items()},
        }

        if self.return_vertex_labels:
            result["labels"] = vertex_labels

        return result
