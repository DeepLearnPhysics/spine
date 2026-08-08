"""Module that defines a generic node classification loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.utils.gnn.cluster import get_cluster_label_batch

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
    ) -> None:
        """Initialize the node regression loss function.

        Parameters
        ----------
        target : str
            Column(s) in the label tensor specifying the regression target(s)
        loss : str, default 'mse'
            Name of the loss function to apply
        """
        # Initialize the parent class
        super().__init__()

        # Parse the regression target
        self.target = target

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, reduction="sum")

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        node_pred: TensorBatch,
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
        """
        # Get the regression labels
        node_assn = get_cluster_label_batch(clust_label, clusts, column=self.target)

        # Create a mask for valid nodes (-1 indicates an invalid label)
        valid_mask = node_assn.numpy_tensor() > -1

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

        return {"accuracy": acc, "loss": loss, "count": len(valid_index)}
