"""Module that defines a generic node classification loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
import torch

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.factories import loss_fn_factory
from spine.utils.gnn.cluster import (
    get_cluster_closest_label_batch,
    get_cluster_label_batch,
)
from spine.utils.weighting import get_class_weights

__all__ = ["NodeClassLoss"]


class NodeClassLoss(torch.nn.Module):
    """Generic loss used to train node identification.

    Takes the C-channel node output of the GNN and optimizes node-wise scores
    such that the score corresponding to the correct class is maximized.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: class
                <dictionary of arguments to pass to the loss>

    See configuration files prefixed with `grappa_` under the `config`
    directory for detailed examples of working configurations.
    """

    # Name of the loss (as specified in the configuration)
    name = "class"

    # Alternative allowed names of the loss
    aliases = ("classification",)

    def __init__(
        self,
        target: str,
        loss: str | dict[str, Any] = "ce",
        balance_loss: bool = False,
        weights: Sequence[float] | None = None,
        use_closest: bool = False,
        secondary_label: int | Sequence[int] = -1,
    ) -> None:
        """Initialize the node classification loss function.

        Parameters
        ----------
        target : str
            Column in the label tensor specifying the classification target
        loss : str, default 'ce'
            Name of the loss function to apply
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        weights : list, optional
            (C) One weight value per class
        use_closest : bool, default False
            For each particle group, assign the label class to the node which
            is closest to the particle start point only
        secondary_label : Union[int, List[int]], default -1
            When using `use_closest=True`, this label is assigned to nodes which
            are not the closest to a the start point of a particle group. These
            numbers can be different for each class if specified as a list
        """
        # Initialize the parent class
        super().__init__()

        # Parse the classification target
        self.target = target

        # Initialize basic parameters
        self.balance_loss = balance_loss
        self.weights = weights
        self.use_closest = use_closest
        self.secondary_label = secondary_label

        # Sanity check
        if weights is not None and balance_loss:
            raise ValueError(
                "Do not provide weights if they are to be computed on the fly."
            )

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(
        self,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        node_pred: TensorBatch,
        coord_label: TensorBatch | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | float | int]:
        """Applies the node classification loss to a batch of data.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        node_pred : TensorBatch
            (C, 2) Node prediction logits (binary output)
        coord_label : TensorBatch, optional
            (P, 1 + D + 8) Label start, end, time and shape for each point
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
        # Get the class labels
        node_assn = get_cluster_label_batch(clust_label, clusts, column=self.target)

        # If requested, adjust the class labeling of particle groups by picking
        # the node closest to the creation point as the reference target
        num_classes = node_pred.shape[1]
        if self.use_closest:
            # Make sure that the start point labeling is provided
            if coord_label is None:
                raise ValueError(
                    "To use the node closest to the particle creation point "
                    "as the reference node, must provide `coord_label`."
                )

            # Convert the default labels into a list of one value per class
            if isinstance(self.secondary_label, int):
                default = np.full(num_classes, self.secondary_label, dtype=int)
            else:
                if len(self.secondary_label) != num_classes:
                    raise ValueError(
                        "Must either provide a single default secondary label "
                        "or exactly one per label class."
                    )
                default = np.array(self.secondary_label, dtype=int)

            # Adjust the class labels
            node_assn = get_cluster_closest_label_batch(
                clust_label, coord_label, clusts, node_assn, default
            )

        # Create a mask for valid nodes (-1 indicates an invalid class ID)
        node_assn_array = node_assn.numpy_tensor()
        valid_mask = node_assn_array > -1

        # Check that the labels and the output tensor size are compatible
        class_mask = node_assn_array < num_classes
        if np.any(~class_mask):
            warn(
                "There are class labels with a value larger than the "
                f"size of the output logit vector ({num_classes}).",
                RuntimeWarning,
            )

        valid_mask &= class_mask

        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        node_assn = node_assn.to_tensor(dtype=torch.long, device=node_pred.device)
        node_assn_tensor = node_assn.torch_tensor()[valid_index]
        node_pred_tensor = node_pred.torch_tensor()[valid_index]

        # Compute the loss. Balance classes if requested
        weights = self.weights
        if self.balance_loss:
            weights = get_class_weights(node_assn_tensor, num_classes=num_classes)
        elif weights is not None:
            weights = torch.as_tensor(
                weights,
                dtype=node_pred_tensor.dtype,
                device=node_pred_tensor.device,
            )

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
        acc_class = [1.0] * num_classes
        if len(valid_index) > 0:
            preds = torch.argmax(node_pred_tensor, dim=1)
            acc = float(torch.sum(preds == node_assn_tensor))
            acc /= len(valid_index)
            for class_id in range(num_classes):
                index = torch.where(node_assn_tensor == class_id)[0]
                if len(index) > 0:
                    acc_class[class_id] = float(
                        torch.sum(preds[index] == class_id)
                    ) / len(index)

        # Prepare and return result
        result = {"loss": loss, "accuracy": acc, "count": len(valid_index)}

        for class_id in range(num_classes):
            result[f"accuracy_class_{class_id}"] = acc_class[class_id]

        return result
