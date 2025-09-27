"""Module that defines a generic node classification loss."""

from warnings import warn

import numpy as np
import torch

from spine.model.layer.factories import loss_fn_factory
from spine.utils.enums import enum_factory
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

    def __init__(self, target, loss="mse"):
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
        self.target = enum_factory("cluster", target)

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, reduction="sum")

    def forward(self, clust_label, clusts, node_pred, **kwargs):
        """Applies the node regression loss to a batch of data.

        Parameters
        ----------
        clust_label : TensorBatch
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
        valid_mask = node_assn.tensor > -1

        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        node_assn = node_assn.to_tensor(device=node_pred.device)
        node_assn = node_assn.tensor[valid_index]
        node_pred = node_pred.tensor[valid_index]

        # Compute the loss
        loss = self.loss_fn(node_pred, node_assn)
        if len(valid_index):
            loss /= len(valid_index)

        # Compute accuracy of assignment (average relative resolution)
        # TODO: Come up with a better implementation (between 0 and 1?)
        acc = 1.0
        if len(valid_index):
            rel_res = (node_pred.view(-1) - node_assn) / node_assn
            acc = float(torch.std(rel_res))

        return {"accuracy": acc, "loss": loss, "count": len(valid_index)}
