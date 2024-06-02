"""Module that defines a generic node classification loss."""

import torch
import numpy as np
from warnings import warn

from spine.model.layers.factories import loss_fn_factory

from spine.utils.weighting import get_class_weights
from spine.utils.gnn.cluster import get_cluster_label_batch

__all__ = ['NodeClassificationLoss']


class NodeClassificationLoss(torch.nn.Module):
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
    name = 'class'
    aliases = ['classification']

    def __init__(self, target, balance_loss=False, loss='ce'):
        """Initialize the node classifcation loss function.

        Parameters
        ----------
        target : int
            Column in the label tensor specifying the classification target
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        loss : str, default 'ce'
            Name of the loss function to apply
        """
        # Initialize the parent class
        super().__init__()

        # Initialize basic parameters
        self.target = target
        self.balance_loss = balance_loss

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(self, clust_label, clusts, node_pred, **kwargs):
        """Applies the node classification  loss to a batch of data.

        Parameters
        ----------
        clust_label : TensorBatch
            (N, 1 + D + N_f) Tensor of cluster labels for the batch
        clusts : IndexBatch
            (C) Index which maps each cluster to a list of voxel IDs
        node_pred : TensorBatch
            (C, 2) Node prediction logits (binary output)
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
        node_assn = get_cluster_label_batch(
                clust_label, clusts, column=self.target)

        # Create a mask for valid nodes (-1 indicates an invalid class ID)
        valid_mask = node_assn.tensor > -1

        # Check that the labels and the output tensor size are compatible
        num_classes = node_pred.shape[1]
        class_mask = node_assn.tensor < num_classes
        if np.any(~class_mask):
            warn("There are class labels with a value larger than the "
                f"size of the output logit vector ({num_classes}).",
                RuntimeWarning)

        valid_mask &= class_mask

        # Apply the valid mask and convert the labels to a torch.Tensor
        valid_index = np.where(valid_mask)[0]
        node_assn = node_assn.to_tensor(
                dtype=torch.long, device=node_pred.device)
        node_assn = node_assn.tensor[valid_index]
        node_pred = node_pred.tensor[valid_index]

        # Compute the loss. Balance classes if requested
        weights = None
        if self.balance_loss:
            weights = get_class_weights(node_assn, num_classes=num_classes)

        loss = self.loss_fn(
                node_pred, node_assn, weight=weights, reduction='sum')
        if len(valid_index):
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned nodes)
        acc = 1.
        if len(valid_index):
            acc = float(torch.sum(torch.argmax(node_pred, dim=1) == node_assn))
            acc /= len(valid_index)

        return {
            'accuracy': acc,
            'loss': loss,
            'count': len(valid_index)
        }
