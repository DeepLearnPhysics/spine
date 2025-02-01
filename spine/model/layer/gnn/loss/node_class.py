"""Module that defines a generic node classification loss."""

import torch
import numpy as np
from warnings import warn

from spine.model.layer.factories import loss_fn_factory

from spine.utils.enums import enum_factory
from spine.utils.weighting import get_class_weights
from spine.utils.gnn.cluster import get_cluster_label_batch

__all__ = ['NodeClassLoss']


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
    name = 'class'

    # Alternative allowed names of the loss
    aliases = ('classification',)

    def __init__(self, target, loss='ce', balance_loss=False, weights=None):
        """Initialize the node classifcation loss function.

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
        """
        # Initialize the parent class
        super().__init__()

        # Parse the classification target
        self.target = enum_factory('cluster', target)

        # Initialize basic parameters
        self.balance_loss = balance_loss
        self.weights = weights

        # Sanity check
        assert weights is None or not balance_loss, (
                "Do not provide weights if they are to be computed on the fly.")

        # Set the loss
        self.loss_fn = loss_fn_factory(loss, functional=True)

    def forward(self, clust_label, clusts, node_pred, **kwargs):
        """Applies the node classification loss to a batch of data.

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
        if self.balance_loss:
            self.weights = get_class_weights(node_assn, num_classes=num_classes)

        loss = self.loss_fn(
                node_pred, node_assn, weight=self.weights, reduction='sum')
        if len(valid_index):
            loss /= len(valid_index)

        # Compute accuracy of assignment (fraction of correctly assigned nodes)
        acc = 1.
        acc_class = [1.] * num_classes
        if len(valid_index):
            preds = torch.argmax(node_pred, dim=1)
            acc = float(torch.sum(preds == node_assn))
            acc /= len(valid_index)
            for c in range(num_classes):
                index = torch.where(node_assn == c)[0]
                if len(index):
                    acc_class[c] = float(torch.sum(preds[index] == c))/len(index)

        # Prepare and return result
        result = {
            'loss': loss,
            'accuracy': acc,
            'count': len(valid_index)
        }

        for c in range(num_classes):
            result[f'accuracy_class_{c}'] = acc_class[c]

        return result
