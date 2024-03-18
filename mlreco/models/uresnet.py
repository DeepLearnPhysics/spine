import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from collections import defaultdict

from mlreco.utils.globals import BATCH_COL, VALUE_COL, GHOST_SHP
from mlreco.utils.data_structures import TensorBatch
from mlreco.utils.logger import logger
from mlreco.utils.unwrap import Unwrapper

from .layers.cnn.act_norm import (
        activations_construct, normalizations_construct)
from .layers.cnn.uresnet_layers import UResNet

__all__ = ['UResNetSegmentation', 'SegmentationLoss']


class UResNetSegmentation(nn.Module):
    """UResNet for semantic segmentation.
    
    Typical configuration should look like:

    .. code-block:: yaml

        model:
          name: uresnet
          modules:
            uresnet:
              # Your config goes here

    See :func:`setup_cnn_configuration` for available parameters for the
    backbone UResNet architecture.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    num_classes: int, default 5
    ghost: bool, default False
    ghost_label: int, default -1
    balance_loss: bool, default False
        Whether to weight the loss using class counts.
    alpha: float, default 1.0
        Weight for UResNet semantic segmentation loss.
    beta: float, default 1.0
        Weight for ghost/non-ghost segmentation loss.

    Returns
    ------
    segmentation : torch.Tensor
    final_tensor : torch.Tensor
    encoder_tensors : list of torch.Tensor
    decoder_tensors : list of torch.Tensor
    ghost : torch.Tensor
    ghost_tensor : torch.Tensor

    See Also
    --------
    SegmentationLoss, mlreco.models.layers.common.uresnet_layers
    """

    INPUT_SCHEMA = [
        ['parse_sparse3d', (float,), (3, 1)]
    ]

    MODULES = ['uresnet']

    RETURNS = {
        'segmentation': Unwrapper.Rule(method='tensor'),
        'final_tensor': Unwrapper.Rule(method='tensor'),
        'encoder_tensors': Unwrapper.Rule(method='tensor_list'),
        'decoder_tensors': Unwrapper.Rule(method='tensor_list'),
        'ghost': Unwrapper.Rule(method='tensor'),
        'ghost_tensor': Unwrapper.Rule(method='tensor')
    }

    def __init__(self, uresnet, uresnet_loss=None):
        """Initializes the standalone UResNet model.

        Parameters
        ----------
        uresnet : dict
            Model configuration
        uresnet_loss : dict, optional
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the model configuration
        self.process_model_config(**uresnet)

        # Initialize the output layer
        self.output = [
            normalizations_construct(self.net.norm_cfg, self.num_filters),
            activations_construct(self.net.act_cfg),
            ]
        self.output = nn.Sequential(*self.output)
        self.linear_segmentation = ME.MinkowskiLinear(
                self.num_filters, self.num_classes)

        # If needed, activate the ghost classification layer
        if self.ghost:
            logger.debug('Ghost Masking is enabled for UResNet Segmentation')
            self.linear_ghost = ME.MinkowskiLinear(self.num_filters, 2)

    def process_model_config(self, num_classes, ghost=False, **backbone):
        """Initialize the underlying UResNet model.

        Parameters
        ----------
        num_classes : int
            Number of classes to classify the voxels as
        ghost : bool, default False
            Whether to add a deghosting step in the classification model
        **backbone : dict
            UResNet backbone configuration
        """
        # Store the semantic segmentation configuration
        self.num_classes = num_classes
        self.ghost = ghost

        # Initialize the UResNet backbone, store the relevant parameters
        self.net = UResNet(backbone) # TODO: Change this name, will break things
        self.num_filters = self.net.num_filters

    def forward(self, input_data):
        """Run a batch of data through the forward function.

        Parameters
        ----------
        input_data: TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        
        Returns
        -------
        dict
            Dictionary of outputs
        """
        # Pass the input data through the UResNet backbone
        result_backbone = self.net(input_data.tensor)

        # Pass the output features through the output layer
        feats = result_backbone['decoder_tensors'][-1]
        feats = self.output(feats)
        seg   = self.linear_segmentation(feats)

        # Store the output as tensor batches
        segmentation = TensorBatch(seg.F, input_data.counts)

        batch_size = input_data.batch_size
        final_tensor = TensorBatch(
                result_backbone['final_tensor'],
                batch_size=batch_size, is_sparse=True)
        encoder_tensors = [TensorBatch(
            t, batch_size=batch_size,
            is_sparse=True) for t in result_backbone['encoder_tensors']]
        decoder_tensors = [TensorBatch(
            t, batch_size=batch_size,
            is_sparse=True) for t in result_backbone['decoder_tensors']]

        result = {
            'segmentation': segmentation,
            'final_tensor': final_tensor,
            'encoder_tensors': encoder_tensors,
            'decoder_tensors': decoder_tensors
        }

        # If needed, pass the output features through the ghost linear layer
        if self.ghost:
            ghost = self.linear_ghost(feats)

            result['ghost'] = TensorBatch(ghost.F, input_data.counts)
            result['ghost_tensor'] = TensorBatch(
                    ghost, input_data.counts, is_sparse=True)

        return result


class SegmentationLoss(torch.nn.modules.loss._Loss):
    """Loss definition for semantic segmentation.

    For a regular flavor UResNet, it is a cross-entropy loss.
    For deghosting, it depends on a configuration parameter `ghost`:

    - If `ghost=True`, we first compute the cross-entropy loss on the ghost
      point classification (weighted on the fly with sample statistics). Then we
      compute a mask = all non-ghost points (based on true information in label)
      and within this mask, compute a cross-entropy loss for the rest of classes.

    - If `ghost=False`, we compute a N+1-classes cross-entropy loss, where N is
      the number of classes, not counting the ghost point class.

    See Also
    --------
    UResNet_Chain
    """
    INPUT_SCHEMA = [
        ['parse_sparse3d', (int,), (3, 1)]
    ]

    RETURNS = {
        'loss': Unwrapper.Rule(method='scalar'),
        'accuracy': Unwrapper.Rule(method='scalar'),
        'seg_loss': Unwrapper.Rule(method='scalar'),
        'seg_accuracy': Unwrapper.Rule(method='scalar'),
        'ghost_loss': Unwrapper.Rule(method='scalar'),
        'ghost_accuracy': Unwrapper.Rule(method='scalar'),
        'ghost_accuracy_class_0': Unwrapper.Rule(method='scalar'),
        'ghost_accuracy_class_1': Unwrapper.Rule(method='scalar')
    }

    def __init__(self, uresnet, uresnet_loss={}):
        """
        Initializes the segmentation loss

        Parameters
        ----------
        uresnet : dict
            Model configuration
        uresnet_loss : dict, optional
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize what we need from the model configuration
        self.process_model_config(**uresnet)

        # Initialize the loss configuration
        self.process_loss_config(**uresnet_loss)

        # Initialize the cross-entropy loss
        # TODO: Make it configurable
        self.xe = torch.nn.functional.cross_entropy

        # Append the expected output depending on how many classes there are
        for c in range(self.num_classes):
            self.RETURNS[f'accuracy_class_{c}'] = \
                    Unwrapper.Rule(method='scalar')

    def process_model_config(self, num_classes, ghost=False, **kwargs):
        """Process the parameters of the upstream model needed for in the loss.

        Parameters
        ----------
        num_classes : int
            Number of classes to classify the voxels as
        ghost : bool, default False
            Whether to add a deghosting step in the classification model
        **kwargs : dict, optional
            Leftover model configuration (no need in the loss)
        """
        # Store the semantic segmentation configuration
        self.num_classes = num_classes
        self.ghost = ghost

    def process_loss_config(self, ghost_label=-1, alpha=1.0, beta=1.0,
                            balance_loss=False):
        """Process the loss function parameters.

        Parameters
        ----------
        ghost_label : int, default -1
            ID of ghost points. If specified (> -1), classify ghosts only
        alpha : float, default 1.0
            Classification loss prefactor
        beta : float, default 1.0
            Ghost mask loss prefactor
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        """
        # Store the loss configuration
        self.ghost_label  = ghost_label
        self.alpha        = alpha
        self.beta         = beta
        self.balance_loss = balance_loss

        # If a ghost label is provided, it cannot be in conjecture with
        # having a dedicated ghost masking layer
        assert not (self.ghost and self.ghost_label > -1), \
                'Cannot classify ghost exclusively (ghost_label) and ' \
                'have a dedicated ghost masking layer at the same time'

    def forward(self, segment_label, segmentation, ghost=None, 
                weights=None, **kwargs):
        """Computes the cross-entropy loss of the semantic segmentation
        predictions. 

        Parameters
        ----------
        segment_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels for the batch
        segmentation : TensorBatch
            (N, N_c) Tensor of logits from the segmentation model
        ghost : TensorBatch
            (N, 2) Tensor of ghost logits from the segmentation model
        weights : torch.Tensor, optional
            (N) Tensor of weights for each pixel in the batch
        **kwargs : dict, optional
            Other outputs of the upstream model which are not relevant here

        Returns
        -------
        dict
            Dictionary of accuracies and losses
        """
        # Get the underlying tensor in each TensorBatch
        segment_label = segment_label.tensor
        segmentation = segmentation.tensor

        # Make sure that the segmentation output and labels have the same length
        assert len(segment_label) == len(segmentation), \
                'The segmentation output length and its labels do not match'
        assert not self.ghost or len(segment_label) == len(ghost), \
                'The ghost output length and its labels do not match'

        # If the loss is to be class-weighted, cannot also provide weights
        assert not self.balance_loss or weights is None, \
                'If weights are provided, cannot class-weight loss'

        # Check that the labels have sensible values
        if self.ghost_label > -1:
            labels = (segment_label[:, VALUE_COL] == self.ghost_label).long()
        else:
            labels = segment_label[:, VALUE_COL].long()
            if torch.any(labels > self.num_classes):
                raise ValueError('The segmentation labels do not match ' \
                        'the number of classes output by the model')

        # If there is a dedicated ghost layer, apply the mask first
        if self.ghost:
            # Count the number of voxels in each class
            ghost_labels = (labels == GHOST_SHP).long()
            ghost_loss, ghost_acc, ghost_acc_class = self.loss_accuracy(
                    ghost, ghost_labels)

            # Restrict the segmentation target to true non-ghosts
            nonghost = torch.nonzero(ghost_labels == 0).flatten()
            segmentation = segmentation[nonghost]
            labels = labels[nonghost]

        # Compute the loss/accuracy of the semantic segmentation step
        seg_loss, seg_acc, seg_acc_class = self.loss_accuracy(
                segmentation, labels, weights)

        # Get the combined loss and accuracies
        result = {}
        if self.ghost:
            result.update({
                'loss': self.alpha * seg_loss + self.beta * ghost_loss,
                'accuracy': (seg_acc + ghost_acc)/2.,
                'seg_loss': seg_loss,
                'seg_accuracy': seg_acc,
                'ghost_loss': ghost_loss,
                'ghost_accuracy': ghost_acc,
                'ghost_accuracy_class_0': ghost_acc_class[0],
                'ghost_accuracy_class_1': ghost_acc_class[1]})

            for c in range(self.num_classes):
                result[f'seg_accuracy_class_{c}'] = seg_acc_class[c]

        else:
            result.update({
                'loss': seg_loss,
                'accuracy': seg_acc})

            for c in range(self.num_classes):
                result[f'accuracy_class_{c}'] = seg_acc_class[c]

        return result

    def loss_accuracy(self, logits, labels, weights=None):
        """Computes the loss, global and classwise accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            (N, N_c) Output logits from the network for each voxel
        labels : torch.Tensor
            (N) Target values for each voxel
        weights : torch.Tensor, optional
            (N) Tensor of weights for each pixel in the batch

        Returns
        -------
        torch.Tensor
            Cross-entropy loss value
        float
            Global accuracy
        np.ndarray
            (N_c) Vector of class-wise accuracy
        """
        # If there is no input, nothing to do
        if not len(logits):
            return 0., 1., np.ones(num_classes, dtype=np.float32)

        # Count the number of voxels in each class
        num_classes = logits.shape[1]
        counts = torch.empty(num_classes,
                dtype=torch.long, device=labels.device)
        for c in range(num_classes):
            counts[c] = torch.sum(labels == c).item()

        # Compute the loss
        if self.balance_loss and torch.all(counts):
            class_weight = len(labels)/num_classes/counts
            loss = self.xe(logits, labels, weight=class_weight)
        else:
            if weights is None:
                loss = self.xe(logits, labels, reduction='mean')
            else:
                loss = (weights*self.xe(logits, labels, reduction='none')).sum()

        # Compute the accuracies
        with torch.no_grad():
            # Per-class prediction accuracy
            preds = torch.argmax(logits, dim=-1)
            acc_class = np.ones(num_classes, dtype=np.float32)
            for c in range(num_classes):
                if counts[c] > 0:
                    mask = torch.nonzero(labels == c).flatten()
                    acc_class[c] = (preds[mask] == c).sum().item() / counts[c]

            # Global prediction accuracy
            acc = (preds == labels).sum().item() / torch.sum(counts).item()

        return loss, acc, acc_class
