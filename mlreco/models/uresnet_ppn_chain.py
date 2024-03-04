import numpy as np
import torch
import torch.nn as nn
import time
from collections import defaultdict

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from .uresnet import UResNetSegmentation, SegmentationLoss
from .layers.common.ppnplus import PPN, PPNLoss

from mlreco.utils.unwrap import Unwrapper


class UResNetPPN(nn.Module):
    """A model made of a UResNet backbone and PPN layers.
    
    Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet:
              # Your backbone uresnet config here
            ppn:
              # Your ppn config here

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

    depth: int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters: int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps: int, default 2
        Convolution block repetition factor
    input_kernel: int, default 3
        Receptive field size for very first convolution after input layer.

    score_threshold: float, default 0.5
    classify_endpoints: bool, default False
        Enable classification of points into start vs end points.
    ppn_resolution: float, default 1.0
    use_true_ghost_mask: bool, default False

    See Also
    --------
    mlreco.models.uresnet.UResNetSegmentation, mlreco.models.layers.common.ppnplus.PPN
    """
    MODULES = ['uresnet', 'ppn']

    RETURNS = dict(UResNetSegmentation.RETURNS, **PPN.RETURNS)

    def __init__(self, uresnet, ppn, uresnet_loss=None, ppn_loss=None):
        """Initialize the UResNet+PPN model.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration dictionary
        ppn : dict
            PPN configuration dictionary
        uresnet_loss : dict, optional
            UResNet loss configuration
        ppn_loss : dict, optional
            PPN loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the UResNet backbone
        self.uresnet = UResNetSegmentation(uresnet)

        # Initialize the PPN layers
        self.ppn = PPN(uresnet, ppn)
        
        # Check that the UResNet and PPN configurations are compatible
        assert self.uresnet.ghost == self.ppn.ghost
        self.ghost = self.uresnet.ghost

    def forward(self, input_data, segment_label=None):
        """Run a batch of data through the foward function.

        Parameters
        ----------
        input_data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        segment_label : TensorBatch, optional
            (N, 1 + D + 1) tensor of voxel/ghost label pairs
        """
        # Pass the input through the backbone UResNet model
        result = self.uresnet(input_data)

        # Pass the decoder feature tensors to the PPN layers
        if self.ghost:
            # Deghost
            if sefl.ppn.use_true_ghost_mask:
                # Use the true ghost labels
                assert segment_label is not None, (
                        "If `use_true_ghost_mask` is set to `True`, must "
                        "provide the `segment_label` tensor.")
                result_ppn = self.ppn(
                        result['final_tensor'], result['decoder_tensors'],
                        result['ghost_tensor'], segment_label)
            else:
                # Use the ghost predictions
                result_ppn = self.ppn(
                        result['final_tensor'], result['decoder_tensors'],
                        result['ghost_tensor'])
        else:
            # No ghost
            result_ppn = self.ppn(
                    result['final_tensor'], result['decoder_tensors'])

        result.update(result_ppn)

        return result


class UResNetPPNLoss(nn.Module):
    """Loss for amodel made of a UResNet backbone and PPN layers.

    It includes a segmentation loss and a PPN loss.

    Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet:
              # Your backbone uresnet config goes here
            ppn:
              # Your ppn config goes here
            ppn_loss:
              # Your ppn loss config goes here

    See Also
    --------
    :class:`mlreco.models.uresnet.SegmentationLoss`,
    :class:`mlreco.models.layers.common.ppnplus.PPNLoss`
    """
    RETURNS = {
        'loss': Unwrapper.Rule(method='scalar'),
        'accuracy': Unwrapper.Rule(method='scalar')
    }

    def __init__(self, uresnet, ppn, ppn_loss, uresnet_loss=None):
        """Initialize the UResNet+PPN model loss.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration dictionary
        ppn : dict
            PPN configuration dictionary
        uresnet_loss : dict, optional
            UResNet loss configuration
        ppn_loss : dict, optional
            PPN loss configuration
        """
        # Initialize the parent clas
        super().__init__()

        # Initialize the segmentation loss
        self.seg_loss = SegmentationLoss(uresnet, uresnet_loss)
        seg_rules = self.seg_loss.RETURNS

        # Initialize the point proposal loss
        self.ppn_loss = PPNLoss(uresnet, ppn, ppn_loss)
        ppn_rules = self.ppn_loss.RETURNS

        # Add unwrapping rules for each submodel output
        self.RETURNS.update({'uresnet_'+k:v for k, v in seg_rules.items()})
        self.RETURNS.update({'ppn_'+k:v for k, v in ppn_rules.items()})

    def forward(self, segment_label, ppn_label, weights=None, **result):
        """Run a batch of data through the loss function.

        Parameters
        ----------
        segment_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels for the batch
        ppn_label : TensorBatch
            (N, 1 + D + N_l) Tensor of PPN labels for the batch
        weights : torch.Tensor, optional
            (N) Tensor of segmentation weights for each pixel in the batch
        **result : dict
            Outputs of the UResNet + PPN forward function
        """
        # Apply the segmentation loss
        result_seg = self.seg_loss(segment_label, weights=weights, **result)

        # Apply the PPN loss
        result_ppn = self.ppn_loss(ppn_label, **result)

        # Combine the two outputs
        result = {
            'loss': result_seg['loss'] + result_ppn['loss'],
            'accuracy': (result_seg['accuracy'] + result_ppn['accuracy'])/2
        }

        result.update({'uresnet_'+k:v for k, v in result_seg.items()})
        result.update({'ppn_'+k:v for k, v in result_ppn.items()})

        return result
