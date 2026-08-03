"""Module that defines a model and a loss to jointly train the semantic
segmentation task and the point proposal task."""

from __future__ import annotations

from typing import Any

import torch

from spine.data import ClusterLabelBatch, TensorBatch

from ...registry import ModelSpec
from ..model import SegmentationLoss, UResNetSegmentation
from .ppn import PPN, PPNLoss

__all__ = ["UResNetPPN", "UResNetPPNLoss"]


class UResNetPPN(torch.nn.Module):
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

    See Also
    --------
    :class:`UResNetSegmentation`, :class:`PPN`
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn: dict[str, Any],
    ) -> None:
        """Initialize the UResNet+PPN model.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration dictionary
        ppn : dict
            PPN configuration dictionary
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the UResNet backbone
        self.uresnet = UResNetSegmentation(uresnet)

        # Initialize the PPN layers
        self.ppn = PPN(uresnet, ppn)

        # Check that the UResNet and PPN configurations are compatible
        if self.uresnet.ghost != self.ppn.ghost:
            raise ValueError("Expected `self.uresnet.ghost == self.ppn.ghost`.")
        self.ghost = self.uresnet.ghost

    def forward(
        self, data: TensorBatch, seg_label: TensorBatch | None = None
    ) -> dict[str, Any]:
        """Run a batch of data through the foward function.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        seg_label : TensorBatch, optional
            (N, 1 + D + 1) tensor of voxel/ghost label pairs
        """
        # Pass the input through the backbone UResNet model
        result = self.uresnet(data)

        # Pass the decoder feature tensors to the PPN layers
        if self.ghost:
            # Deghost
            if self.ppn.use_true_ghost_mask:
                # Use the true ghost labels
                if seg_label is None:
                    raise ValueError(
                        "If `use_true_ghost_mask` is set to `True`, must "
                        "provide the `seg_label` tensor."
                    )
                result_ppn = self.ppn(
                    result["final_tensor"],
                    result["decoder_tensors"],
                    result["ghost_tensor"],
                    seg_label,
                )
            else:
                # Use the ghost predictions
                result_ppn = self.ppn(
                    result["final_tensor"],
                    result["decoder_tensors"],
                    result["ghost_tensor"],
                )
        else:
            # No ghost
            result_ppn = self.ppn(result["final_tensor"], result["decoder_tensors"])

        result.update(result_ppn)

        return result


class UResNetPPNLoss(torch.nn.Module):
    """Loss for a model made of a UResNet backbone and PPN layers.

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
    :class:`spine.model.uresnet.SegmentationLoss`,
    :class:`spine.model.uresnet.ppn.ppn.PPNLoss`
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn: dict[str, Any],
        uresnet_loss: dict[str, Any],
        ppn_loss: dict[str, Any],
    ) -> None:
        """Initialize the UResNet+PPN model loss.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration dictionary
        ppn : dict
            PPN model configuration supplied as part of the manager's shared
            model/loss contract. The current loss infers optional heads from
            the model outputs.
        uresnet_loss : dict
            UResNet loss configuration
        ppn_loss : dict
            PPN loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the segmentation loss
        self.seg_loss = SegmentationLoss(uresnet, uresnet_loss)

        # Initialize the point proposal loss
        self.ppn_loss = PPNLoss(uresnet, ppn_loss)

    def forward(
        self,
        seg_label: TensorBatch,
        ppn_label: TensorBatch,
        clust_label: ClusterLabelBatch | None = None,
        weights: TensorBatch | None = None,
        **result: Any,
    ) -> dict[str, Any]:
        """Compute the combined segmentation and point-proposal loss.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels for the batch
        ppn_label : TensorBatch
            (N, 1 + D + N_l) Tensor of PPN labels for the batch
        clust_label : ClusterLabelBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        weights : torch.Tensor, optional
            (N) Tensor of segmentation weights for each pixel in the batch
        **result : dict
            Outputs of the UResNet + PPN forward function

        Returns
        -------
        dict
            Combined loss and accuracy together with prefixed segmentation
            and point-proposal component metrics.
        """
        # Apply the segmentation loss
        result_seg = self.seg_loss(seg_label, weights=weights, **result)

        # Apply the PPN loss
        result_ppn = self.ppn_loss(ppn_label, clust_label=clust_label, **result)

        # Combine the two outputs
        result = {
            "loss": result_seg["loss"] + result_ppn["loss"],
            "accuracy": (result_seg["accuracy"] + result_ppn["accuracy"]) / 2,
        }

        result.update({"uresnet_" + k: v for k, v in result_seg.items()})
        result.update({"ppn_" + k: v for k, v in result_ppn.items()})

        return result


MODEL_SPEC = ModelSpec("uresnet_ppn", UResNetPPN, UResNetPPNLoss)
