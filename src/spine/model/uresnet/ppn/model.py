"""Module that defines a model and a loss to jointly train the semantic
segmentation task and the point proposal task."""

from __future__ import annotations

from typing import Any

import torch

from spine.data import ClusterLabelBatch, TensorBatch

from ...registry import ModelSpec
from ..model import SegmentationLoss, UResNetSegmentation
from .ppn import PPN, PPNLoss
from .vertex import VertexPPN, VertexPPNLoss

__all__ = ["UResNetPPN", "UResNetPPNLoss"]


class UResNetPPN(torch.nn.Module):
    """Combine UResNet with particle-point and/or vertex proposal tasks.

    Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet:
              # Your backbone uresnet config here
            ppn:  # Optional when only vertex regression is requested
              # Particle-point PPN configuration
            vertex:  # Optional
              # Interaction-vertex proposal configuration
            proposal_decoder:
              shared: false  # Share the decoder when both tasks are present

    See Also
    --------
    :class:`UResNetSegmentation`, :class:`PPN`, :class:`VertexPPN`
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn: dict[str, Any] | None = None,
        vertex: dict[str, Any] | None = None,
        proposal_decoder: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the UResNet+PPN model.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration dictionary
        ppn : dict, optional
            Particle-point PPN configuration.
        vertex : dict, optional
            Interaction-vertex proposal configuration.
        proposal_decoder : dict, optional
            Cross-task decoder options. ``shared`` defaults to ``False`` and
            only affects models that configure both proposal tasks.

        Raises
        ------
        ValueError
            If no proposal task is configured or decoder sharing is requested
            without both tasks.
        TypeError
            If the decoder block contains an unknown option.
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the UResNet backbone
        self.uresnet = UResNetSegmentation(uresnet)

        if ppn is None and vertex is None:
            raise ValueError("Configure at least one of `ppn` and `vertex`.")
        decoder_config = {} if proposal_decoder is None else dict(proposal_decoder)
        self.shared_proposal_decoder = bool(decoder_config.pop("shared", False))
        if decoder_config:
            unexpected = ", ".join(sorted(decoder_config))
            raise TypeError(f"Unexpected proposal-decoder option: {unexpected}.")
        if self.shared_proposal_decoder and (ppn is None or vertex is None):
            raise ValueError(
                "A shared proposal decoder requires both `ppn` and `vertex`."
            )

        # Existing PPN-only configurations retain the same `ppn` module path.
        self.ppn = None
        self.vertex = None
        if ppn is not None:
            shared_vertex = vertex if self.shared_proposal_decoder else None
            self.ppn = PPN(uresnet, ppn, vertex=shared_vertex)
        if vertex is not None and not self.shared_proposal_decoder:
            self.vertex = VertexPPN(uresnet, vertex)

        self.ghost = self.uresnet.ghost
        self.predicts_ppn = self.ppn is not None
        self.predicts_vertex = vertex is not None

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

        # Run each configured proposal path over the common UResNet pyramid.
        if self.ppn is not None:
            ghost = result.get("ghost_tensor") if self.ghost else None
            result.update(
                self.ppn(
                    result["final_tensor"],
                    result["decoder_tensors"],
                    ghost,
                    seg_label,
                )
            )
        if self.vertex is not None:
            ghost = result.get("ghost_tensor") if self.ghost else None
            result.update(
                self.vertex(
                    result["final_tensor"],
                    result["decoder_tensors"],
                    ghost,
                )
            )

        return result


class UResNetPPNLoss(torch.nn.Module):
    """Supervise UResNet and its configured point-proposal tasks.

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
        uresnet_loss: dict[str, Any],
        ppn: dict[str, Any] | None = None,
        ppn_loss: dict[str, Any] | None = None,
        vertex: dict[str, Any] | None = None,
        vertex_loss: dict[str, Any] | None = None,
        proposal_decoder: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the UResNet+PPN model loss.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration dictionary
        ppn : dict, optional
            PPN model configuration supplied as part of the manager's shared
            model/loss contract. The current loss infers optional heads from
            the model outputs.
        uresnet_loss : dict
            UResNet loss configuration
        ppn_loss : dict, optional
            PPN loss configuration.
        vertex : dict, optional
            Vertex model configuration supplied by the shared manager contract.
        vertex_loss : dict, optional
            Vertex loss configuration.
        proposal_decoder : dict, optional
            Decoder-sharing configuration supplied by the model contract.
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the segmentation loss
        self.seg_loss = SegmentationLoss(uresnet, uresnet_loss)

        del proposal_decoder
        if ppn is None and vertex is None:
            raise ValueError("Configure at least one proposal task loss.")
        if ppn is not None and ppn_loss is None:
            raise ValueError("A configured `ppn` task requires `ppn_loss`.")
        if vertex is not None and vertex_loss is None:
            raise ValueError("A configured `vertex` task requires `vertex_loss`.")
        self.ppn_loss = PPNLoss(uresnet, ppn_loss) if ppn_loss is not None else None
        self.vertex_loss = (
            VertexPPNLoss(uresnet, vertex_loss) if vertex_loss is not None else None
        )

    def forward(
        self,
        seg_label: TensorBatch,
        ppn_label: TensorBatch | None = None,
        vertex_label: TensorBatch | None = None,
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
        vertex_label : TensorBatch, optional
            Parsed interaction-vertex coordinates.
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

        task_results = [("uresnet", result_seg)]
        if self.ppn_loss is not None:
            if ppn_label is None:
                raise ValueError("PPN supervision requires `ppn_label`.")
            task_results.append(
                (
                    "ppn",
                    self.ppn_loss(
                        ppn_label,
                        clust_label=clust_label,
                        **result,
                    ),
                )
            )
        if self.vertex_loss is not None:
            if vertex_label is None:
                raise ValueError("Vertex supervision requires `vertex_label`.")
            task_results.append(("vertex", self.vertex_loss(vertex_label, **result)))

        # Every task contributes its native loss; reported accuracy is the
        # unweighted mean of the independently interpretable task metrics.
        output: dict[str, Any] = {
            "loss": sum(task["loss"] for _, task in task_results),
            "accuracy": sum(task["accuracy"] for _, task in task_results)
            / len(task_results),
        }
        for prefix, task_result in task_results:
            output.update(
                {f"{prefix}_{key}": value for key, value in task_result.items()}
            )
        return output


MODEL_SPEC = ModelSpec("uresnet_ppn", UResNetPPN, UResNetPPNLoss)
