"""Final prediction heads for batched dense features."""

from __future__ import annotations

from typing import Any

import torch

from spine.data import TensorBatch

from .evidential import EvidentialModel
from .mlp import MLP

__all__ = ["FinalLinear", "FinalMLP", "FinalEvidential"]


class FinalLinear(torch.nn.Module):
    """Apply a linear prediction head to batched dense features."""

    name = "linear"

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the linear prediction head.

        Parameters
        ----------
        in_channels : int
            Number of features coming from the upstream feature extractor
        out_channels : int
            Number of logits to output
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the linear layer
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, input_feats: TensorBatch) -> TensorBatch:
        """Passes a set of features through the final linear layer.

        Parameters
        ----------
        input_feats : TensorBatch
            (N, F) Batched tensor of input features

        Returns
        -------
        TensorBatch
            (N, F) Batched tensor of logits
        """
        x = self.linear(input_feats.torch_tensor())

        return TensorBatch(x, input_feats.counts)


class FinalMLP(torch.nn.Module):
    """Apply an MLP prediction head to batched dense features."""

    name = "mlp"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        positive_out: bool = False,
        **mlp: Any,
    ) -> None:
        """Initialize the MLP prediction head.

        Parameters
        ----------
        in_channels : int
            Number of features coming from the upstream feature extractor
        out_channels : int
            Number of logits to output
        positive_out : bool, default False
            If `True`, pass the output through a Softplus layer
        **mlp : dict
            MLP configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the MLP backbone
        self.mlp = MLP(in_channels, **mlp)

        # Initialize the final linear layer
        self.linear = torch.nn.Linear(self.mlp.feature_size, out_channels)

        # Initialize the softplus layer, if requested
        self.positive_output = positive_out
        if positive_out:
            self.softplus = torch.nn.Softplus()

    def forward(self, input_feats: TensorBatch) -> TensorBatch:
        """Passes a set of features through the final linear layer.

        Parameters
        ----------
        input_feats : TensorBatch
            (N, F) Batched tensor of input features

        Returns
        -------
        TensorBatch
            (N, F) Batched tensor of logits
        """
        x = self.mlp(input_feats.torch_tensor())
        x = self.linear(x)
        if self.positive_output:
            x = self.softplus(x)

        return TensorBatch(x, input_feats.counts)


class FinalEvidential(torch.nn.Module):
    """Predict evidential-regression parameters from batched dense features."""

    name = "evidential"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **evidential: Any,
    ) -> None:
        """Initialize the evidential prediction head.

        Parameters
        ----------
        in_channels : int
            Number of features coming from the upstream feature extractor
        out_channels : int
            Number of output parameters. Must be four for a
            normal-inverse-gamma distribution.
        **evidential : dict
            Evidential configuration

        Raises
        ------
        ValueError
            If ``out_channels`` is not four.
        """
        # Initialize the parent class
        super().__init__()

        if out_channels != 4:
            raise ValueError(
                "Evidential regression requires exactly four output channels."
            )

        # Initialize the evidential model
        self.evidential = EvidentialModel(in_channels, **evidential)

    def forward(self, input_feats: TensorBatch) -> TensorBatch:
        """Passes a set of features through the final linear layer.

        Parameters
        ----------
        input_feats : TensorBatch
            (N, F) Batched tensor of input features

        Returns
        -------
        TensorBatch
            (N, F) Batched tensor of logits
        """
        x = self.evidential(input_feats.torch_tensor())

        return TensorBatch(x, input_feats.counts)
