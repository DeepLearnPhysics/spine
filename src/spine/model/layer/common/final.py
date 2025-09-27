"""Defines models that take feature vector developped by a dedicated
feature extractor networks and produces the required type of output.
"""

from torch import nn

from spine.data import TensorBatch

from .evidential import EvidentialModel
from .mlp import MLP

__all__ = ["FinalLinear", "FinalMLP", "FinalEvidential"]


class FinalLinear(nn.Module):
    """Simple wrapper class for a final linear layer operation."""

    name = "linear"

    def __init__(self, in_channels, out_channels):
        """Initializes the linear layer.

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
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, input_feats):
        """Passes a set of features through the final linear layer.

        Parameters
        ----------
        input_feats : TensorBatch
            (N, F) Batched tensor of input features

        Results
        -------
        TensorBatch
            (N, F) Batched tensor of logits
        """
        x = self.linear(input_feats.tensor)

        return TensorBatch(x, input_feats.counts)


class FinalMLP(nn.Module):
    """Simple wrapper class for a final MLP model."""

    name = "mlp"

    def __init__(self, in_channels, out_channels, positive_out=False, **mlp):
        """Initializes the linear layer.

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
        self.linear = nn.Linear(self.mlp.feature_size, out_channels)

        # Initialize the softplus layer, if requested
        self.positive_output = positive_output
        if positive_output:
            self.softplus = nn.Softplus()

    def forward(self, input_feats):
        """Passes a set of features through the final linear layer.

        Parameters
        ----------
        input_feats : TensorBatch
            (N, F) Batched tensor of input features

        Results
        -------
        TensorBatch
            (N, F) Batched tensor of logits
        """
        x = self.mlp(input_feats.tensor)
        x = self.linear(x)
        if self.positive_output:
            x = self.softplus(x)

        return TensorBatch(x, input_feats.counts)


class FinalEvidential(nn.Module):
    """Simple wrapper class for a final Evidential model."""

    name = "evidential"

    def __init__(self, in_channels, out_channels, **evidential):
        """Initializes the linear layer.

        Parameters
        ----------
        in_channels : int
            Number of features coming from the upstream feature extractor
        out_channels : int
            Number of logits to output (always 4, ignores this argument)
        **evidential : dict
            Evidential configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the evidential model
        self.evidential = EvidentialModel(in_channels, **evidential)

    def forward(self, input_feats):
        """Passes a set of features through the final linear layer.

        Parameters
        ----------
        input_feats : TensorBatch
            (N, F) Batched tensor of input features

        Results
        -------
        TensorBatch
            (N, F) Batched tensor of logits
        """
        x = self.evidential(input_feats.tensor)

        return TensorBatch(x, input_feats.counts)
