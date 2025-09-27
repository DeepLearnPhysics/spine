"""Defines a layer that converts logit output into an evidential output."""

import torch
from torch import nn

from .mlp import MLP


class EvidentialModel(nn.Module):
    """Model which produces evidential predictions with an MLP backbone."""

    def __init__(self, in_channels, mlp, eps=0.0, logspace=False):
        """Initialize the evidential network.

        Parameters
        ----------
        in_channels : int
            Number of features from the upstream feature extractor
        mlp : dict
            MLP configuration dictionary
        eps : float, default 0.0
            Offset to apply to the softplus output
        logspace : bool, default False
            Whether to take the sigmoid of gamma, or not
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the MLP backbone
        self.mlp = MLP(in_channels, **mlp)

        # Initialize the final linear layer
        # TODO: understand why 4, this probably makes sense
        self.linear = nn.Linear(mlp.feature_size, 4)

        # Initialize the output layer parameters
        self.eps = eps
        self.softplus = nn.Softplus
        self.logspace = logspace
        self.gamma = nn.Sigmoid() if logspace else nn.Identity()

    def forward(self, input_feats):
        """Passes a set of features through the evidential model.

        Parameters
        ----------
        input_feats : torch.Tesnor
            (N, F) Tensor of input features

        Results
        -------
        torch.Tensor
            (N, F) Tensor of evidence
        """
        # Pass data through the MLP and the linear layer
        x = self.mlp(input_feats)
        x = self.linear(x)

        # Convert the output to an evidence
        # TODO: Would be nice to understand this
        vab = self.softplus(x[:, :3]) + self.eps
        alpha = torch.clamp(vab[:, 1] + 1.0, min=1.0).view(-1, 1)
        gamma = 2.0 * self.gamma(x[:, 3]).view(-1, 1)
        out = torch.cat(
            [gamma, vab[:, 0].view(-1, 1), alpha, vab[:, 2].view(-1, 1)], dim=1
        )
        if not self.logspace:
            evidence = torch.clamp(out, min=self.eps)
        else:
            evidence = out

        return evidence
