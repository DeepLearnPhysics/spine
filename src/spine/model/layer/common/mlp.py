"""Module which defines a very generic multi-layer perceptron with
fully configurable parameters to be used elsewhere.
"""

from typing import List, Union

import numpy as np
import torch
from torch import nn

from .act_norm import act_factory, norm_factory

__all__ = ["MLP"]


class MLP(nn.Module):
    """Generic multi-layer perceptron to be used as a feature extractor."""

    name = "mlp"

    def __init__(self, in_channels, depth, width, activation, normalization):
        """Initialize the MLP.

        Parameters
        ----------
        in_channels : int
            Number of input features
        depth : int
            Number of hidden layers
        width : Union[int, List[int]]
            Number of neurons in each hidden layer
        activation : Union[str, dict]
            Activation function configuration
        normalization : Union[str, dict]
            Normalization function configuration
        """
        # Initialize the parent class
        super().__init__()

        # Store the attribtues
        self.in_channels = in_channels
        self.depth = depth
        self.act_cfg = activation
        self.norm_cfg = normalization

        # Process the width
        if isinstance(width, int):
            self.width = [width] * depth
        else:
            assert len(width) == depth, (
                "If provided as an array, the `width` must be given "
                "once for each hidden layer (specified in `depth`)"
            )
            self.width = width

        self.feature_size = self.width[-1]

        # Initialize the model
        self.model = nn.Sequential()
        num_feats = in_channels
        for i in range(depth):
            # Add a layer of hidden neurons
            self.model.append(nn.Linear(num_feats, self.width[i]))
            self.model.append(norm_factory(normalization, self.width[i]))
            self.model.append(act_factory(activation))

            num_feats = self.width[i]

    def forward(self, x):
        """Pass a tensor of features through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            (N, F) Tensor of features

        Paramters
        ---------
        torch.Tensor
            (N, W) Updated tensor of features
        """
        return self.model(x)
