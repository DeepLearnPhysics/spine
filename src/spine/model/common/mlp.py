"""Configurable multi-layer perceptron."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch

from spine.utils.factory import Config

from .act_norm import act_factory, norm_factory

__all__ = ["MLP", "MLPConfig"]


class MLPConfig(TypedDict):
    """Configuration required to construct an :class:`MLP`."""

    depth: int
    width: int | Sequence[int]
    activation: Config
    normalization: Config


class MLP(torch.nn.Module):
    """Apply a configurable stack of dense hidden layers."""

    name = "mlp"

    def __init__(
        self,
        in_channels: int,
        depth: int,
        width: int | Sequence[int],
        activation: Config,
        normalization: Config,
    ) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        in_channels : int
            Number of input features
        depth : int
            Number of hidden layers
        width : int or sequence of int
            Number of neurons in each hidden layer
        activation : str or mapping
            Activation function configuration
        normalization : str or mapping
            Normalization function configuration

        Raises
        ------
        ValueError
            If a feature count or depth is not positive, or a width sequence
            does not contain exactly one entry per hidden layer.
        """
        # Initialize the parent class
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}.")
        if depth < 1:
            raise ValueError(f"`depth` must be positive, got {depth}.")

        # Store the attributes
        self.in_channels = in_channels
        self.depth = depth
        self.act_cfg = activation
        self.norm_cfg = normalization

        # Process the width
        if isinstance(width, int):
            self.width = [width] * depth
        else:
            if len(width) != depth:
                raise ValueError(
                    "If provided as an array, the `width` must be given "
                    "once for each hidden layer (specified in `depth`)"
                )
            self.width = list(width)
        if any(value < 1 for value in self.width):
            raise ValueError("Every hidden-layer width must be positive.")

        self.feature_size = self.width[-1]

        # Initialize the model
        self.model = torch.nn.Sequential()
        num_feats = in_channels
        for i in range(depth):
            # Add a layer of hidden neurons
            self.model.append(torch.nn.Linear(num_feats, self.width[i]))
            self.model.append(norm_factory(normalization, self.width[i]))
            self.model.append(act_factory(activation))

            num_feats = self.width[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a tensor of features through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            (N, F) Tensor of features

        Returns
        -------
        torch.Tensor
            (N, W) Updated tensor of features
        """
        return self.model(x)
