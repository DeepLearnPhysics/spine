"""Sparse squeeze-and-excitation UResNet backbone."""

from __future__ import annotations

from typing import Any

import torch

from .blocks import SEResNetBlock
from .uresnext import UResNeXt

__all__ = ["SENet"]


class SENet(UResNeXt):
    """Sparse U-shaped backbone with squeeze-and-excitation residual blocks.

    The class reuses the UResNeXt encoder-decoder assembly but substitutes
    :class:`SEResNetBlock` at every feature level.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        se_ratio: int = 8,
    ) -> None:
        """Initialize the squeeze-and-excitation backbone.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        se_ratio : int, default 8
            Channel reduction ratio in each squeeze-and-excitation block.

        Raises
        ------
        ValueError
            If ``se_ratio`` is not positive.
        """
        if se_ratio < 1:
            raise ValueError(f"`se_ratio` must be positive, got {se_ratio}.")
        self.se_ratio = se_ratio

        # UResNeXt owns the common encoder-decoder assembly. Cardinality one
        # disables its divisibility constraint; `_make_block` below supplies
        # SE residual blocks instead of grouped ResNeXt blocks.
        super().__init__(cfg, cardinality=1, dilations=(1,))

    def _make_block(
        self,
        in_features: int,
        out_features: int,
    ) -> torch.nn.Module:
        """Build one squeeze-and-excitation residual block.

        Parameters
        ----------
        in_features : int
            Number of input feature channels.
        out_features : int
            Number of output feature channels.

        Returns
        -------
        torch.nn.Module
            Initialized SE residual block.
        """
        return SEResNetBlock(
            in_features,
            out_features,
            se_ratio=self.se_ratio,
            dimension=self.dimension,
            activation=self.act_cfg,
            normalization=self.norm_cfg,
        )
