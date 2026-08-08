"""Factories for reusable convolutional neural network components."""

from __future__ import annotations

import torch

from spine.utils.factory import Config, instantiate

from .encoder import SparseResidualEncoder

__all__ = ["encoder_factory"]


def encoder_factory(cfg: Config) -> torch.nn.Module:
    """Instantiate an image encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Image encoder configuration.

    Returns
    -------
    torch.nn.Module
        Instantiated image encoder.
    """
    encoder_dict = {"cnn": SparseResidualEncoder}
    return instantiate(encoder_dict, cfg)
