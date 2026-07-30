"""Factories for backend-neutral sparse activations and normalizations."""

from __future__ import annotations

import torch

from spine.utils.factory import Config, Registry, instantiate

__all__ = ["act_factory", "norm_factory"]


def act_dict() -> Registry:
    """Build the registry of activation layers available to sparse CNNs.

    Returns
    -------
    Registry
        Mapping from configuration names to activation module classes.
    """
    from spine.model import sparse

    from . import nonlinearities

    activations = {
        "none": torch.nn.Identity,
        "relu": sparse.ReLU,
        "prelu": sparse.PReLU,
        "selu": sparse.SELU,
        "celu": sparse.CELU,
        "tanh": sparse.Tanh,
        "sigmoid": sparse.Sigmoid,
        "lrelu": sparse.LeakyReLU,
        "elu": sparse.ELU,
        "mish": nonlinearities.Mish,
    }

    return activations


def norm_dict() -> Registry:
    """Build the registry of normalization layers available to sparse CNNs.

    Returns
    -------
    Registry
        Mapping from configuration names to normalization module classes.
    """
    from spine.model import sparse

    from . import normalizations

    norm_layers = {
        "none": torch.nn.Identity,
        "batch_norm": sparse.BatchNorm,
        "instance_norm": sparse.InstanceNorm,
        "pixel_norm": normalizations.PixelNorm,
    }

    return norm_layers


def act_factory(cfg: Config) -> torch.nn.Module:
    """Instantiate an activation layer from configuration.

    Parameters
    ----------
    cfg : str or mapping
        Activation name or configuration containing the layer name and its
        constructor arguments.

    Returns
    -------
    torch.nn.Module
        Instantiated activation layer.
    """
    return instantiate(act_dict(), cfg)


def norm_factory(cfg: Config, num_features: int | None = None) -> torch.nn.Module:
    """Instantiate a normalization layer from configuration.

    Parameters
    ----------
    cfg : str or mapping
        Normalization name or configuration containing the layer name and its
        constructor arguments.
    num_features : int, optional
        Number of feature channels to normalize. Layers without channel-wise
        parameters ignore this value.

    Returns
    -------
    torch.nn.Module
        Instantiated normalization layer.
    """
    return instantiate(norm_dict(), cfg, num_features=num_features)
