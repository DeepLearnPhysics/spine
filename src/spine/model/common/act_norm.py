"""Factories for dense activation and normalization layers."""

from __future__ import annotations

import torch

from spine.config.factory import Config, Registry, instantiate

__all__ = ["act_factory", "norm_factory"]


def act_dict() -> Registry:
    """Build the registry of supported dense activation layers.

    Returns
    -------
    Registry
        Mapping from configuration names to activation module classes.
    """

    activations = {
        "none": torch.nn.Identity,
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "lrelu": torch.nn.LeakyReLU,
        "prelu": torch.nn.PReLU,
        "selu": torch.nn.SELU,
        "celu": torch.nn.CELU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softmax": torch.nn.Softmax,
        "softplus": torch.nn.Softplus,
        "mish": torch.nn.Mish,
    }

    return activations


def norm_dict() -> Registry:
    """Build the registry of supported dense normalization layers.

    Returns
    -------
    Registry
        Mapping from configuration names to normalization module classes.
    """

    normalizations = {
        "none": torch.nn.Identity,
        "batch_norm": torch.nn.BatchNorm1d,
        "instance_norm": torch.nn.InstanceNorm1d,
        "group_norm": torch.nn.GroupNorm,
        "layer_norm": torch.nn.LayerNorm,
    }

    return normalizations


def act_factory(cfg: Config) -> torch.nn.Module:
    """Instantiates an activation layer.

    Parameters
    ----------
    cfg : str or mapping
        Activation name or configuration containing constructor arguments.

    Returns
    -------
    torch.nn.Module
        Instantiated activation layer.
    """
    return instantiate(act_dict(), cfg)


def norm_factory(cfg: Config, num_features: int | None = None) -> torch.nn.Module:
    """Instantiates a normalization layer.

    Parameters
    ----------
    cfg : str or mapping
        Normalization name or configuration containing constructor arguments.
    num_features : int, optional
        Number of features to normalize.

    Returns
    -------
    torch.nn.Module
        Instantiated normalization layer.
    """
    return instantiate(norm_dict(), cfg, num_features=num_features)
