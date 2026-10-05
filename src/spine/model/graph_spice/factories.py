"""Factories for clustering-model components."""

from __future__ import annotations

import torch

from spine.config.factory import (
    Config,
    Registry,
    extract_module_parameters,
    instantiate,
    resolve_module_provider,
)

from .kernel import BilinearKernel, DefaultKernel, MLPKernel
from .loss import EdgeLoss

__all__ = ["backbone_factory", "kernel_factory", "loss_factory"]


def backbone_dict() -> Registry:
    """Build the registry of supported clustering backbones.

    Returns
    -------
    dict
        Mapping from configuration names to backbone classes.
    """
    from spine.model.cnn.fpn import FPN
    from spine.model.cnn.uresnet_layers import UResNet

    models = {"uresnet": UResNet, "fpn": FPN}

    return models


def backbone_factory(cfg: Config) -> torch.nn.Module:
    """Instantiate a clustering backbone.

    Parameters
    ----------
    cfg : mapping
        Backbone configuration. Unlike parameter-free factories, a backbone
        must be configured with a mapping containing a ``provider`` (legacy ``name`` is deprecated).

    Returns
    -------
    torch.nn.Module
        Instantiated backbone.
    """
    if isinstance(cfg, str):
        raise ValueError(
            "CNN backbones require a configuration block, not only a name."
        )

    name = resolve_module_provider(cfg)
    if name is None:
        raise ValueError("Backbone configuration requires a `provider`.")
    config = extract_module_parameters(
        {key: value for key, value in cfg.items() if key not in {"name", "provider"}},
        context=f"Backbone {name}",
    )

    backbones = backbone_dict()
    try:
        model_type = backbones[name]
    except KeyError as err:
        raise ValueError(
            f"Unknown backbone '{name}'. Available backbones: {list(backbones)}."
        ) from err
    return model_type(config)


def kernel_factory(cfg: Config) -> torch.nn.Module:
    """Instantiate a supported edge kernel.

    Parameters
    ----------
    cfg : str or mapping
        Kernel configuration

    Returns
    -------
    torch.nn.Module
        Instantiated edge kernel.
    """
    kernels = {
        DefaultKernel.name: DefaultKernel,
        BilinearKernel.name: BilinearKernel,
        MLPKernel.name: MLPKernel,
    }
    return instantiate(kernels, cfg)


def loss_factory(cfg: Config) -> torch.nn.Module:
    """Instantiate a supported clustering loss.

    Parameters
    ----------
    cfg : str or mapping
        Clustering loss configuration

    Returns
    -------
    torch.nn.Module
        Instantiated clustering loss.
    """
    losses = {EdgeLoss.name: EdgeLoss}
    return instantiate(losses, cfg)
