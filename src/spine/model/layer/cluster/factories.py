"""Factories to build the CNN-based clustering model components."""

from __future__ import annotations

import torch

from spine.utils.factory import Config, Registry, instantiate, module_dict

from . import kernel, loss

__all__ = ["backbone_factory", "kernel_factory", "loss_factory"]


def backbone_dict() -> Registry:
    """Returns dictionary of backbone classes using name keys.

    Returns
    -------
    dict
        Dictionary of available backbones
    """
    from spine.model.layer.cnn.fpn import FPN
    from spine.model.layer.cnn.uresnet_layers import UResNet

    models = {"uresnet": UResNet, "fpn": FPN}

    return models


def backbone_factory(cfg: Config) -> torch.nn.Module:
    """Instantiates a backbone model from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Backbone configuration

    Returns
    -------
    object
        Instantiated backbone function
    """
    if isinstance(cfg, str):
        raise ValueError(
            "CNN backbones require a configuration block, not only a name."
        )

    config = dict(cfg)
    try:
        name = config.pop("name")
    except KeyError as err:
        raise ValueError("Backbone configuration requires a `name`.") from err

    models = backbone_dict()
    try:
        model_type = models[name]
    except KeyError as err:
        raise ValueError(
            f"Unknown backbone '{name}'. Available backbones: " f"{list(models)}."
        ) from err
    return model_type(config)


def kernel_factory(cfg: Config) -> torch.nn.Module:
    """Instantiates an edge kernel from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Kernel configuration

    Returns
    -------
    object
        Instantiated kernel function
    """
    kernel_dict = module_dict(kernel)
    return instantiate(kernel_dict, cfg)


def loss_factory(cfg: Config) -> torch.nn.Module:
    """Instantiates a clustering loss from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Clustering loss configuration

    Returns
    -------
    object
        Instantiated clustering loss function
    """
    loss_dict = module_dict(loss)
    return instantiate(loss_dict, cfg)
