"""Factories for model-independent losses, metrics, and output heads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from spine.config.factory import Config, instantiate, module_dict
from spine.model.common.evidential import EDLRegressionLoss, EVDLoss

from . import final, losses, metric

__all__ = ["loss_fn_factory", "metric_fn_factory", "final_factory"]


def loss_fn_factory(
    cfg: Config,
    functional: bool = False,
    **kwargs: Any,
) -> torch.nn.Module | Callable[..., Any]:
    """Instantiates a loss function from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Final layer configuration
    functional : bool, default False
        Whether to return the loss function as a functional
    **kwargs : dict, optional
        Additional parameters to pass to the loss function

    Returns
    -------
    object
        Instantiated loss function
    """
    loss_dict = {
        "ce": torch.nn.CrossEntropyLoss,
        "bce": torch.nn.BCELoss,
        "bce_logits": torch.nn.BCEWithLogitsLoss,
        "mm": torch.nn.MultiMarginLoss,
        "huber": torch.nn.HuberLoss,
        "l1": torch.nn.L1Loss,
        "l2": torch.nn.MSELoss,
        "mse": torch.nn.MSELoss,
        "evd": EVDLoss,
        "edl": EDLRegressionLoss,
        **module_dict(losses),
    }

    loss_dict_func = {
        "ce": torch.nn.functional.cross_entropy,
        "bce": torch.nn.functional.binary_cross_entropy,
        "bce_logits": torch.nn.functional.binary_cross_entropy_with_logits,
        "mm": torch.nn.functional.multi_margin_loss,
        "huber": torch.nn.functional.huber_loss,
        "l1": torch.nn.functional.l1_loss,
        "l2": torch.nn.functional.mse_loss,
        "mse": torch.nn.functional.mse_loss,
    }

    if not functional:
        return instantiate(loss_dict, cfg, **kwargs)

    if not isinstance(cfg, str) and ("name" not in cfg or len(cfg) != 1):
        raise ValueError("For a functional, only provide the function name.")

    name = cfg if isinstance(cfg, str) else cfg["name"]
    try:
        return loss_dict_func[name]
    except KeyError as err:
        raise KeyError(
            f"Could not find the functional {name} in the "
            f"available list: {loss_dict_func.keys()}"
        ) from err


def metric_fn_factory(cfg: Config) -> torch.nn.Module:
    """Instantiates a metric function from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Metric function configuration

    Returns
    -------
    object
        Instantiated metric function
    """
    metric_layers = module_dict(metric)
    return instantiate(metric_layers, cfg)


def final_factory(in_channels: int, **cfg: Any) -> torch.nn.Module:
    """Instantiates a final layer from a configuration dictionary.

    Parameters
    ----------
    in_channels : int
        Number of features input into the final layer
    **cfg : dict
        Final layer configuration

    Returns
    -------
    object
        Instantiated final layer
    """
    final_layers = module_dict(final)
    return instantiate(final_layers, cfg, in_channels=in_channels)
