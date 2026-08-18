"""Training utilities for optimizers and learning rate schedulers.

This module provides factory functions for creating PyTorch optimizers and
learning rate schedulers from configuration dictionaries, including support
for custom optimizers like AdaBound.
"""

from importlib import import_module

from spine.config.factory import instantiate, module_dict
from spine.utils.conditional import TORCH_AVAILABLE, torch

__all__ = ["optim_dict", "optim_factory", "lr_sched_factory"]


def optim_dict():
    """Dictionary of valid optimizers."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for optimizer functionality. "
            "Use the released SPINE container or install a compatible "
            "PyTorch ecosystem manually."
        )

    # Load custom implementations only after establishing that the optional
    # runtime is available. This keeps core-only imports independent of Torch.
    adabound = import_module("spine.model.optim.adabound")
    optimizers = {
        "AdaBound": adabound.AdaBound,
        "AdaBoundW": adabound.AdaBoundW,
    }

    # Append the default optimizers from torch
    optimizers.update(module_dict(torch.optim))

    return optimizers


def optim_factory(cfg, params):
    """Instantiates an optimizer from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Optimizer configuration
    params : dict
        Torch model parameters to optimize

    Returns
    -------
    object
        Instantiated optimizer
    """
    return instantiate(optim_dict(), cfg, params=params)


def lr_sched_factory(cfg, optimizer):
    """Instantiates a learning-rate scheduler from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Learning-rate scheduler configuration
    optimizer : object
        Torch optimizer instance

    Returns
    -------
    object
        Instantiated learning-rate optimizer
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for learning rate scheduler functionality. "
            "Use the released SPINE container or install a compatible "
            "PyTorch ecosystem manually."
        )
    lr_sched_dict = module_dict(torch.optim.lr_scheduler)

    return instantiate(lr_sched_dict, cfg, optimizer=optimizer)
