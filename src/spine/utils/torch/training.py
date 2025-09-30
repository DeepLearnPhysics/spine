"""Training utilities for optimizers and learning rate schedulers.

This module provides factory functions for creating PyTorch optimizers and
learning rate schedulers from configuration dictionaries, including support
for custom optimizers like AdaBound.
"""

from ..conditional import TORCH_AVAILABLE, torch
from ..factory import instantiate, module_dict

# Conditionally import AdaBound optimizers
AdaBound = None
AdaBoundW = None
_ADABOUND_AVAILABLE = False
try:
    from .adabound import AdaBound, AdaBoundW

    _ADABOUND_AVAILABLE = True
except ImportError:
    pass

__all__ = ["optim_dict", "optim_factory", "lr_sched_factory"]


def optim_dict():
    """Dictionary of valid optimizers."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for optimizer functionality. "
            "Install with: pip install spine-ml[model]"
        )

    # Start with empty optimizer dict
    optimizers = {}

    # Add AdaBound optimizers if available
    if _ADABOUND_AVAILABLE and AdaBound is not None and AdaBoundW is not None:
        optimizers["AdaBound"] = AdaBound
        optimizers["AdaBoundW"] = AdaBoundW

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
            "Install with: pip install spine-ml[model]"
        )
    lr_sched_dict = module_dict(torch.optim.lr_scheduler)

    return instantiate(lr_sched_dict, cfg, optimizer=optimizer)
