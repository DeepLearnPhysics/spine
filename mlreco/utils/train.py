"""Factory to build an optimizer function from a configuration dictionary."""

from .factory import module_dict, instantiate
from .adabound import AdaBound, AdaBoundW

__all__ = ['optim_factory', 'lr_shed_factory']


def optim_dict():
    """Dictionary of valid optimizers."""
    
    # Start with locally defined optimizers
    optimizers = {
        'AdaBound': AdaBound,
        'AdaBoundW': AdaBoundW
    }

    # Append the default optimizers from torch
    from torch import optim
    optimizers.update(module_dict(optim))

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
    from torch.optim import lr_scheduler
    lr_sched_dict = module_dict(lr_scheduler)

    return instantiate(lr_sched_dict, cfg, optimizer=optimizer)
