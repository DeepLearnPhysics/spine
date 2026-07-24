"""Factories for backend-neutral sparse activations and normalizations."""

from copy import deepcopy

from spine.utils.factory import instantiate

__all__ = ["act_factory", "norm_factory"]


def act_dict():
    """Dictionary of valid activation functions."""
    from torch.nn import Identity

    from spine.model import sparse

    from . import nonlinearities

    activations = {
        "none": Identity,
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


def norm_dict():
    """Dictionary of valid normalization functions."""
    from torch.nn import Identity

    from spine.model import sparse

    from . import normalizations

    norm_layers = {
        "none": Identity,
        "batch_norm": sparse.BatchNorm,
        "instance_norm": sparse.InstanceNorm,
        "pixel_norm": normalizations.PixelNorm,
    }

    return norm_layers


def act_factory(cfg):
    """Instantiates an activation layer.

    Parameters
    ----------
    cfg : dict
        Activation layer configuration

    Return
    ------
    object
        Instantiated activation layer
    """
    return instantiate(act_dict(), cfg)


def norm_factory(cfg, num_features=None):
    """Instantiates a normalization layer.

    Parameters
    ----------
    cfg : dict
        Normalization layer configuration
    num_features : int
        Number of features to normalize

    Return
    ------
    object
        Instantiated normalization layer
    """
    return instantiate(norm_dict(), cfg, num_features=num_features)
