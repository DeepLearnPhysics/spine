"""Functions which initialize activation and normalization layers
used generically by torch models.
"""

from copy import deepcopy

from torch import nn

from spine.utils.factory import instantiate

__all__ = ["act_factory", "norm_factory"]


def act_dict():
    """Dictionary of recognized activation functions."""

    activations = {
        "none": nn.Identity,
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax,
        "softplus": nn.Softplus,
        "mish": nn.Mish,
    }

    return activations


def norm_dict():
    """Dictionary of recognized normalization functions."""

    normalizations = {
        "none": nn.Identity,
        "batch_norm": nn.BatchNorm1d,
        "instance_norm": nn.InstanceNorm1d,
        "group_norm": nn.GroupNorm,
        "layer_norm": nn.LayerNorm,
    }

    return normalizations


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
