"""Functions which initialize activation and normalization layers
used generically by torch models. 
"""

from copy import deepcopy
from torch import nn

from mlreco.utils.factory import instantiate

__all__ = ['act_construct', 'norm_construct']


def act_dict():
    """Dictionary of valid activation functions."""

    activations = {
        'elu': nn.ELU,
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
        'celu': nn.CELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'softmax': nn.Softmax,
        'softplus': nn.Softplus,
        'mish': nn.Mish
    }

    return activations


def norm_dict():
    """Dictionary of valid normalization functions."""

    normalizations = {
        'none': nn.Identity,
        'batch_norm': nn.BatchNorm1d,
        'instance_norm': nn.InstanceNorm1d,
        'group_norm': nn.GroupNorm,
        'layer_norm': nn.LayerNorm
    }

    return normalizations


def act_construct(cfg):
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


def norm_construct(cfg, num_features=None):
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
    if isinstance(cfg, str):
        cfg = {'name': cfg}

    if num_features is not None:
        cfg = deepcopy(cfg)
        cfg['kwargs'] = cfg.get('kwargs', {})
        cfg['kwargs']['num_features'] = num_features

    return instantiate(norm_dict(), cfg)