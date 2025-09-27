"""Functions which initialize activation and normalization layers
used by CNNs. These factories build activations and normalizations layers
are based on the MinkowskiEngine package.
"""

from copy import deepcopy

from spine.utils.factory import instantiate

__all__ = ["act_factory", "norm_factory"]


def act_dict():
    """Dictionary of valid activation functions."""
    import MinkowskiEngine as ME
    import MinkowskiEngine.MinkowskiNonlinearity as MENL
    from torch.nn import Identity

    from . import nonlinearities

    activations = {
        "none": Identity,
        "relu": ME.MinkowskiReLU,
        "prelu": ME.MinkowskiPReLU,
        "selu": ME.MinkowskiSELU,
        "celu": ME.MinkowskiCELU,
        "tanh": ME.MinkowskiTanh,
        "sigmoid": ME.MinkowskiSigmoid,
        "lrelu": MENL.MinkowskiLeakyReLU,
        "elu": MENL.MinkowskiELU,
        "mish": nonlinearities.MinkowskiMish,
    }

    return activations


def norm_dict():
    """Dictionary of valid normalization functions."""
    import MinkowskiEngine as ME
    from torch.nn import Identity

    from . import normalizations

    norm_layers = {
        "none": Identity,
        "batch_norm": ME.MinkowskiBatchNorm,
        "instance_norm": ME.MinkowskiInstanceNorm,
        "pixel_norm": normalizations.MinkowskiPixelNorm,
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
