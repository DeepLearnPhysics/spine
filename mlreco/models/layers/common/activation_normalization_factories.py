"""Functions which initialize activation and normalization layers."""

from copy import deepcopy

from mlreco.utils.factory import instantiate

__all__ = ['activations_construct', 'normalizations_construct']


def activations_dict():
    """Dictionary of valid activation functions."""
    import MinkowskiEngine as ME
    import MinkowskiEngine.MinkowskiNonlinearity as MENL
    from . import nonlinearities

    activations = {
        'relu': ME.MinkowskiReLU,
        'prelu': ME.MinkowskiPReLU,
        'selu': ME.MinkowskiSELU,
        'celu': ME.MinkowskiCELU,
        'tanh': ME.MinkowskiTanh,
        'sigmoid': ME.MinkowskiSigmoid,
        'lrelu': MENL.MinkowskiLeakyReLU,
        'elu': MENL.MinkowskiELU,
        'mish': nonlinearities.MinkowskiMish
    }

    return activations


def normalizations_dict():
    """Dictionary of valid normalization functions."""
    import MinkowskiEngine as ME
    from . import normalizations
    from .blocks import Identity

    norm_layers = {
        'none': Identity,
        'batch_norm': ME.MinkowskiBatchNorm,
        'instance_norm': ME.MinkowskiInstanceNorm,
        'pixel_norm': normalizations.MinkowskiPixelNorm
    }

    return norm_layers


def activations_construct(cfg):
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
    return instantiate(activations_dict(), cfg)


def normalizations_construct(cfg, num_features=None):
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

    return instantiate(normalizations_dict(), cfg)
