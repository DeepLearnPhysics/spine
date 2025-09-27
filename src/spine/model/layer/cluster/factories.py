"""Factories to build the CNN-based clustering model components."""

from spine.utils.factory import instantiate, module_dict

from . import kernel, loss

__all__ = ["backbone_factory", "kernel_factory", "loss_factory"]


def backbone_dict():
    """Returns dictionary of backbone classes using name keys.

    Returns
    -------
    dict
        Dictionary of available backbones
    """
    from spine.model.layer.cnn import fpn, uresnet

    models = {"uresnet": uresnet.UResNet, "fpn": fpn.FPN}

    return models


def backbone_factory(name):
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
    backbone_dict = backbone_dict()
    return instantiate(backbone_dict, cfg)


def kernel_factory(cfg):
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


def loss_factory(cfg):
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
