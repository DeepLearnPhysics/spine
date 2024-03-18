"""Factories to build generic layers components."""

from mlreco.utils.factory import module_dict, instantiate

from .common import final

__all__ = ['final_construct', 'loss_fn_construct']


def final_construct(in_channels, **cfg):
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


def loss_fn_construct(cfg, functional=False):
    """Instantiates a loss function from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Final layer configuration
    functional : bool, default False
        Whether to return the loss function as a functional
        

    Returns
    -------
    object
        Instantiated loss function
    """
    from torch import nn
    from mlreco.models.experimental.bayes.evidential import EVDLoss

    loss_dict = {
        'ce': nn.CrossEntropyLoss,
        'bce': nn.BCELoss,
        'mm': nn.MultiMarginLoss,
        'huber': nn.HuberLoss,
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'mse': nn.MSELoss,
        'evd': EVDLoss
    }

    loss_dict_func = {
        'ce': nn.functional.cross_entropy,
        'bce': nn.functional.binary_cross_entropy,
        'mm': nn.functional.multi_margin_loss,
        'huber': nn.functional.huber_loss,
        'l1': nn.functional.l1_loss,
        'l2': nn.functional.mse_loss,
        'mse': nn.functional.mse_loss
    }

    if not functional:
        return instantiate(loss_dict, cfg)
    else:
        assert (isinstance(cfg, str) or
                ('name' in cfg and len(cfg) == 1)), (
                       'For a functional, only provide the function name')

        name = cfg if isinstance(cfg, str) else cfg['name']
        try:
            return loss_dict_func[name]
        except KeyError as err:
            raise KeyError("Could not find the functional {name} in the "
                           "availabel list: {loss_dict_func.keys()}")
