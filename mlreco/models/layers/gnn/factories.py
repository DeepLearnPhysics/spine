"""Factories to build the GNN model components."""

from mlreco.utils.factory import module_dict, instantiate

from . import graphs, encoders, models, losses

__all__ = ['graph_construct', 'gnn_model_construct', 'node_encoder_construct',
           'edge_encoder_construct', 'global_encoder_construct', 
           'node_loss_construct', 'edge_loss_construct', 
           'global_loss_construct']


def graph_construct(cfg, classes):
    """Instantiates a graph constructor from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Graph constructor configuration
    classes : Union[int, list]
        List of classes to build a graph on

    Returns
    -------
    object
        Instantiated graph constructor
    """
    graph_dict = module_dict(graphs)
    return instantiate(graph_dict, cfg, classes=classes)


def gnn_model_construct(cfg):
    """Instantiates a GNN model from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        GNN model configuration

    Returns
    -------
    object
        Instantiated GNN model
    """
    gnn_model_dict = module_dict(models)
    return instantiate(gnn_model_dict, cfg)


def node_encoder_construct(cfg):
    """Instantiates a node encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Node encoder configuration

    Returns
    -------
    object
        Instantiated node encoder
    """
    node_encoder_dict = module_dict(encoders, pattern='Node')
    return instantiate(node_encoder_dict, cfg)


def edge_encoder_construct(cfg):
    """Instantiates an edge encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Edge encoder configuration

    Returns
    -------
    object
        Instantiated edge encoder
    """
    edge_encoder_dict = module_dict(encoders, pattern='Edge')
    return instantiate(edge_encoder_dict, cfg)


def global_encoder_construct(cfg):
    """Instantiates a global graph encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Global graph encoder configuration

    Returns
    -------
    object
        Instantiated global graph encoder
    """
    global_encoder_dict = module_dict(encoders, pattern='Global')
    return instantiate(global_encoder_dict, cfg)


def node_loss_construct(cfg):
    """Instantiates a node loss from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Node loss configuration

    Returns
    -------
    object
        Instantiated node loss
    """
    node_loss_dict = module_dict(losses, pattern='Node')
    return instantiate(node_loss_dict, cfg)


def edge_loss_construct(cfg):
    """Instantiates an edge loss from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Edge loss configuration

    Returns
    -------
    object
        Instantiated edge loss
    """
    edge_loss_dict = module_dict(losses, pattern='Edge')
    return instantiate(edge_loss_dict, cfg)


def global_loss_construct(cfg):
    """Instantiates a global graph loss from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Global graph loss configuration

    Returns
    -------
    object
        Instantiated global graph loss
    """
    global_loss_dict = module_dict(losses, pattern='Global')
    return instantiate(global_loss_dict, cfg)
