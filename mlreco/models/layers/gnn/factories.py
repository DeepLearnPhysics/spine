"""Factories to build the GNN model components."""

from mlreco.utils.factory import module_dict, instantiate

from . import graphs, encoders, message_passing, losses


def graph_construct(cfg):
    """Instantiates a graph constructor from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Graph constructor configuration

    Returns
    -------
    object
        Instantiated graph constructor
    """
    graph_dict = module_dict(graphs)
    return instantiate(graph_dict, cfg)


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
    gnn_model_dict = module_dict(message_passing)
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
