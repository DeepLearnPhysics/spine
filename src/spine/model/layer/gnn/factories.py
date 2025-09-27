"""Factories to build the GNN model components."""

from spine.utils.factory import instantiate, module_dict

from . import encode, graph, loss, model

__all__ = [
    "graph_factory",
    "gnn_model_factory",
    "node_encoder_factory",
    "edge_encoder_factory",
    "global_encoder_factory",
    "node_loss_factory",
    "edge_loss_factory",
    "global_loss_factory",
]


def graph_factory(cfg, classes):
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
    graph_dict = module_dict(graph)
    return instantiate(graph_dict, cfg, classes=classes)


def gnn_model_factory(cfg, node_pred, edge_pred, global_pred):
    """Instantiates a GNN model from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        GNN model configuration
    node_pred : bool
        Whether the model should return node features or not
    edge_pred : bool
        Whether the model should return edge features or not
    global_pred : bool
        Whether the model should return global features or not

    Returns
    -------
    object
        Instantiated GNN model
    """
    gnn_model_dict = module_dict(model)
    return instantiate(
        gnn_model_dict,
        cfg,
        node_pred=node_pred,
        edge_pred=edge_pred,
        global_pred=global_pred,
    )


def node_encoder_factory(cfg):
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
    node_encoder_dict = module_dict(encode, pattern="Node")
    return instantiate(node_encoder_dict, cfg)


def edge_encoder_factory(cfg):
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
    edge_encoder_dict = module_dict(encode, pattern="Edge")
    return instantiate(edge_encoder_dict, cfg)


def global_encoder_factory(cfg):
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
    global_encoder_dict = module_dict(encode, pattern="Global")
    return instantiate(global_encoder_dict, cfg)


def node_loss_factory(cfg):
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
    node_loss_dict = module_dict(loss, pattern="Node")
    return instantiate(node_loss_dict, cfg)


def edge_loss_factory(cfg):
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
    edge_loss_dict = module_dict(loss, pattern="Edge")
    return instantiate(edge_loss_dict, cfg)


def global_loss_factory(cfg):
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
    global_loss_dict = module_dict(loss, pattern="Global")
    return instantiate(global_loss_dict, cfg)
