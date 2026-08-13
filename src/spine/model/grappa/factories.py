"""Factories to build the GNN model components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from spine.config.factory import Config, instantiate, module_dict

from . import encode, graph, loss, message_passing
from .graph.base import GraphBase

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


class FeatureEncoder(torch.nn.Module):
    """Base typing contract for graph feature encoders."""

    feature_size: int


class GNNModel(torch.nn.Module):
    """Base typing contract for graph message-passing models."""

    node_feats: int
    edge_feats: int
    global_feats: int
    node_feature_size: int
    edge_feature_size: int
    global_feature_size: int


def graph_factory(cfg: Config, classes: int | Sequence[int]) -> GraphBase:
    """Instantiates a graph constructor from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Graph constructor configuration
    classes : Union[int, list]
        List of classes to build a graph on

    Returns
    -------
    GraphBase
        Instantiated graph constructor
    """
    graph_dict = module_dict(graph)
    return instantiate(graph_dict, cfg, classes=classes)


def gnn_model_factory(
    cfg: Config,
    node_pred: bool,
    edge_pred: bool,
    global_pred: bool,
) -> GNNModel:
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
    GNNModel
        Instantiated GNN model
    """
    gnn_model_dict = module_dict(message_passing)
    return cast(
        GNNModel,
        instantiate(
            gnn_model_dict,
            cfg,
            node_pred=node_pred,
            edge_pred=edge_pred,
            global_pred=global_pred,
        ),
    )


def node_encoder_factory(cfg: Config) -> FeatureEncoder:
    """Instantiates a node encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Node encoder configuration

    Returns
    -------
    FeatureEncoder
        Instantiated node encoder
    """
    node_encoder_dict = module_dict(encode, pattern="Node")
    return cast(FeatureEncoder, instantiate(node_encoder_dict, cfg))


def edge_encoder_factory(cfg: Config) -> FeatureEncoder:
    """Instantiates an edge encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Edge encoder configuration

    Returns
    -------
    FeatureEncoder
        Instantiated edge encoder
    """
    edge_encoder_dict = module_dict(encode, pattern="Edge")
    return cast(FeatureEncoder, instantiate(edge_encoder_dict, cfg))


def global_encoder_factory(cfg: Config) -> FeatureEncoder:
    """Instantiates a global graph encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Global graph encoder configuration

    Returns
    -------
    FeatureEncoder
        Instantiated global graph encoder
    """
    global_encoder_dict = module_dict(encode, pattern="Global")
    return cast(FeatureEncoder, instantiate(global_encoder_dict, cfg))


def node_loss_factory(cfg: Config) -> torch.nn.Module:
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


def edge_loss_factory(cfg: Config) -> torch.nn.Module:
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


def global_loss_factory(cfg: Config) -> torch.nn.Module:
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
