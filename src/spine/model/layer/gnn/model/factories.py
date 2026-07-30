"""Module to build GNN message passing submodules."""

from __future__ import annotations

from typing import cast

import torch

from spine.utils.factory import Config, instantiate, module_dict

from . import layer

__all__ = ["node_layer_factory", "edge_layer_factory", "global_layer_factory"]


class FeatureUpdate(torch.nn.Module):
    """Base typing contract for message-passing update modules."""

    feature_size: int


def node_layer_factory(
    cfg: Config,
    node_in: int,
    edge_in: int,
    glob_in: int,
) -> FeatureUpdate:
    """Instantiates a GNN node update layer from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        GNN node update layer configuration
    node_in : int
        Number of input node features
    edge_in : int
        Number of input edge features
    glob_in : int
        Number of input global graph features

    Returns
    -------
    FeatureUpdate
        Instantiated GNN node update layer
    """
    layer_dict = module_dict(layer, pattern="Node")
    return cast(
        FeatureUpdate,
        instantiate(
            layer_dict,
            cfg,
            node_in=node_in,
            edge_in=edge_in,
            glob_in=glob_in,
        ),
    )


def edge_layer_factory(
    cfg: Config,
    node_in: int,
    edge_in: int,
    glob_in: int,
) -> FeatureUpdate:
    """Instantiates a GNN edge update layer from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        GNN edge update layer configuration
    node_in : int
        Number of input node features
    edge_in : int
        Number of input edge features
    glob_in : int
        Number of input global graph features

    Returns
    -------
    FeatureUpdate
        Instantiated GNN edge update layer
    """
    layer_dict = module_dict(layer, pattern="Edge")
    return cast(
        FeatureUpdate,
        instantiate(
            layer_dict,
            cfg,
            node_in=node_in,
            edge_in=edge_in,
            glob_in=glob_in,
        ),
    )


def global_layer_factory(
    cfg: Config,
    node_in: int,
    glob_in: int,
) -> FeatureUpdate:
    """Instantiates a GNN global update layer from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        GNN global update layer configuration
    node_in : int
        Number of input node features
    glob_in : int
        Number of input global graph features

    Returns
    -------
    FeatureUpdate
        Instantiated GNN global update layer
    """
    layer_dict = module_dict(layer, pattern="Global")
    return cast(
        FeatureUpdate,
        instantiate(
            layer_dict,
            cfg,
            node_in=node_in,
            glob_in=glob_in,
        ),
    )
