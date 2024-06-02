from typing import Union, Callable, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import torch

from mlreco.utils.metrics import *

from sklearn.neighbors import kneighbors_graph # Move this
import scipy.sparse as sp

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
from sklearn.cluster import DBSCAN

from mlreco import TensorBatch

import copy


def knn_sklearn(coords, k):
    """Create a kNN graph using `scikit-learn`.

    Parameters
    ----------
    coords : Union[np.ndarray, torch.Tensor]
        (N, 3) Set of point coordinates
    k : int
        Number of neighbors in the kNN graph

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (2, E) Edge index
    """
    # If there is less than two points, no edge to be found
    if len(coords) < 2:
        return np.empty((2, 0), dtype=np.int64)

    # Get the appropriate number of neighbors
    k = min(k, len(coords)-1)

    # Dispatch
    if isinstance(coords, torch.Tensor):
        device = coords.device
        G = kneighbors_graph(
                coords.cpu().numpy(), n_neighbors=n_neighbors).tocoo()
        out = np.vstack([G.row, G.col])
        return torch.Tensor(out).long().to(device=device)

    elif isinstance(coords, np.ndarray):
        G = kneighbors_graph(coords, n_neighbors=n_neighbors).tocoo()
        out = np.vstack([G.row, G.col])
        return out

    else:
        raise ValueError(
                f"Coordinate format not recognized: {type(coords)}. Should be "
                 "either `np.ndarray` or `torch.Tensor`.")
