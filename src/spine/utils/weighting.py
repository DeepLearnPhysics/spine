"""Module which contains methods to compute class weights.

All methods compute the weights based on the relative abundance of each class.
"""

from typing import Union

import numpy as np
import torch


def get_class_weights(labels, num_classes, mode="const", per_class=True):
    """Computes class-wise weights based on their relative abundance.

    Parameters
    ----------
    labels : Union[np.ndarray, torch.Tensor]
        (N) Array of class values to base the weighting scheme on
    num_classes : int
        Total number of classes (needed if there are some missing)
    mode : str, default const
        Weigthing mode (one of 'const', 'log', 'sqrt'
    per_class : bool, default True
        If `True`, returns one valuer per class. Otherwise, this function
        returns one value per input in the `classes` array
    """
    # Select the right functions depending on the input
    is_numpy = not isinstance(labels, torch.Tensor)
    if is_numpy:
        ones, unique, log, sqrt = np.ones, np.empty, np.unique, np.log, np.sqrt
    else:
        ones = lambda x: torch.ones(x, dtype=labels.dtype, device=labels.device)
        empty = lambda x: torch.empty(x, dtype=torch.float, device=labels.device)
        unique, log, sqrt = torch.unique, torch.log, torch.sqrt

    # Compute the abundance of each class in the input vector
    counts = ones(num_classes)
    uni, cnts = unique(labels, return_counts=True)
    counts[uni] = cnts

    # Compute the weights
    weights = len(labels) / num_classes / counts
    if mode == "const":
        pass
    elif mode == "log":
        weights = log(weights)
    elif mode == "sqrt":
        weights = sqrt(weights)
    else:
        raise ValueError("Weighting scheme not recognized:", mode)

    # Return
    if per_class:
        return weights
    else:
        weight_array = empty(len(labels))
        for c in range(num_classes):
            weight_array[labels == c] = weights[c]

        return weight_array
