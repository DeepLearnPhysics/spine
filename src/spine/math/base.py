"""Numba JIT compiled implementation of basic functions.

Most of these functions are implemented here because vanilla numba does not
support optional arguments, such as `axis` for most functions or
`return_counts` for the `unique` function.
"""

import numba as nb
import numpy as np

__all__ = [
    "seed",
    "unique",
    "sum",
    "mean",
    "mode",
    "argmax",
    "argmin",
    "amax",
    "amin",
    "all",
    "softmax",
    "log_loss",
]


@nb.njit(cache=True)
def seed(seed_value: int) -> None:
    """Sets the numpy random seed for all Numba jitted functions.

    Note that setting the seed using `np.random.seed` outside a Numba jitted
    function does *not* set the seed of Numba functions.

    Parameters
    ----------
    seed_value : int
        Random number generator seed
    """
    np.random.seed(seed_value)


@nb.njit(cache=True)
def unique(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numba implementation of `np.unique(x, return_counts=True)`.

    Parameters
    ----------
    x : np.ndarray
        (N,) array of values

    Returns
    -------
    np.ndarray
        (U,) array of unique values
    np.ndarray
        (U,) array of counts of each unique value in the original array
    """
    # Nothing to do if the input is empty
    uniques = np.empty(len(x), dtype=x.dtype)
    counts = np.empty(len(x), dtype=np.int64)
    if len(x) == 0:
        return uniques, counts

    # Build the list of unique values and counts
    x = np.sort(x.flatten())
    uniques[0] = x[0]
    idx = 1
    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            uniques[idx] = x[i + 1]
            counts[idx - 1] = i + 1
            idx += 1

    counts[idx - 1] = len(x)

    # Narrow vectors down
    uniques = uniques[:idx]
    counts = counts[:idx]

    # Adjust counts
    counts[1:] = counts[1:] - counts[:-1]

    return uniques, counts


@nb.njit(cache=True)
def sum(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.sum(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `sum` values
    """
    assert axis == 0 or axis == 1
    summ = np.empty(x.shape[1 - axis], dtype=x.dtype)
    if axis == 0:
        for i in range(x.shape[1]):
            summ[i] = np.sum(x[:, i])
    else:
        for i, xi in enumerate(x):
            summ[i] = np.sum(xi)

    return summ


@nb.njit(cache=True)
def mean(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.mean(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `mean` values
    """
    assert axis == 0 or axis == 1
    mean_values = np.empty(x.shape[1 - axis], dtype=x.dtype)
    if axis == 0:
        for i in range(x.shape[1]):
            mean_values[i] = np.mean(x[:, i])
    else:
        for i, xi in enumerate(x):
            mean_values[i] = np.mean(xi)

    return mean_values


@nb.njit(cache=True)
def mode(x: np.ndarray) -> int:
    """Numba implementation of `scipy.stats.mode(x)`.

    Parameters
    ----------
    x : np.ndarray
        (N,) array of values

    Returns
    -------
    int
        Most-probable value in the array
    """
    values, counts = unique(x)

    return values[np.argmax(counts)]


@nb.njit(cache=True)
def argmin(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.argmin(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `argmin` values
    """
    assert axis == 0 or axis == 1
    argmin_values = np.empty(x.shape[1 - axis], dtype=np.int32)
    if axis == 0:
        for i in range(x.shape[1]):
            argmin_values[i] = np.argmin(x[:, i])
    else:
        for i, xi in enumerate(x):
            argmin_values[i] = np.argmin(xi)

    return argmin_values


@nb.njit(cache=True)
def argmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.argmax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `argmax` values
    """
    assert axis == 0 or axis == 1
    argmax_values = np.empty(x.shape[1 - axis], dtype=np.int32)
    if axis == 0:
        for i in range(x.shape[1]):
            argmax_values[i] = np.argmax(x[:, i])

    else:
        for i, xi in enumerate(x):
            argmax_values[i] = np.argmax(xi)

    return argmax_values


@nb.njit(cache=True)
def amin(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.amin(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `min` values
    """
    assert axis == 0 or axis == 1
    xmin = np.empty(x.shape[1 - axis], dtype=x.dtype)
    if axis == 0:
        for i in range(x.shape[1]):
            xmin[i] = np.min(x[:, i])

    else:
        for i, xi in enumerate(x):
            xmin[i] = np.min(xi)

    return xmin


@nb.njit(cache=True)
def amax(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.amax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `max` values
    """
    assert axis == 0 or axis == 1
    xmax = np.empty(x.shape[1 - axis], dtype=x.dtype)
    if axis == 0:
        for i in range(x.shape[1]):
            xmax[i] = np.max(x[:, i])

    else:
        for i, xi in enumerate(x):
            xmax[i] = np.max(xi)

    return xmax


@nb.njit(cache=True)
def all(x: np.ndarray, axis: int) -> np.ndarray:
    """Numba implementation of `np.all(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) Array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,) or (M,) array of `all` outputs
    """
    assert axis == 0 or axis == 1
    all_values = np.empty(x.shape[1 - axis], dtype=np.bool_)
    if axis == 0:
        for i in range(x.shape[1]):
            all_values[i] = np.all(x[:, i])

    else:
        for i, xi in enumerate(x):
            all_values[i] = np.all(xi)

    return all_values


@nb.njit(cache=True)
def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """
    Numba implementation of `scipy.special.softmax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N, M) array of softmax scores
    """
    assert axis == 0 or axis == 1
    if axis == 0:
        xmax = amax(x, axis=0)
        logsumexp = np.log(np.sum(np.exp(x - xmax), axis=0)) + xmax
        return np.exp(x - logsumexp)
    else:
        xmax = amax(x, axis=1).reshape(-1, 1)
        logsumexp = np.log(np.sum(np.exp(x - xmax), axis=1)).reshape(-1, 1) + xmax
        return np.exp(x - logsumexp)


@nb.njit(cache=True)
def log_loss(label: np.ndarray, pred: np.ndarray) -> float:
    """Numba implementation of cross-entropy loss.

    Parameters
    ----------
    label : np.ndarray
        (N,) array of boolean labels (0 or 1)
    pred : np.ndarray
        (N,) array of float scores (between 0 and 1)

    Returns
    -------
    float
        Cross-entropy loss
    """
    if len(label) > 0:
        return -(
            np.sum(np.log(pred[label])) + np.sum(np.log(1.0 - pred[~label]))
        ) / len(label)
    else:
        return 0.0
