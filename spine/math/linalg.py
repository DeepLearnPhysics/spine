"""Numba JIT compiled implementation of linear algebra routines."""

import numpy as np
import numba as nb

__all__ = ['norm', 'submatrix']


@nb.njit(cache=True)
def norm(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """Numba implementation of `np.linalg.norm(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `norm` values
    """
    assert axis == 0 or axis == 1
    xnorm = np.empty(x.shape[1-axis], dtype=x.dtype)
    if axis == 0:
        for i in range(len(xnorm)):
            xnorm[i] = np.linalg.norm(x[:,i])
    else:
        for i in range(len(xnorm)):
            xnorm[i] = np.linalg.norm(x[i])

    return xnorm


@nb.njit(cache=True)
def submatrix(x: nb.float32[:,:],
              index1: nb.int32[:],
              index2: nb.int32[:]) -> nb.float32[:,:]:
    """Numba implementation of matrix subsampling.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    index1 : np.ndarray
        (N') array of indices along axis 0 in the input matrix
    index2 : np.ndarray
        (M') array of indices along axis 1 in the input matrix

    Returns
    -------
    np.ndarray
        (N',M') array of values from the original matrix
    """
    subx = np.empty((len(index1), len(index2)), dtype=x.dtype)
    for i, i1 in enumerate(index1):
        for j, i2 in enumerate(index2):
            subx[i, j] = x[i1, i2]

    return subx


@nb.njit(cache=True)
def contingency_table(x: nb.int32[:],
                      y: nb.int32[:],
                      nx: nb.int32 = None,
                      ny: nb.int32 = None) -> nb.int64[:, :]:
    """Build a contingency table for two sets of labels.

    Parameters
    ----------
    x : np.ndarray
        (N) Array of integrer values
    y : np.ndarray
        (M) Array of integrer values
    nx : int, optional
        Number of integer values allowed in `x`, N
    ny : int, optional
        Number of integer values allowd in `y`, M

    Returns
    -------
    np.ndarray
        (N, M) Contingency table
    """
    # If not provided, assume that the max label is the max of the label array
    if not nx:
        nx = np.max(x) + 1
    if not ny:
        ny = np.max(y) + 1

    # Bin the table
    table = np.zeros((nx, ny), dtype=np.int64)
    for i, j in zip(x, y):
        table[i, j] += 1

    return table
