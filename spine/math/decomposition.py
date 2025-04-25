"""Numba JIT compiled implementation of decomposition routines."""

import numba as nb
import numpy as np

__all__ = ['principal_components']


@nb.njit(cache=True)
def principal_components(x: nb.float32[:,:]) -> nb.float32[:,:]:
    """Computes the principal components of a point cloud by computing the
    eigenvectors of the centered covariance matrix.

    Parameters
    ----------
    x : np.ndarray
        (N, d) Coordinates in d dimensions

    Returns
    -------
    np.ndarray
        (d, d) List of principal components (row-ordered)
    """
    # Get covariance matrix
    A = np.cov(x.T, ddof = len(x) - 1).astype(x.dtype) # Casting needed...

    # Get eigenvectors
    _, v = np.linalg.eigh(A)
    v = np.ascontiguousarray(np.fliplr(v).T)

    return v
