"""Numba JIT compiled implementation of decomposition routines."""

import numba as nb
import numpy as np

__all__ = ["PCA", "principal_components"]

PCA_DTYPE = (("n_components", nb.int64),)


@nb.experimental.jitclass(PCA_DTYPE)
class PCA:
    """Class-version of the Numba-accelerate :func:`principal_components` function.

    Attributes
    ----------
    n_components : int
        Number of PCA components
    components : np.ndarray
        (N_c, D) List of principal axes
    explained_variance : np.ndarray
        (N_c) Variance along each of the principal axes
    """

    def __init__(self, n_components: nb.int64):
        """Initialize the PCA parameters.

        Parameters
        ----------
        n_components : int
            Number of PCA components, N_c
        """
        # Store parameters
        assert n_components > 0, "Must at least include one component."
        self.n_components = n_components

    def fit(self, x):
        """Computes the covariance and eigen-decompose the data.

        Parameters
        ----------
        x : np.ndarray
            (N, D) array of point coordinates in some D-dimensional space

        Returns
        -------
        components : np.ndarray
            (N_c, D) List of principal axes
        explained_variance : np.ndarray
            (N_c) Variance along each of the principal axes
        """
        # Check input
        assert x.shape[1] >= self.n_components, (
            f"The dimensionality of the data ({x.shape[1]}) is smaller "
            f"than the number of components ({self.n_components}."
        )

        # Compute the covariance matrix
        A = np.cov(x.T, ddof=len(x) - 1).astype(x.dtype)

        # Eigen-decompose the covariance matrix
        w, v = np.linalg.eigh(A)
        w, v = np.flip(w), np.ascontiguousarray(np.fliplr(v).T)

        # Store output
        return v[: self.n_components], w[: self.n_components] / (len(x) - 1)


@nb.njit(cache=True)
def principal_components(x: nb.float32[:, :]) -> nb.float32[:, :]:
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
    A = np.cov(x.T, ddof=len(x) - 1).astype(x.dtype)  # Casting needed...

    # Get eigenvectors
    _, v = np.linalg.eigh(A)
    v = np.ascontiguousarray(np.fliplr(v).T)

    return v
