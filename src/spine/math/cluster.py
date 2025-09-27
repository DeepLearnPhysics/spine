"""Numba JIT compiled implementation of clustering routines."""

import numba as nb
import numpy as np

from .distance import METRICS, get_metric_id
from .graph import connected_components, radius_graph

__all__ = ["DBSCAN", "dbscan"]

DBSCAN_DTYPE = (
    ("eps", nb.float32),
    ("min_samples", nb.int64),
    ("metric_id", nb.int64),
    ("p", nb.float32),
)


@nb.experimental.jitclass(DBSCAN_DTYPE)
class DBSCAN:
    """Class-version of the Numba-accelerate :func:`dbscan` function.

    Attributes
    ----------
    eps : float
        Distance scale (determines neighborhood)
    min_samples : int
        Minimum number of neighbors (including oneself) to be considered
        a core point
    metric : str
        Distance metric to be used to establish neighborhood
    """

    def __init__(
        self,
        eps: nb.float32,
        min_samples: nb.int64 = 1,
        metric: nb.types.string = "euclidean",
        p: nb.int64 = 2.0,
    ):
        """Initialize the DBSCAN parameters.

        Parameters
        ----------
        eps : float
            Distance scale (determines neighborhood)
        min_samples : int
            Minimum number of neighbors (including oneself) to be considered
            a core point
        metric : str
            Distance metric to be used to establish neighborhood
        p : float, default 2.
            p-norm factor for the Minkowski metric, if used
        """
        # For Euclidean, save time by using squared Euclidean
        if metric == "euclidean":
            metric = "sqeuclidean"
            eps = eps * eps

        # Store parameters
        self.eps = eps
        self.min_samples = min_samples
        self.metric_id = get_metric_id(metric, p)
        self.p = p

    def fit_predict(self, x):
        """Runs DBSCAN on 3D points and returns the group assignments.

        Parameters
        ----------
        x : np.ndarray
            (N, 3) array of point coordinates
        eps : float
            Distance below which two points are considered neighbors
        min_samples : int
            Minimum number of neighbors for a point to be a core point
        metric : str, default 'euclidean'
            Distance metric used to compute pdist

        Returns
        -------
        np.ndarray
            (N) Group assignments
        """
        # Produce a radius graph
        edge_index = radius_graph(x, self.eps, self.metric_id, self.p)

        # Build groups
        return connected_components(
            edge_index, len(x), self.min_samples, directed=False
        )


@nb.njit(cache=True)
def dbscan(
    x: nb.float32[:, :],
    eps: nb.float32,
    min_samples: nb.int64 = 1,
    metric_id: nb.int64 = METRICS["euclidean"],
    p: nb.float32 = 2.0,
) -> nb.float32[:]:
    """Runs DBSCAN on 3D points and returns the group assignments.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of point coordinates
    eps : float
        Distance below which two points are considered neighbors
    min_samples : int
        Minimum number of neighbors for a point to be a core point
    metric : str, default 'euclidean'
        Distance metric used to compute pdist
    p : float, default 2.
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    np.ndarray
        (N) Group assignments
    """
    # Produce a radius graph
    edge_index = radius_graph(x, eps, metric_id, p)

    # Build groups
    return connected_components(edge_index, len(x), min_samples, directed=False)
