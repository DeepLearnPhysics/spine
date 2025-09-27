"""Numba JIT compiled implementation of neighbor query routines.

In particular, this module supports:
- Radius-based neighbor classification
- kNN-based neighbor classification
"""

import numba as nb
import numpy as np

from .base import mode
from .distance import METRICS, cdist, get_metric_id

__all__ = ["RadiusNeighborsClassifier", "KNeighborsClassifier"]


RNC_DTYPE = (
    ("radius", nb.float32),
    ("metric_id", nb.int64),
    ("p", nb.float32),
    ("iterate", nb.boolean),
)


KNC_DTYPE = (("k", nb.int64), ("metric_id", nb.int64), ("p", nb.float32))


@nb.experimental.jitclass(RNC_DTYPE)
class RadiusNeighborsClassifier:
    """Class which assigns labels to points based on radial neighborhood
    majority vote.

    More specifically, for each point that is to be labeled:
    - Find all labeled points within some radius R;
    - Label the point based on majority vote.

    If there are no labeled points in the neighborhood of a query point, a
    label of -1 is assigned to the query point.

    Currently this is bruteforced with cdist, but in the future this is
    intended to be used with a KDTree backend for quicker query.

    Attributes
    ----------
    radius : float
        Radius around which to check
    metric_id : int
        Distance metric enumerator
    p : float
        p-norm factor for the Minkowski metric, if used
    iterate : bool
        Whether to recurse the search until no new labels are assigned
    """

    def __init__(
        self,
        radius: nb.float32,
        metric: nb.types.string = "euclidean",
        p: nb.float32 = 2.0,
        iterate: nb.boolean = True,
    ):
        """Initialize the RadiusNeighborsClassifier parameters.

        Parameters
        ----------
        radius : float
            Radius around which to check
        metric : str, default 'euclidean'
            Distance metric
        p : float, default 2.
            p-norm factor for the Minkowski metric, if used
        iterate : bool, default True
            Whether to recurse the search until no new labels are assigned
        """
        # For Euclidean, save time by using squared Euclidean
        if metric == "euclidean":
            metric = "sqeuclidean"
            radius = radius * radius

        # Store parameters
        self.radius = radius
        self.metric_id = get_metric_id(metric, p)
        self.p = p
        self.iterate = iterate

    def fit_predict(self, X: nb.float32[:, :], y: nb.float32[:], Xq: nb.float32[:, :]):
        """Assign labels to a set of points given a set of reference points.

        Parameters
        ----------
        X : np.ndarray
            (N, 3) Set of reference points
        y : np.ndarray
            (N) Labels of reference points
        Xq : nb.ndarray
            (M, 3) Set of query points

        Returns
        -------
        np.ndarray
            (M) Labels assigned to the query points
        np.ndarray
            Index of points which have not been sucessfully assigned
        """
        # Loop over query points until no new labels can be assigned
        num_query = len(Xq)
        labels = np.empty(num_query, dtype=np.int64)
        orphan_index = np.arange(num_query, dtype=np.int64)
        while num_query > 0:
            # Start by computing the distance between the query and reference
            dists = cdist(Xq, X, metric_id=self.metric_id, p=self.p)

            # Fetch the mask of reference points closer than some radius
            mask = dists < self.radius

            # Loop over query points
            assigned = np.zeros(num_query, dtype=nb.boolean)
            for i in range(num_query):
                # Find the set of points within the predefined radius
                index = np.where(mask[i])[0]

                # Use the mode to define the label
                if len(index):
                    labels[orphan_index[i]] = mode(y[index])
                    assigned[i] = True
                else:
                    labels[orphan_index[i]] = -1

            # If the number of orphans is unchanged, break
            orphan_update = np.where(~assigned)[0]
            if len(orphan_update) == 0 or len(orphan_update) == num_query:
                orphan_index = orphan_index[orphan_update]
                break

            # If no recursion is required, abort loop
            if not self.iterate:
                orphan_index = orphan_index[orphan_update]
                break

            # Update the reference and query points
            label_update = np.where(assigned)[0]
            X = Xq[label_update]
            Xq = Xq[orphan_update]
            y = labels[orphan_index[label_update]]

            # Update orphan list
            orphan_index = orphan_index[orphan_update]
            num_query = len(orphan_index)

        return labels, orphan_index


@nb.experimental.jitclass(KNC_DTYPE)
class KNeighborsClassifier:
    """Class which assigns labels to points based on a nearest neighbor
    majority vote.

    More specifically, for each point that is to be labeled:
    - Find the k closest labeled points;
    - Label the point based on majority vote.

    If there are no labeled points in the neighborhood of a query point, a
    label of -1 is assigned to the query point.

    Currently this is bruteforced with cdist, but in the future this is
    intended to be used with a KDTree backend for quicker query.

    Attributes
    ----------
    k : int
        Number of neighbors to query
    metric_id : int
        Distance metric enumerator
    p : float
        p-norm factor for the Minkowski metric, if used
    """

    def __init__(
        self, k: nb.int64, metric: nb.types.string = "euclidean", p: nb.float32 = 2.0
    ):
        """Initialize the RadiusNeighborsClassifier parameters.

        Parameters
        ----------
        k : int
            Number of neighbors to query
        metric : str, default 'euclidean'
            Distance metric
        p : float, default 2.
            p-norm factor for the Minkowski metric, if used
        """
        # For Euclidean, save time by using squared Euclidean
        if metric == "euclidean":
            metric = "sqeuclidean"

        # Store parameters
        self.k = k
        self.metric_id = get_metric_id(metric, p)
        self.p = p

    def fit_predict(self, X: nb.float32[:, :], y: nb.float32[:], Xq: nb.float32[:, :]):
        """Assign labels to a set of points given a set of reference points.

        Parameters
        ----------
        X : np.ndarray
            (N, 3) Set of reference points
        y : np.ndarray
            (N) Labels of reference points
        Xq : nb.ndarray
            (M, 3) Set of query points

        Returns
        -------
        np.ndarray
            (M) Labels assigned to the query points
        np.ndarray
            Index of points which have not been sucessfully assigned
        """
        # If there are no labeled points provided, nothing to do
        if len(X) == 0:
            return (
                np.full(len(Xq), -1, dtype=np.int64),
                np.arange(len(Xq), dtype=np.int64),
            )

        # Start by computing the distance between the query and reference
        dists = cdist(Xq, X, metric_id=self.metric_id, p=self.p)

        # Loop over query poins
        labels = np.empty(len(Xq), dtype=np.int64)
        for i in range(len(Xq)):
            # Find the list k closest labels
            index = np.argsort(dists[i])[: self.k]

            # Use the mode to define the label
            labels[i] = mode(y[index])

        return labels, np.empty(0, dtype=np.int64)
