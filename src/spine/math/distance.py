"""Numba JIT compiled implementation of distance computation routines.

This module is entirely dedicated to 3D points, which is the core representation
of objects targeted by this software package.
"""

import numba as nb
import numpy as np

from .base import argmin, mean

__all__ = [
    "cityblock",
    "euclidean",
    "sqeuclidean",
    "minkowski",
    "chebyshev",
    "pdist",
    "cdist",
    "farthest_pair",
    "closest_pair",
    "closest_pair_legacy",
]

MINKOWSKI = 0
CITYBLOCK = 1
EUCLIDEAN = 2
SQEUCLIDEAN = 3
CHEBYSHEV = 4

# Available distance metrics. Keep the public mapping for callers, while using
# named integer constants internally so Numba sees stable scalar IDs.
METRICS = {
    "minkowski": MINKOWSKI,
    "cityblock": CITYBLOCK,
    "euclidean": EUCLIDEAN,
    "sqeuclidean": SQEUCLIDEAN,
    "chebyshev": CHEBYSHEV,
}


@nb.njit(cache=True)
def get_metric_id(metric: str, p: float) -> int:
    """Checks on the metric name, returns an enumerated form of the metric.

    Parameters
    ----------
    metric : str, default 'euclidean'
        Distance metric
    p : float
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    int
        Enumerated form of the distance metric
    """
    if metric == "minkowski":
        if p == 1.0:
            return CITYBLOCK
        elif p == 2.0:
            return EUCLIDEAN
        else:
            return MINKOWSKI
    elif metric == "cityblock":
        return CITYBLOCK
    elif metric == "euclidean":
        return EUCLIDEAN
    elif metric == "sqeuclidean":
        return SQEUCLIDEAN
    elif metric == "chebyshev":
        return CHEBYSHEV
    else:
        raise ValueError(f"Distance metric not recognized: {metric}")


@nb.njit(cache=True)
def cityblock(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the cityblock distance (L1) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3,) Coordinates of the first point
    y : np.ndarray
        (3,) Coordinates of the second point

    Returns
    -------
    float
        Cityblock distance
    """
    return abs(y[0] - x[0]) + abs(y[1] - x[1]) + abs(y[2] - x[2])


@nb.njit(cache=True)
def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Euclidean distance (L2) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3,) Coordinates of the first point
    y : np.ndarray
        (3,) Coordinates of the second point

    Returns
    -------
    float
        Euclidean distance
    """
    return np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2)


@nb.njit(cache=True)
def sqeuclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the squared Euclidean distance (L2) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3,) Coordinates of the first point
    y : np.ndarray
        (3,) Coordinates of the second point

    Returns
    -------
    float
        Squared Euclidean distance
    """
    return (y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2


@nb.njit(cache=True)
def chebyshev(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Chebyshev distance (Linf) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3,) Coordinates of the first point
    y : np.ndarray
        (3,) Coordinates of the second point

    Returns
    -------
    float
        Chebyshev distance
    """
    return max(abs(y[0] - x[0]), abs(y[1] - x[1]), abs(y[2] - x[2]))


@nb.njit(cache=True)
def minkowski(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """Compute the Minkowski distance (Lp) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3,) Coordinates of the first point
    y : np.ndarray
        (3,) Coordinates of the second point

    Returns
    -------
    float
        Minkowski distance
    """
    return pow(
        abs(y[0] - x[0]) ** p + abs(y[1] - x[1]) ** p + abs(y[2] - x[2]) ** p, 1.0 / p
    )


@nb.njit(cache=True)
def pdist(
    x: np.ndarray, metric_id: int = METRICS["euclidean"], p: float = 2.0
) -> np.ndarray:
    """Numba implementation of
    `scipy.spatial.distance.pdist(x, metric=metric, p=p)` in 3D.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of point coordinates in the set
    metric_id : int, default 2 (Euclidean)
        Distance metric enumerator
    p : float, default 2.
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    np.ndarray
        (N, N) array of pair-wise Euclidean distances
    """
    # Check on the input
    assert x.shape[1] == 3, "Only supports 3D points for now."

    # Dispatch (faster this way than dispatching at each distance call)
    if metric_id == MINKOWSKI:
        return _pdist_minkowski(x, p)
    elif metric_id == CITYBLOCK:
        return _pdist_cityblock(x)
    elif metric_id == EUCLIDEAN:
        return _pdist_euclidean(x)
    elif metric_id == SQEUCLIDEAN:
        return _pdist_sqeuclidean(x)
    elif metric_id == CHEBYSHEV:
        return _pdist_chebyshev(x)
    else:
        raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _pdist_cityblock(x: np.ndarray) -> np.ndarray:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i, xi in enumerate(x):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = cityblock(xi, x[j])

    return res


@nb.njit(cache=True)
def _pdist_euclidean(x: np.ndarray) -> np.ndarray:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i, xi in enumerate(x):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = euclidean(xi, x[j])

    return res


@nb.njit(cache=True)
def _pdist_sqeuclidean(x: np.ndarray) -> np.ndarray:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i, xi in enumerate(x):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = sqeuclidean(xi, x[j])

    return res


@nb.njit(cache=True)
def _pdist_chebyshev(x: np.ndarray) -> np.ndarray:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i, xi in enumerate(x):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = chebyshev(xi, x[j])

    return res


@nb.njit(cache=True)
def _pdist_minkowski(x: np.ndarray, p: float) -> np.ndarray:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i, xi in enumerate(x):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = minkowski(xi, x[j], p)

    return res


@nb.njit(cache=True)
def cdist(
    x1: np.ndarray,
    x2: np.ndarray,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> np.ndarray:
    """Numba implementation of Euclidean
    `scipy.spatial.distance.cdist(x, metric=p=2)` in 3D.

    Parameters
    ----------
    x1 : np.ndarray
        (N, 3) array of point coordinates in the first set
    x2 : np.ndarray
        (M, 3) array of point coordinates in the second set
    metric_id : int, default 2 (Euclidean)
        Distance metric enumerator
    p : float, default 2.
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    np.ndarray
        (N, M) array of pair-wise Euclidean distances
    """
    # Check on the input
    assert x1.shape[1] == 3 and x2.shape[1] == 3, "Only supports 3D points for now."

    # Dispatch (faster this way than dispatching at each distance call)
    if metric_id == MINKOWSKI:
        return _cdist_minkowski(x1, x2, p)
    elif metric_id == CITYBLOCK:
        return _cdist_cityblock(x1, x2)
    elif metric_id == EUCLIDEAN:
        return _cdist_euclidean(x1, x2)
    elif metric_id == SQEUCLIDEAN:
        return _cdist_sqeuclidean(x1, x2)
    elif metric_id == CHEBYSHEV:
        return _cdist_chebyshev(x1, x2)
    else:
        raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _cdist_cityblock(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1, x1i in enumerate(x1):
        for i2, x2i in enumerate(x2):
            res[i1, i2] = cityblock(x1i, x2i)

    return res


@nb.njit(cache=True)
def _cdist_euclidean(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1, x1i in enumerate(x1):
        for i2, x2i in enumerate(x2):
            res[i1, i2] = euclidean(x1i, x2i)

    return res


@nb.njit(cache=True)
def _cdist_sqeuclidean(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1, x1i in enumerate(x1):
        for i2, x2i in enumerate(x2):
            res[i1, i2] = sqeuclidean(x1i, x2i)

    return res


@nb.njit(cache=True)
def _cdist_chebyshev(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1, x1i in enumerate(x1):
        for i2, x2i in enumerate(x2):
            res[i1, i2] = chebyshev(x1i, x2i)

    return res


@nb.njit(cache=True)
def _cdist_minkowski(x1: np.ndarray, x2: np.ndarray, p: float) -> np.ndarray:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1, x1i in enumerate(x1):
        for i2, x2i in enumerate(x2):
            res[i1, i2] = minkowski(x1i, x2i, p)

    return res


@nb.njit(cache=True)
def farthest_pair(
    x: np.ndarray,
    iterative: bool = False,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> tuple[int, int, float]:
    """Algorithm which finds the two points which are farthest from each other
    in a set, in the Euclidean sense.

    Two algorithms are available:

    - `brute`: computes all pairwise distances and uses `argmax`.
    - `iterative`: repeatedly jumps to the current farthest point until
      convergence. It is not exact, but it is fast.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of point coordinates
    iterative : bool
        If `True`, uses an iterative, fast approximation
    metric_id : int, default 2 (Euclidean)
        Distance metric enumerator
    p : float
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    int
        ID of the first point that makes up the pair
    int
        ID of the second point that makes up the pair
    float
        Distance between the two points
    """
    # To save time, if Euclidean distance is used, use its square
    is_euclidean = False
    if metric_id == EUCLIDEAN:
        is_euclidean = True
        metric_id = SQEUCLIDEAN

    # Dispatch
    if not iterative:
        # Find the distance between every pair of points
        dist_mat = pdist(x, metric_id, p)

        # Select the pair with the farthest distance, fetch indexes
        index = np.argmax(dist_mat)
        i, j = index // len(x), index % len(x)

        # Record farthest distance
        dist = dist_mat[i, j]

    else:
        # Seed the search with the point farthest from the centroid
        centroid = mean(x, 0)
        start_idx = np.argmax(cdist(centroid[None, :], x, metric_id, p))

        # Jump to the farthest point until convergence
        pair_idxs, set_id = [start_idx, start_idx], 0
        dist = -np.inf
        while True:
            previous_dist = dist
            other_id = 1 - set_id

            dists = cdist(x[pair_idxs[set_id]][None, :], x, metric_id, p).flatten()
            farthest_idx = np.argmax(dists)
            farthest_dist = float(dists[farthest_idx])

            if farthest_dist <= previous_dist:
                break

            pair_idxs[other_id] = farthest_idx
            dist = farthest_dist
            set_id = other_id

        # Unroll index
        i, j = pair_idxs

    # If needed, take the square root of the distance
    if is_euclidean:
        dist = np.sqrt(dist)

    return int(i), int(j), float(dist)


@nb.njit(cache=True)
def closest_pair_legacy(
    x1: np.ndarray,
    x2: np.ndarray,
    iterative: bool = False,
    seed: bool = True,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> tuple[int, int, float]:
    """Legacy closest-pair implementation kept for model compatibility.

    This preserves the historical iterative behavior, including the missing
    set switch after each closest-point update. New code should use
    :func:`closest_pair`.
    """
    # To save time, if Euclidean distance is used, use its square
    is_euclidean = False
    if metric_id == EUCLIDEAN:
        is_euclidean = True
        metric_id = SQEUCLIDEAN

    # Find the two points in two sets of points that are closest to each other
    if not iterative:
        # Compute every pair-wise distances between the two sets
        dist_mat = cdist(x1, x2, metric_id, p)

        # Select the closest pair of point, fetch indexes
        index = np.argmin(dist_mat)
        i, j = index // len(x2), index % len(x2)

        # Record closest distance
        dist = dist_mat[i, j]

    else:
        # Pick the point to start iterating from
        xarr = [x1, x2]
        idxs, set_id, dist, tempdist = [0, 0], 0, 1e9, 1e9 + 1.0
        if seed:
            # Find the end points of the two sets
            for i, xi in enumerate(xarr):
                other_id = 1 - i
                seed_idxs = np.array(farthest_pair(xi, True)[:2])
                seed_dists = cdist(xi[seed_idxs], xarr[other_id], metric_id, p)
                seed_argmins = argmin(seed_dists, axis=1)
                seed_mins = np.array(
                    [seed_dists[0][seed_argmins[0]], seed_dists[1][seed_argmins[1]]]
                )
                if np.min(seed_mins) < dist:
                    set_id = other_id
                    seed_choice = int(np.argmin(seed_mins))
                    idxs[i] = int(seed_idxs[seed_choice])
                    idxs[set_id] = int(seed_argmins[seed_choice])
                    dist = float(seed_mins[seed_choice])

        # Historically this loop did not switch `set_id` after updating the
        # closest point in the opposite set. Preserve that behavior here for
        # compatibility with trained models and reference outputs.
        while dist < tempdist:
            tempdist = dist
            other_id = 1 - set_id
            dists = cdist(
                xarr[set_id][idxs[set_id]][None, :], xarr[other_id], metric_id, p
            ).flatten()
            closest_idx = int(np.argmin(dists))
            idxs[other_id] = closest_idx
            dist = float(dists[closest_idx])

        # Unroll index
        i, j = idxs

    # If needed, take the square root of the distance
    if is_euclidean:
        dist = np.sqrt(dist)

    return int(i), int(j), float(dist)


@nb.njit(cache=True)
def closest_pair(
    x1: np.ndarray,
    x2: np.ndarray,
    iterative: bool = False,
    seed: bool = True,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> tuple[int, int, float]:
    """Algorithm which finds the two points which are closest to each other
    from two separate sets.

    Two algorithms are available:

    - `brute`: computes all cross-distances and uses `argmin`.
    - `iterative`: repeatedly jumps to the current closest point until
      convergence. It is not exact, but it is fast.

    Parameters
    ----------
    x1 : np.ndarray
        (N, 3) array of point coordinates in the first set
    x2 : np.ndarray
        (M, 3) array of point coordinates in the second set
    iterative : bool
        If `True`, uses an iterative, fast approximation
    seed : bool
        Whether or not to use the two farthest points in one of the two sets
        to seed the iterative algorithm
    metric_id : int, default 2 (Euclidean)
        Distance metric enumerator
    p : float, default 2.
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    int
        ID of the first point that makes up the pair
    int
        ID of the second point that makes up the pair
    float
        Distance between the two points
    """
    # To save time, if Euclidean distance is used, use its square
    is_euclidean = False
    if metric_id == EUCLIDEAN:
        is_euclidean = True
        metric_id = SQEUCLIDEAN

    # Find the two points in two sets of points that are closest to each other
    if not iterative:
        # Compute every pair-wise distances between the two sets
        dist_mat = cdist(x1, x2, metric_id, p)

        # Select the closest pair of point, fetch indexes
        index = np.argmin(dist_mat)
        i, j = index // len(x2), index % len(x2)

        # Record closest distance
        dist = dist_mat[i, j]

    else:
        # Pick the point to start iterating from
        point_sets = [x1, x2]
        pair_idxs, set_id = [0, 0], 0
        dist = np.inf
        if seed:
            # Find the end points of the two sets
            for i, xi in enumerate(point_sets):
                other_id = 1 - i
                seed_idxs = np.array(farthest_pair(xi, True)[:2])
                seed_dists = cdist(xi[seed_idxs], point_sets[other_id], metric_id, p)
                seed_argmins = argmin(seed_dists, axis=1)
                seed_mins = np.array(
                    [seed_dists[0][seed_argmins[0]], seed_dists[1][seed_argmins[1]]]
                )
                if np.min(seed_mins) < dist:
                    set_id = other_id
                    seed_choice = int(np.argmin(seed_mins))
                    pair_idxs[i] = int(seed_idxs[seed_choice])
                    pair_idxs[set_id] = int(seed_argmins[seed_choice])
                    dist = float(seed_mins[seed_choice])

        # Find the closest point in the other set, repeat until convergence
        while True:
            previous_dist = dist
            other_id = 1 - set_id
            dists = cdist(
                point_sets[set_id][pair_idxs[set_id]][None, :],
                point_sets[other_id],
                metric_id,
                p,
            ).flatten()
            closest_idx = int(np.argmin(dists))
            closest_dist = float(dists[closest_idx])

            if closest_dist >= previous_dist:
                break

            pair_idxs[other_id] = closest_idx
            dist = closest_dist
            set_id = other_id

        # Unroll index
        i, j = pair_idxs

    # If needed, take the square root of the distance
    if is_euclidean:
        dist = np.sqrt(dist)

    return int(i), int(j), float(dist)
