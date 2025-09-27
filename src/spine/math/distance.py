"""Numba JIT compiled implementation of distance computation routines.

This module is entirely dedicated to 3D points, which is the core representation
of objects targetted by this software package.
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
]

# Available distance metrics (casting is important for numba optimization)
METRICS = {
    "minkowski": np.int64(0),
    "cityblock": np.int64(1),
    "euclidean": np.int64(2),
    "sqeuclidean": np.int64(3),
    "chebyshev": np.int64(4),
}


@nb.njit(cache=True)
def get_metric_id(metric: nb.types.string, p: nb.float32) -> nb.int64:
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
            return np.int64(1)
        elif p == 2.0:
            return np.int64(2)
        else:
            return np.int64(0)
    elif metric == "cityblock":
        return np.int64(1)
    elif metric == "euclidean":
        return np.int64(2)
    elif metric == "sqeuclidean":
        return np.int64(3)
    elif metric == "chebyshev":
        return np.int64(4)
    else:
        raise ValueError(f"Distance metric not recognized: {metric}")


@nb.njit(cache=True)
def cityblock(x: nb.float32[:], y: nb.float32[:]) -> nb.float32:
    """Compute the cityblock distance (L1) between to 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3) Coorinates of the first point
    y : np.ndarray
        (3) Coorinates of the second point

    Returns
    -------
    float
        Cityblock distance
    """
    return abs(y[0] - x[0]) + abs(y[1] - x[1]) + abs(y[2] - x[2])


@nb.njit(cache=True)
def euclidean(x: nb.float32[:], y: nb.float32[:]) -> nb.float32:
    """Compute the Euclidean distance (L2) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3) Coorinates of the first point
    y : np.ndarray
        (3) Coorinates of the second point

    Returns
    -------
    float
        Euclidean distance
    """
    return np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2)


@nb.njit(cache=True)
def sqeuclidean(x: nb.float32[:], y: nb.float32[:]) -> nb.float32:
    """Compute the squared Euclidean distance (L2) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3) Coorinates of the first point
    y : np.ndarray
        (3) Coorinates of the second point

    Returns
    -------
    float
        Squared Euclidean distance
    """
    return (y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2


@nb.njit(cache=True)
def chebyshev(x: nb.float32[:], y: nb.float32[:]) -> nb.float32:
    """Compute the Chebyshev distance (Linf) between to 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3) Coorinates of the first point
    y : np.ndarray
        (3) Coorinates of the second point

    Returns
    -------
    float
        Chebyshev distance
    """
    return max(abs(y[0] - x[0]), abs(y[1] - x[1]), abs(y[2] - x[2]))


@nb.njit(cache=True)
def minkowski(x: nb.float32[:], y: nb.float32[:], p: nb.float32) -> nb.float32:
    """Compute the Minkowski distance (Lp) between two 3D points.

    Parameters
    ----------
    x : np.ndarray
        (3) Coorinates of the first point
    y : np.ndarray
        (3) Coorinates of the second point

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
    x: nb.float32[:, :], metric_id: nb.int64 = METRICS["euclidean"], p: nb.float32 = 2.0
) -> nb.float32[:, :]:
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

    # Dispatch (faster this way than dipatching at each distance call)
    if metric_id == np.int64(0):
        return _pdist_minkowski(x, p)
    elif metric_id == np.int64(1):
        return _pdist_cityblock(x)
    elif metric_id == np.int64(2):
        return _pdist_euclidean(x)
    elif metric_id == np.int64(3):
        return _pdist_sqeuclidean(x)
    elif metric_id == np.int64(4):
        return _pdist_chebyshev(x)
    else:
        raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _pdist_cityblock(x: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i in range(len(x)):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = cityblock(x[i], x[j])

    return res


@nb.njit(cache=True)
def _pdist_euclidean(x: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i in range(len(x)):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = euclidean(x[i], x[j])

    return res


@nb.njit(cache=True)
def _pdist_sqeuclidean(x: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i in range(len(x)):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = sqeuclidean(x[i], x[j])

    return res


@nb.njit(cache=True)
def _pdist_chebyshev(x: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i in range(len(x)):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = chebyshev(x[i], x[j])

    return res


@nb.njit(cache=True)
def _pdist_minkowski(x: nb.float32[:, :], p: nb.float32) -> nb.float32[:, :]:
    res = np.empty((len(x), len(x)), dtype=x.dtype)
    for i in range(len(x)):
        res[i, i] = 0.0
        for j in range(i + 1, len(x)):
            res[i, j] = res[j, i] = minkowski(x[i], x[j], p)

    return res


@nb.njit(cache=True)
def cdist(
    x1: nb.float32[:, :],
    x2: nb.float32[:, :],
    metric_id: nb.int64 = METRICS["euclidean"],
    p: nb.float32 = 2.0,
) -> nb.float32[:, :]:
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

    # Dispatch (faster this way than dipatching at each distance call)
    if metric_id == np.int64(0):
        return _cdist_minkowski(x1, x2, p)
    elif metric_id == np.int64(1):
        return _cdist_cityblock(x1, x2)
    elif metric_id == np.int64(2):
        return _cdist_euclidean(x1, x2)
    elif metric_id == np.int64(3):
        return _cdist_sqeuclidean(x1, x2)
    elif metric_id == np.int64(4):
        return _cdist_chebyshev(x1, x2)
    else:
        raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _cdist_cityblock(x1: nb.float32[:, :], x2: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            res[i1, i2] = cityblock(x1[i1], x2[i2])

    return res


@nb.njit(cache=True)
def _cdist_euclidean(x1: nb.float32[:, :], x2: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            res[i1, i2] = euclidean(x1[i1], x2[i2])

    return res


@nb.njit(cache=True)
def _cdist_sqeuclidean(x1: nb.float32[:, :], x2: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            res[i1, i2] = sqeuclidean(x1[i1], x2[i2])

    return res


@nb.njit(cache=True)
def _cdist_chebyshev(x1: nb.float32[:, :], x2: nb.float32[:, :]) -> nb.float32[:, :]:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            res[i1, i2] = chebyshev(x1[i1], x2[i2])

    return res


@nb.njit(cache=True)
def _cdist_minkowski(
    x1: nb.float32[:, :], x2: nb.float32[:, :], p: nb.float32
) -> nb.float32[:, :]:
    res = np.empty((len(x1), len(x2)), dtype=x1.dtype)
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            res[i1, i2] = minkowski(x1[i1], x2[i2], p)

    return res


@nb.njit(cache=True)
def farthest_pair(
    x: nb.float32[:, :],
    iterative: nb.boolean = False,
    metric_id: nb.int64 = METRICS["euclidean"],
    p: nb.float32 = 2.0,
) -> (nb.int64, nb.int64, nb.float32):
    """Algorithm which finds the two points which are farthest from each other
    in a set, in the Euclidean sense.

    Two algorithms on offer:
    - `brute`: compute pdist, use argmax (exact)
    - `iterative`: Start with the first point in one set, find the farthest
                   point in the other, move to that point, repeat. This
                   algorithm is *not* exact, but a good and very quick proxy.

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
    euclidean = False
    if metric_id == np.int64(2):
        euclidean = True
        metric_id = np.int64(3)

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
        idxs, subidx, dist, tempdist = [start_idx, start_idx], 0, 0.0, -1.0
        while dist > tempdist:
            tempdist = dist
            dists = cdist(x[idxs[subidx]][None, :], x, metric_id, p).flatten()
            idxs[~subidx] = np.argmax(dists)
            dist = dists[idxs[~subidx]]
            subidx = ~subidx

        # Unroll index
        i, j = idxs

    # If needed, take the square root of the distance
    if euclidean:
        dist = np.sqrt(dist)

    return i, j, dist


@nb.njit(cache=True)
def closest_pair(
    x1: nb.float32[:, :],
    x2: nb.float32[:, :],
    iterative: nb.boolean = False,
    seed: nb.boolean = True,
    metric_id: nb.int64 = METRICS["euclidean"],
    p: nb.float32 = 2.0,
) -> (nb.int64, nb.int64, nb.float32):
    """Algorithm which finds the two points which are closest to each other
    from two separate sets.

    Two algorithms on offer:
    - `brute`: compute cdist, use argmin
    - `iterative`: Start with one point in one set, find the closest
                   point in the other set, move to theat point, repeat. This
                   algorithm is *not* exact, but a good and very quick proxy.

    Parameters
    ----------
    x1 : np.ndarray
        (Nx3) array of point coordinates in the first set
    x1 : np.ndarray
        (Nx3) array of point coordinates in the second set
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
    euclidean = False
    if metric_id == np.int64(2):
        euclidean = True
        metric_id = np.int64(3)

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
            for i, x in enumerate(xarr):
                seed_idxs = np.array(farthest_pair(xarr[i], True)[:2])
                seed_dists = cdist(xarr[i][seed_idxs], xarr[~i], metric_id, p)
                seed_argmins = argmin(seed_dists, axis=1)
                seed_mins = np.array(
                    [seed_dists[0][seed_argmins[0]], seed_dists[1][seed_argmins[1]]]
                )
                if np.min(seed_mins) < dist:
                    set_id = ~i
                    seed_choice = np.argmin(seed_mins)
                    idxs[int(~set_id)] = seed_idxs[seed_choice]
                    idxs[int(set_id)] = seed_argmins[seed_choice]
                    dist = seed_mins[seed_choice]

        # Find the closest point in the other set, repeat until convergence
        while dist < tempdist:
            tempdist = dist
            dists = cdist(
                xarr[set_id][idxs[set_id]][None, :], xarr[~set_id], metric_id, p
            ).flatten()
            idxs[~set_id] = np.argmin(dists)
            dist = dists[idxs[~set_id]]
            subidx = ~set_id

        # Unroll index
        i, j = idxs

    # If needed, take the square root of the distance
    if euclidean:
        dist = np.sqrt(dist)

    return i, j, dist
