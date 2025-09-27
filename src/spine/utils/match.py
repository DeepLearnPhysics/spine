"""Functions to find the best overlaps between point sets."""

import numba as nb
import numpy as np

from spine.math.distance import cdist

__all__ = [
    "overlap_counts",
    "overlap_iou",
    "overlap_weighted_iou",
    "overlap_dice",
    "overlap_weighted_dice",
    "overlap_chamfer",
]


@nb.njit(cache=True, parallel=True)
def overlap_count(
    index_x: nb.types.List(nb.int64[:]), index_y: nb.types.List(nb.int64[:])
) -> nb.int64[:, :]:
    """Computes a set overlap matrix by overlap count.

    Parameters
    ----------
    index_x: nb.types.List[np.ndarray]
        (N) nb.types.List of tensor index, one per object to match
    index_y: nb.types.List[np.ndarray]
        (M) nb.types.List of tensor index, one per object to be matched to

    Returns
    -------
    np.ndarray
        (M, N) Overlap count matrix
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.int64)
    for i in nb.prange(len(index_x)):
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if np.max(px) < np.min(py) or np.max(py) < np.min(px):
                        continue
                    overlap_matrix[i, j] = len(set(px).intersection(set(py)))

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_iou(
    index_x: nb.types.List(nb.int64[:]), index_y: nb.types.List(nb.int64[:])
) -> nb.float32[:, :]:
    """Computes a set overlap matrix by IoU.

    IoU stands for Intersection-over-Union.

    Parameters
    ----------
    index_x: nb.types.List[np.ndarray]
        (N) nb.types.List of tensor index, one per object to match
    index_y: nb.types.List[np.ndarray]
        (M) nb.types.List of tensor index, one per object to be matched to

    Returns
    -------
    np.ndarray
        (M, N) Overlap IoU matrix
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if np.max(px) < np.min(py) or np.max(py) < np.min(px):
                        continue
                    cap = len(set(px).intersection(set(py)))
                    if cap > 0:
                        cup = len(set(px).union(set(py)))
                        overlap_matrix[i, j] = cap / cup

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_weighted_iou(
    index_x: nb.types.List(nb.int64[:]), index_y: nb.types.List(nb.int64[:])
) -> nb.float32[:, :]:
    """Computes a set overlap matrix by IoU, weighted by the set sizes.

    IoU stands for Intersection-over-Union. The weighting scheme is as follows:
    w = (|size_x + size_y| / (|size_x - size_y| + 1).

    Parameters
    ----------
    index_x: nb.types.List[np.ndarray]
        (N) nb.types.List of tensor index, one per object to match
    index_y: nb.types.List[np.ndarray]
        (M) nb.types.List of tensor index, one per object to be matched to

    Returns
    -------
    np.ndarray
        (M, N) Overlap weighted IoU matrix
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if np.max(px) < np.min(py) or np.max(py) < np.min(px):
                        continue
                    cap = len(set(px).intersection(set(py)))
                    if cap > 0:
                        cup = len(set(px).union(set(py)))
                        n, m = px.shape[0], py.shape[0]
                        overlap_matrix[i, j] = (cap / cup) * (n + m) / (1 + abs(n - m))

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_dice(
    index_x: nb.types.List(nb.int64[:]), index_y: nb.types.List(nb.int64[:])
) -> nb.float32[:, :]:
    """Computes a set overlap matrix by Dice coefficient.

    The Dice coefficient corresponds to the 2 times the intersection of two
    sets over the sum of set sizes.

    Parameters
    ----------
    index_x: nb.types.List[np.ndarray]
        (N) nb.types.List of tensor index, one per object to match
    index_y: nb.types.List[np.ndarray]
        (M) nb.types.List of tensor index, one per object to be matched to

    Returns
    -------
    np.ndarray
        (M, N) Overlap weighted IoU matrix
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if np.max(px) < np.min(py) or np.max(py) < np.min(px):
                        continue
                    cap = len(set(px).intersection(set(py)))
                    if cap > 0:
                        cup = len(px) + len(py)
                        n, m = px.shape[0], py.shape[0]
                        overlap_matrix[i, j] = 2.0 * cap / cup

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_weighted_dice(
    index_x: nb.types.List(nb.int64[:]), index_y: nb.types.List(nb.int64[:])
) -> nb.float32[:, :]:
    """Computes a set overlap matrix by Dice coefficient, weighted by the
    set sizes.

    The Dice coefficient corresponds to the 2 times the intersection of two
    sets over the sum of set sizes. The weighting scheme is as follows:
    w = (|size_x + size_y| / (|size_x - size_y| + 1).

    Parameters
    ----------
    index_x: nb.types.List[np.ndarray]
        (N) nb.types.List of tensor index, one per object to match
    index_y: nb.types.List[np.ndarray]
        (M) nb.types.List of tensor index, one per object to be matched to

    Returns
    -------
    np.ndarray
        (M, N) Overlap weighted IoU matrix
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if np.max(px) < np.min(py) or np.max(py) < np.min(px):
                        continue
                    cap = len(set(px).intersection(set(py)))
                    if cap > 0:
                        cup = len(px) + len(py)
                        n, m = px.shape[0], py.shape[0]
                        w = (n + m) / (1 + abs(n - m))
                        overlap_matrix[i, j] = (2.0 * cap / cup) * w

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_chamfer(
    points_x: nb.types.List(nb.int64[:]), points_y: nb.types.List(nb.int64[:])
) -> nb.float32[:, :]:
    """Computes a set overlap matrix by Chamfer distance.

    This function can match two arbitrary points clouds, hence there is no need
    for the two particle lists to share the same underlying voxel sets.

    Parameters
    ----------
    points_x: nb.types.List[np.ndarray]
        (N, 3) nb.types.List of coordinates, one per object to match
    points_y: nb.types.List[np.ndarray]
        (M, 3) nb.types.List of coordinates, one per object to be matched to

    Returns
    -------
    np.ndarray
        (M, N) Chamfer distance matrix

    Notes
    -----
    Unlike the overlap metrics, this metric should be minimized.
    """
    overlap_matrix = np.full(np.inf, (len(points_x), len(points_y)), dtype=np.float32)
    for i in nb.prange(len(points_x)):
        px = points_x[i]
        if len(px):
            for j, py in enumerate(points_y):
                if len(py):
                    # Compute the voxel pairwise distances
                    dist = cdist(px, py)

                    # Compute the average chamfer distance
                    loss_x = np.min(dist, axis=1)
                    loss_y = np.min(dist, axis=0)
                    loss = loss_x.sum() / len(loss_x) + loss_y.sum() / len(loss_y)

                    overlap_matrix[i, j] = loss

    return overlap_matrix
