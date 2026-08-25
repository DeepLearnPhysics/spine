"""Functions to find the best overlaps between point sets."""

from collections.abc import Sequence

import numba as nb
import numpy as np

from spine.math.distance import cdist

__all__ = [
    "overlap_count",
    "overlap_metrics",
    "overlap_iou",
    "overlap_weighted_iou",
    "overlap_dice",
    "overlap_weighted_dice",
    "overlap_chamfer",
]


@nb.njit(cache=True)
def intersection_size_sorted(x: np.ndarray, y: np.ndarray) -> int:
    """Compute the size of the intersection of two sorted unique arrays."""
    i = j = count = 0
    while i < len(x) and j < len(y):
        if x[i] == y[j]:
            count += 1
            i += 1
            j += 1
        elif x[i] < y[j]:
            i += 1
        else:
            j += 1

    return count


@nb.njit(cache=True, parallel=True)
def overlap_count(
    index_x: Sequence[np.ndarray], index_y: Sequence[np.ndarray]
) -> np.ndarray:
    """Computes a set overlap matrix by overlap count.

    Parameters
    ----------
    index_x : Sequence[np.ndarray]
        (N) Sorted unique voxel indexes, one per object to match.
    index_y : Sequence[np.ndarray]
        (M) Sorted unique voxel indexes, one per object to be matched to.

    Returns
    -------
    np.ndarray
        (N, M) Overlap count matrix.
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.int64)
    for i in nb.prange(len(index_x)):  # pylint: disable=not-an-iterable
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if px[-1] < py[0] or py[-1] < px[0]:
                        continue
                    overlap_matrix[i, j] = intersection_size_sorted(px, py)

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_metrics(
    index_x: Sequence[np.ndarray], index_y: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute count, purity, efficiency and IoU overlap matrices.

    The direction of the asymmetric metrics is explicit: ``index_x`` is the
    predicted collection and ``index_y`` is the truth collection. Purity is
    therefore the fraction of a predicted instance owned by a truth instance,
    while efficiency is the fraction of that truth instance recovered by the
    prediction. All metrics are evaluated from a single intersection pass.

    Parameters
    ----------
    index_x : Sequence[np.ndarray]
        (N) Sorted unique voxel indexes, one per predicted instance.
    index_y : Sequence[np.ndarray]
        (M) Sorted unique voxel indexes, one per truth instance.

    Returns
    -------
    np.ndarray
        (N, M) Intersection count matrix.
    np.ndarray
        (N, M) Predicted-instance purity matrix.
    np.ndarray
        (N, M) Truth-instance efficiency matrix.
    np.ndarray
        (N, M) Intersection-over-union matrix.

    Notes
    -----
    Empty or disjoint pairs receive zero for every metric. Input index arrays
    must be sorted and unique for the linear intersection kernel to be valid.
    A caller can identify the majority truth match for prediction ``i`` with
    ``np.argmax(counts[i])`` and read all three qualities at that matrix entry.
    """
    shape = (len(index_x), len(index_y))
    counts = np.zeros(shape, dtype=np.int64)
    purities = np.zeros(shape, dtype=np.float32)
    efficiencies = np.zeros(shape, dtype=np.float32)
    ious = np.zeros(shape, dtype=np.float32)

    # Traverse each pair once, sharing its intersection across all metrics.
    indexes = np.arange(len(index_x), dtype=np.int64)
    for k in nb.prange(len(index_x)):  # pylint: disable=not-an-iterable
        i = indexes[k]
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    # Sorted bounds cheaply reject the common disjoint case.
                    if px[-1] < py[0] or py[-1] < px[0]:
                        continue

                    cap = intersection_size_sorted(px, py)
                    if cap > 0:
                        counts[i, j] = cap
                        purities[i, j] = cap / len(px)
                        efficiencies[i, j] = cap / len(py)
                        ious[i, j] = cap / (len(px) + len(py) - cap)

    return counts, purities, efficiencies, ious


@nb.njit(cache=True, parallel=True)
def overlap_iou(
    index_x: Sequence[np.ndarray], index_y: Sequence[np.ndarray]
) -> np.ndarray:
    """Computes a set overlap matrix by IoU.

    IoU stands for Intersection-over-Union.

    Parameters
    ----------
    index_x : Sequence[np.ndarray]
        (N) Sorted unique voxel indexes, one per object to match.
    index_y : Sequence[np.ndarray]
        (M) Sorted unique voxel indexes, one per object to be matched to.

    Returns
    -------
    np.ndarray
        (N, M) Overlap IoU matrix.
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):  # pylint: disable=not-an-iterable
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if px[-1] < py[0] or py[-1] < px[0]:
                        continue
                    cap = intersection_size_sorted(px, py)
                    if cap > 0:
                        cup = len(px) + len(py) - cap
                        overlap_matrix[i, j] = cap / cup

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_weighted_iou(
    index_x: Sequence[np.ndarray], index_y: Sequence[np.ndarray]
) -> np.ndarray:
    """Computes a set overlap matrix by IoU, weighted by the set sizes.

    IoU stands for Intersection-over-Union. The weighting scheme is as follows:
    `w = abs(size_x + size_y) / (abs(size_x - size_y) + 1)`.

    Parameters
    ----------
    index_x : Sequence[np.ndarray]
        (N) Sorted unique voxel indexes, one per object to match.
    index_y : Sequence[np.ndarray]
        (M) Sorted unique voxel indexes, one per object to be matched to.

    Returns
    -------
    np.ndarray
        (N, M) Weighted IoU matrix.
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):  # pylint: disable=not-an-iterable
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if px[-1] < py[0] or py[-1] < px[0]:
                        continue
                    cap = intersection_size_sorted(px, py)
                    if cap > 0:
                        cup = len(px) + len(py) - cap
                        n, m = px.shape[0], py.shape[0]
                        overlap_matrix[i, j] = (cap / cup) * (n + m) / (1 + abs(n - m))

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_dice(
    index_x: Sequence[np.ndarray], index_y: Sequence[np.ndarray]
) -> np.ndarray:
    """Computes a set overlap matrix by Dice coefficient.

    The Dice coefficient corresponds to the 2 times the intersection of two
    sets over the sum of set sizes.

    Parameters
    ----------
    index_x : Sequence[np.ndarray]
        (N) Sorted unique voxel indexes, one per object to match.
    index_y : Sequence[np.ndarray]
        (M) Sorted unique voxel indexes, one per object to be matched to.

    Returns
    -------
    np.ndarray
        (N, M) Dice coefficient matrix.
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):  # pylint: disable=not-an-iterable
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if px[-1] < py[0] or py[-1] < px[0]:
                        continue
                    cap = intersection_size_sorted(px, py)
                    if cap > 0:
                        denom = len(px) + len(py)
                        overlap_matrix[i, j] = 2.0 * cap / denom

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_weighted_dice(
    index_x: Sequence[np.ndarray], index_y: Sequence[np.ndarray]
) -> np.ndarray:
    """Computes a set overlap matrix by Dice coefficient, weighted by the
    set sizes.

    The Dice coefficient corresponds to the 2 times the intersection of two
    sets over the sum of set sizes. The weighting scheme is as follows:
    `w = abs(size_x + size_y) / (abs(size_x - size_y) + 1)`.

    Parameters
    ----------
    index_x : Sequence[np.ndarray]
        (N) Sorted unique voxel indexes, one per object to match.
    index_y : Sequence[np.ndarray]
        (M) Sorted unique voxel indexes, one per object to be matched to.

    Returns
    -------
    np.ndarray
        (N, M) Weighted Dice coefficient matrix.
    """
    overlap_matrix = np.zeros((len(index_x), len(index_y)), dtype=np.float32)
    for i in nb.prange(len(index_x)):  # pylint: disable=not-an-iterable
        px = index_x[i]
        if len(px):
            for j, py in enumerate(index_y):
                if len(py):
                    if px[-1] < py[0] or py[-1] < px[0]:
                        continue
                    cap = intersection_size_sorted(px, py)
                    if cap > 0:
                        denom = len(px) + len(py)
                        n, m = px.shape[0], py.shape[0]
                        w = (n + m) / (1 + abs(n - m))
                        overlap_matrix[i, j] = (2.0 * cap / denom) * w

    return overlap_matrix


@nb.njit(cache=True, parallel=True)
def overlap_chamfer(
    points_x: Sequence[np.ndarray], points_y: Sequence[np.ndarray]
) -> np.ndarray:
    """Computes a set overlap matrix by Chamfer distance.

    This function can match two arbitrary points clouds, hence there is no need
    for the two particle lists to share the same underlying voxel sets.

    Parameters
    ----------
    points_x : Sequence[np.ndarray]
        (N) Point clouds of shape ``(P_i, 3)``, one per object to match.
    points_y : Sequence[np.ndarray]
        (M) Point clouds of shape ``(P_j, 3)``, one per object to match against.

    Returns
    -------
    np.ndarray
        (N, M) Chamfer distance matrix.

    Notes
    -----
    Unlike the overlap metrics, this metric should be minimized.
    """
    overlap_matrix = np.full((len(points_x), len(points_y)), np.inf, dtype=np.float32)
    for i in nb.prange(len(points_x)):  # pylint: disable=not-an-iterable
        px = points_x[i]
        if len(px):
            for j, py in enumerate(points_y):
                if len(py):
                    # Compute the voxel pairwise distances
                    dist = cdist(px, py)

                    # Reduce each point-cloud direction explicitly because
                    # Numba does not support the ``axis`` argument to min.
                    loss_x = 0.0
                    for k in range(len(px)):
                        loss_x += np.min(dist[k])

                    loss_y = 0.0
                    for k in range(len(py)):
                        loss_y += np.min(dist[:, k])

                    loss = loss_x / len(px) + loss_y / len(py)

                    overlap_matrix[i, j] = loss

    return overlap_matrix
