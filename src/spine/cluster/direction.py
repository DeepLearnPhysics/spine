"""Cluster direction and local energy-deposition measurements."""

# Numerical kernels intentionally expose their scalar controls directly and
# keep intermediate quantities named alongside the derivation.
# pylint: disable=not-an-iterable,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals

from __future__ import annotations

from collections.abc import Sequence

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import ArrayLike, ClusterLabelBatch, IndexBatch, TensorBatch
from spine.utils.jit import numbafy

__all__ = [
    "cluster_dedx",
    "cluster_dedx_dir",
    "cluster_direction",
    "get_cluster_dedxs",
    "get_cluster_dedxs_batch",
    "get_cluster_directions",
    "get_cluster_directions_batch",
]


def get_cluster_directions_batch(
    data: ClusterLabelBatch | TensorBatch,
    starts: TensorBatch,
    clusts: IndexBatch,
    max_dist: float = -1.0,
    optimize: bool = False,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_directions`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    starts : TensorBatch
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : IndexBatch
        (C) List of cluster indexes
    max_dist : float, default -1.0
        Neighborhood radius around the point used to estimate the direction
    optimize : bool, default False
        If `True`, the neighborhood radius is optimized on the fly for
        each cluster.

    Returns
    -------
    TensorBatch
        (C, 3) List of cluster directions
    """
    dirs = get_cluster_directions(
        data.coords.tensor, starts.tensor, clusts.index_list, max_dist, optimize
    )

    return TensorBatch(dirs, clusts.counts)


def get_cluster_dedxs_batch(
    data: ClusterLabelBatch | TensorBatch,
    starts: TensorBatch,
    clusts: IndexBatch,
    max_dist: float = -1.0,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_dedxs`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    starts : TensorBatch
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : IndexBatch
        (C) List of cluster indexes
    max_dist : float, default -1.0
        Neighborhood radius around the point used t compute the dE/dx

    Returns
    -------
    TensorBatch
        (C) List of cluster dE/dx value close to the start points
    """
    dedxs = get_cluster_dedxs(
        data.coords.tensor,
        data.values.tensor,
        starts.tensor,
        clusts.index_list,
        max_dist,
    )

    return TensorBatch(dedxs, clusts.counts)


@numbafy(
    cast_args=["coords", "starts"],
    list_args=["clusts"],
    keep_torch=True,
    ref_arg="coords",
)
def get_cluster_directions(
    coords: ArrayLike,
    starts: ArrayLike,
    clusts: Sequence[ArrayLike],
    max_dist: float = -1.0,
    optimize: bool = False,
) -> ArrayLike:
    """Estimates the direction of each cluster.

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    starts : np.ndarray
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    max_dist : float, default -1.0
        Neighborhood radius around the point used to estimate the direction
    optimize : bool, default False
        If `True`, the neighborhood radius is optimized on the fly for
        each cluster.

    Returns
    -------
    torch.tensor:
        (C, 3) Direction vector of each cluster, with the same dtype as
        ``starts``
    """
    if len(clusts) == 0:
        return np.empty(starts.shape, dtype=starts.dtype)
    if coords.shape[1] != 3:
        raise ValueError("Cluster directions require three-dimensional coordinates.")
    return _get_cluster_directions(coords, starts, clusts, max_dist, optimize)


@nb.njit(parallel=True, cache=True)
def _get_cluster_directions(
    voxels: np.ndarray,
    starts: np.ndarray,
    clusts: Sequence[np.ndarray],
    max_dist: float = -1.0,
    optimize: bool = False,
) -> np.ndarray:

    dirs = np.empty(starts.shape, starts.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        dirs[k] = cluster_direction(
            voxels[clusts[ids[k]]], starts[k], max_dist, optimize
        )

    return dirs


@nb.njit(cache=True)
def cluster_direction(
    voxels: np.ndarray,
    start: np.ndarray,
    max_dist: float = -1.0,
    optimize: bool = False,
) -> np.ndarray:
    """Estimates the orientation of a cluster.

    It follows the following procedure:
    - By default, it takes the normalized mean direction from the cluster
      start point to the cluster voxels
    - If `max_dist` is specified, it restricts the cluster voxels
      to those within a `max_dist` radius from the start point
    - If `optimize` is True, it selects the neighborhood which
      minimizes the transverse spread w.r.t. the direction

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates
    starts : np.ndarray
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    max_dist : float, default -1.0
        Neighborhood radius around the point used to estimate the direction
    optimize : bool, default False
        If `True`, the neighborhood radius is optimized on the fly for
        each cluster.

    Returns
    -------
    np.ndarray
        (3) Direction vector
    """
    # If max_dist is set, limit the set of voxels to those within a sphere
    # of radius max_dist
    if voxels.shape[1] != 3:
        raise ValueError("Input must contain three-dimensional coordinates.")

    # ``start`` may be a non-contiguous view of a batched point array. Numba
    # requires a contiguous view before reshaping it into a one-row matrix.
    start_row = np.ascontiguousarray(start).reshape(1, -1)
    if max_dist > 0:
        dist_mat = sm.distance.cdist(start_row, voxels).flatten()
        voxels = voxels[dist_mat <= max(max_dist, np.min(dist_mat))]

    # If optimize is set, select the radius by minimizing the transverse spread
    if optimize and len(voxels) > 2:
        # Order the cluster points by increasing distance to the start point
        dist_mat = sm.distance.cdist(start_row, voxels).flatten()
        order = np.argsort(dist_mat)
        voxels = voxels[order]
        dist_mat = dist_mat[order]

        # Find the PCA relative secondary spread for each point
        labels = -np.ones(len(voxels), dtype=voxels.dtype)
        meank = sm.mean(voxels[:3], 0)
        covk = (np.transpose(voxels[:3] - meank) @ (voxels[:3] - meank)) / 3
        for i in range(2, len(voxels)):
            # Get the eigenvalues, compute relative transverse spread
            # The float64 cast is required by the available LAPACK backend.
            w = np.linalg.eigvalsh(covk.astype(np.float64)).astype(voxels.dtype)
            labels[i] = (
                np.sqrt(w[2] / (w[0] + w[1])) if (w[0] + w[1]) / w[2] > 1e-6 else 0.0
            )

            # If the value is the same as the previous, choose this one
            if labels[i] == labels[i - 1]:
                labels[i - 1] = -1.0

            # Increment mean and matrix
            if i != len(voxels) - 1:
                meank = ((i + 1) * meank + voxels[i + 1]) / (i + 2)
                covk = (i + 1) * covk / (i + 2) + (voxels[i + 1] - meank).reshape(
                    -1, 1
                ) * (voxels[i + 1] - meank) / (i + 1)

        # Subselect voxels that are most track-like
        max_id = np.argmax(labels)
        voxels = voxels[: max_id + 1]

    # If no voxels were selected, return dummy value
    if len(voxels) == 0 or (len(voxels) == 1 and np.all(voxels[0] == start)):
        return np.array([1.0, 0.0, 0.0], dtype=voxels.dtype)

    # Compute mean direction with respect to start point, normalize it
    rel_voxels = np.empty((len(voxels), 3), dtype=voxels.dtype)
    for i, voxel in enumerate(voxels):
        rel_voxels[i] = voxel - start

    mean = sm.mean(rel_voxels, 0)
    norm = np.sqrt(np.sum(mean**2))
    if norm:
        return mean / norm

    return mean


@numbafy(
    cast_args=["coords", "values", "starts"],
    list_args=["clusts"],
    keep_torch=True,
    ref_arg="coords",
)
def get_cluster_dedxs(
    coords: ArrayLike,
    values: ArrayLike,
    starts: ArrayLike,
    clusts: Sequence[ArrayLike],
    max_dist: float = -1.0,
    anchor: bool = False,
) -> ArrayLike:
    """Computes the initial local dE/dxs of each cluster.

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    values : np.ndarray
        Value deposited at each voxel.
    starts : np.ndarray
        (C, 3) Start points w.r.t. which to estimate the local dE/dxs
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    max_dist : float, default -1.0
        Neighborhood radius around the point used to compute the dE/dx
    anchor : bool, default False
        If true, anchor the start point to the closest cluster point

    Returns
    -------
    np.ndarray
        (C) Local dE/dx values for each cluster
    """
    if len(clusts) == 0:
        return np.empty(0, dtype=coords.dtype)

    return _get_cluster_dedxs(
        coords,
        values,
        starts,
        clusts,
        max_dist,
        anchor,
    )


@nb.njit(parallel=True, cache=True)
def _get_cluster_dedxs(
    voxels: np.ndarray,
    values: np.ndarray,
    starts: np.ndarray,
    clusts: Sequence[np.ndarray],
    max_dist: float = -1.0,
    anchor: bool = False,
) -> np.ndarray:

    dedxs = np.empty(len(clusts), voxels.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        dedxs[k] = cluster_dedx(
            voxels[clusts[ids[k]]],
            values[clusts[ids[k]]],
            starts[k],
            max_dist,
            anchor,
        )

    return dedxs


@nb.njit(cache=True)
def cluster_dedx(
    voxels: np.ndarray,
    values: np.ndarray,
    start: np.ndarray,
    max_dist: float = 5.0,
    anchor: bool = False,
) -> float:
    """Computes the initial local dE/dx of a cluster.

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates
    values : np.ndarray
        (N) Voxel values
    start : np.ndarray
        (3) Start point w.r.t. which to compute the local dE/dx
    max_dist : float, default 5.0
        Neighborhood radius around the point used to compute the dE/dx
    anchor : bool, default False
        If true, anchor the start point to the closest cluster point

    Returns
    -------
    float
        Local dE/dx value around the start point
    """
    # Sanity check
    if voxels.shape[1] != 3:
        raise ValueError("Input must contain three-dimensional coordinates.")

    start = start.astype(voxels.dtype)

    # If necessary, anchor start point to the closest cluster point
    if anchor:
        dists = sm.distance.cdist(start.reshape(1, -1), voxels).flatten()
        start = voxels[np.argmin(dists)].astype(start.dtype)  # Dirty

    # If max_dist is set, limit the set of voxels to those within a sphere of
    # radius max_dist around the start point
    dists = sm.distance.cdist(start.reshape(1, -1), voxels).flatten()
    if max_dist > 0.0:
        index = np.where(dists <= max_dist)[0]
        if len(index) < 2:
            return 0.0

        values, dists = values[index], dists[index]

    # Compute the total energy in the neighborhood and the max distance, return ratio
    if np.max(dists) == 0.0:
        return 0.0

    return np.sum(values) / np.max(dists)


@nb.njit(cache=True)
def cluster_dedx_dir(
    voxels: np.ndarray,
    values: np.ndarray,
    start: np.ndarray,
    start_dir: np.ndarray,
    max_dist: float = 3.0,
    anchor: bool = False,
) -> tuple[float, float, float, float, int]:
    """Computes the initial local dE/dx of a cluster by leveraging an
    existing cluster direction estimate.

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates
    values : np.ndarray
        (N) Voxel values
    start : np.ndarray
        (3) Start point w.r.t. which to compute the local dE/dx
    start_dir : np.ndarray
        (3) Start direction of the cluster
    max_dist : float, default 5.0
        Neighborhood radius around the point used to compute the dE/dx

    Returns
    -------
    float
        Local dE/dx value around the start point
    float
        Energy deposited around the start point (dE)
    float
        Length around the start point (dx)
    float
        Relative spread around the cluster direction (quality metric)
    int
        Number of voxels in the neighborhood around the start poin
    """
    # Sanity check
    if voxels.shape[1] != 3:
        raise ValueError("Input must contain three-dimensional coordinates.")

    start = start.astype(voxels.dtype)

    # If necessary, anchor start point to the closest cluster point
    if anchor:
        dists = sm.distance.cdist(start.reshape(1, -1), voxels).flatten()
        start = voxels[np.argmin(dists)].astype(start.dtype)  # Dirty

    # If max_dist is set, limit the set of voxels to those within a sphere of
    # radius max_dist around the start point
    dists = sm.distance.cdist(start.reshape(1, -1), voxels).flatten()
    if max_dist > 0.0:
        index = np.where(dists <= max_dist)[0]
        if len(index) < 2:
            return 0.0, 0.0, 0.0, -1.0, len(index)

        voxels, values, dists = voxels[index], values[index], dists[index]

    # Project the voxels within the sphere onto the reco direction
    voxels_proj = np.dot(voxels - start, start_dir)

    # Mask the voxels to only include the top hemisphere
    index = np.where((voxels_proj >= -1e-3) & (voxels_proj <= max_dist))[0]
    if len(index) < 2:
        return 0.0, 0.0, 0.0, -1.0, len(index)

    voxels, voxels_proj, values = voxels[index], voxels_proj[index], values[index]

    # Compute length as the length along the start direction
    energy = np.sum(values)
    dx = np.max(voxels_proj) - np.min(voxels_proj)

    # Calculate the mean spread from the reco direction (quality metric)
    voxels_sp = voxels - start
    vectors_to_axis = voxels_sp - np.outer(voxels_proj, start_dir)
    spreads = np.sqrt(np.sum(vectors_to_axis**2, axis=1))
    spread = np.sum(spreads) / len(index)

    return energy / dx, energy, dx, spread, len(index)
