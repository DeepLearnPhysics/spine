"""Scalar and geometric feature extraction for voxel clusters."""

# Feature kernels keep the covariance decomposition explicit and use Numba's
# parallel iterator, which Pylint cannot infer as iterable.
# pylint: disable=duplicate-code,not-an-iterable,too-many-locals

from __future__ import annotations

from collections.abc import Sequence

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import ArrayLike, ClusterLabelBatch, IndexBatch, TensorBatch
from spine.utils.conditional import torch
from spine.utils.jit import numbafy

from .label import get_cluster_label_batch

__all__ = [
    "get_cluster_centers",
    "get_cluster_energies",
    "get_cluster_features",
    "get_cluster_features_base",
    "get_cluster_features_batch",
    "get_cluster_features_extended",
    "get_cluster_sizes",
]


def get_cluster_features_batch(
    data: ClusterLabelBatch | TensorBatch,
    clusts: IndexBatch,
    add_value: bool = False,
    add_shape: bool = False,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_features`.

    Parameters
    ----------
    data : ClusterLabelBatch or TensorBatch
        Structured labels or raw sparse voxel data. Structured labels are
        required when ``add_shape`` is enabled.
    clusts : IndexBatch
        (C) List of cluster indexes
    add_value : bool, default False
        Append mean and RMS voxel-value features.
    add_shape : bool, default False
        Append the semantic shape assigned to each cluster.

    Returns
    -------
    TensorBatch
        (C) List of cluster dE/dx value close to the start points
    """
    if add_shape and not isinstance(data, ClusterLabelBatch):
        raise TypeError(
            "Semantic cluster features require structured cluster labels or "
            "an explicit `extra` feature tensor."
        )

    values = None
    if add_value:
        values = data.values.tensor

    if isinstance(data, ClusterLabelBatch):
        feats = get_cluster_features(data.coords.tensor, clusts.index_list, values)
        if add_shape:
            shapes = get_cluster_label_batch(data, clusts, "shape").data
            if isinstance(feats, torch.Tensor):
                shapes = torch.as_tensor(
                    shapes,
                    dtype=feats.dtype,
                    device=feats.device,
                )
                feats = torch.cat((feats, shapes[:, None]), dim=1)
            else:
                shapes = np.asarray(shapes, dtype=feats.dtype)
                feats = np.concatenate((feats, shapes[:, None]), axis=1)
    else:
        feats = get_cluster_features(data.coords.tensor, clusts.index_list, values)

    return TensorBatch(feats, clusts.counts)


@numbafy(cast_args=["coords"], list_args=["clusts"], keep_torch=True, ref_arg="coords")
def get_cluster_centers(coords: ArrayLike, clusts: Sequence[ArrayLike]) -> ArrayLike:
    """Returns the coordinate of the centroid associated with each cluster.

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C, 3) Tensor of cluster centers
    """
    if len(clusts) == 0:
        return np.empty((0, 3), dtype=coords.dtype)

    return _get_cluster_centers(coords, clusts)


@nb.njit(cache=True)
def _get_cluster_centers(
    coords: np.ndarray, clusts: Sequence[np.ndarray]
) -> np.ndarray:

    centers = np.empty((len(clusts), 3), dtype=coords.dtype)
    for i, c in enumerate(clusts):
        centers[i] = np.sum(coords[c], axis=0) / len(c)

    return centers


@numbafy(cast_args=["data"], list_args=["clusts"])
def get_cluster_sizes(data: ArrayLike, clusts: Sequence[ArrayLike]) -> ArrayLike:
    """Returns the sizes of each cluster.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C) List of cluster sizes
    """
    if len(clusts) == 0:
        return np.empty(0, dtype=np.int64)

    return _get_cluster_sizes(data, clusts)


@nb.njit(cache=True)
def _get_cluster_sizes(_data: np.ndarray, clusts: Sequence[np.ndarray]) -> np.ndarray:

    sizes = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        sizes[i] = len(c)

    return sizes


@numbafy(cast_args=["values"], list_args=["clusts"], keep_torch=True, ref_arg="values")
def get_cluster_energies(values: ArrayLike, clusts: Sequence[ArrayLike]) -> ArrayLike:
    """Returns the total charge/energy deposited by each cluster.

    Parameters
    ----------
    values : np.ndarray
        Value deposited at each voxel.
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C) List of cluster pixel sums
    """
    if len(clusts) == 0:
        return np.empty(0, dtype=values.dtype)

    return _get_cluster_energies(values, clusts)


@nb.njit(cache=True)
def _get_cluster_energies(
    values: np.ndarray, clusts: Sequence[np.ndarray]
) -> np.ndarray:

    energies = np.empty(len(clusts), dtype=values.dtype)
    for i, c in enumerate(clusts):
        energies[i] = np.sum(values[c])

    return energies


def get_cluster_features(
    coords: ArrayLike,
    clusts: Sequence[ArrayLike],
    values: ArrayLike | None = None,
    shapes: ArrayLike | None = None,
) -> ArrayLike:
    """Returns an array of features for each cluster.

    The basic 16 geometric features are composed of:
    - Center (3)
    - Covariance matrix (9)
    - Principal axis (3)
    - Voxel count (1)

    The flag `add_value` adds the following 2 features:
    - Mean energy (1)
    - RMS energy (1)

    The flag `add_shape` adds the particle shape information:
    - Semantic type (1), i.e. most represented type in cluster

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    values : np.ndarray, optional
        Per-voxel values used to append mean and RMS features.
    shapes : np.ndarray, optional
        Per-voxel semantic labels used to append a cluster shape feature.

    Returns
    -------
    np.ndarray
        (C, N_c) Tensor of cluster features
    """
    feats = get_cluster_features_base(coords, clusts)
    if values is not None or shapes is not None:
        feats_ext = get_cluster_features_extended(values, shapes, clusts)
        if isinstance(coords, np.ndarray):
            feats = np.hstack((feats, feats_ext))
        else:
            feats_ext = torch.as_tensor(
                feats_ext, dtype=coords.dtype, device=coords.device
            )
            feats = torch.cat((feats, feats_ext), dim=1)

    return feats


@numbafy(cast_args=["coords"], list_args=["clusts"], keep_torch=True, ref_arg="coords")
def get_cluster_features_base(
    coords: ArrayLike, clusts: Sequence[ArrayLike]
) -> ArrayLike:
    """Returns an array of 16 geometric features for each of cluster.

    The 16 geometric features are composed of:
    - Center (3)
    - Covariance matrix (9)
    - Principal axis (3)
    - Voxel count (1)

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C, 16) Tensor of cluster features
    """
    if len(clusts) == 0:
        return np.empty((0, 16), dtype=coords.dtype)  # Cannot type empty list

    return _get_cluster_features_base(coords, clusts)


@nb.njit(parallel=True, cache=True)
def _get_cluster_features_base(
    coords: np.ndarray, clusts: Sequence[np.ndarray]
) -> np.ndarray:

    # Loop over the clusters (parallelize). The `prange` function creates a
    # uint64 iterator which is cast to int64 to access a list, and throws a
    # warning. To avoid this, use a separate counter to acces clusts.
    feats = np.empty((len(clusts), 16), dtype=coords.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get list of voxels in the cluster
        clust = clusts[ids[k]]
        x = np.ascontiguousarray(coords[clust])

        # Get cluster center
        center = sm.mean(x, 0)

        # Get orientation matrix
        covariance = np.cov(x.T, ddof=len(x) - 1).astype(x.dtype)

        # Center data
        x = x - center

        # Get eigenvectors, normalize orientation matrix and eigenvalues to
        # largest. If points are superimposed, i.e. if the largest eigenvalue
        # != 0, no need to keep going
        # The float64 cast is required by the available LAPACK backend.
        w, v = np.linalg.eigh(covariance.astype(np.float64))
        w, v = w.astype(x.dtype), v.astype(x.dtype)
        if w[2] == 0.0:
            feats[k, :3] = center
            feats[k, 3:15] = 0.0
            feats[k, 15] = len(clust)
            continue
        dirwt = 1.0 - w[1] / w[2]
        normalized_covariance = covariance / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:, 2]

        # Projection all points, x, along the principal axis
        x0 = np.dot(x, v0)

        # Evaluate the distance from the points to the principal axis
        xp0 = x - np.outer(x0, v0)
        np0 = np.empty(len(xp0), dtype=coords.dtype)
        for i, displacement in enumerate(xp0):
            np0[i] = np.linalg.norm(displacement)

        # Flip the principal direction if it is not pointing towards the
        # maximum spread
        sc = np.dot(x0, np0)
        if sc < 0:
            # Numba does not support unary `-`, have to flip manually
            v0 = np.zeros(3, dtype=coords.dtype) - v0

        # Weight direction
        v0 = dirwt * v0

        # Append
        feats[k, :3] = center
        feats[k, 3:12] = normalized_covariance.flatten()
        feats[k, 12:15] = v0
        feats[k, 15] = len(clust)

    return feats


@numbafy(cast_args=["values", "shapes"], list_args=["clusts"])
def get_cluster_features_extended(
    values: ArrayLike | None,
    shapes: ArrayLike | None,
    clusts: Sequence[ArrayLike],
) -> ArrayLike:
    """Returns an array of 3 additional features for each of cluster.

    The flag `add_value` adds the following 2 features:
    - Mean energy (1)
    - RMS energy (1)

    The flag `add_shape` adds the particle shape information:
    - Semantic type (1), i.e. most represented type in cluster

    Parameters
    ----------
    values : np.ndarray, optional
        Value deposited at each voxel.
    shapes : np.ndarray, optional
        Semantic shape associated with each voxel.
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    Returns
    -------
    np.ndarray
        (C, 1/2/3) Tensor of additional cluster features
    """
    if values is None and shapes is None:
        raise ValueError("Provide values, shapes, or both.")
    reference = values if values is not None else shapes
    assert reference is not None
    add_value = values is not None
    add_shape = shapes is not None
    if len(clusts) == 0:
        return np.empty((0, add_value * 2 + add_shape), dtype=reference.dtype)

    if values is None:
        values = np.empty(0, dtype=reference.dtype)
    if shapes is None:
        shapes = np.empty(0, dtype=reference.dtype)
    elif add_value and shapes.dtype != values.dtype:
        # A single output array cannot preserve independent energy and label
        # dtypes. Match the continuous feature dtype used by the base features.
        shapes = shapes.astype(values.dtype)
    return _get_cluster_features_extended(values, shapes, clusts, add_value, add_shape)


@nb.njit(parallel=True, cache=True)
def _get_cluster_features_extended(
    values: np.ndarray,
    shapes: np.ndarray,
    clusts: Sequence[np.ndarray],
    add_value: bool = True,
    add_shape: bool = True,
) -> np.ndarray:
    feats = np.empty((len(clusts), add_value * 2 + add_shape), dtype=values.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get cluster
        clust = clusts[ids[k]]

        # Get mean and RMS energy in the cluster, if requested
        if add_value:
            mean_value = np.mean(values[clust])
            std_value = np.std(values[clust])
            feats[k, :2] = np.array([mean_value, std_value], dtype=values.dtype)

        # Get the cluster semantic class, if requested
        if add_shape:
            types, cnts = sm.unique(shapes[clust])
            major_sem_type = types[np.argmax(cnts)]
            feats[k, -1] = major_sem_type

    return feats
