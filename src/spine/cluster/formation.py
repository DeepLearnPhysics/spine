"""Cluster formation and connected-component refinement operations."""

# These kernels mirror DBSCAN's scalar configuration and retain intermediate
# arrays explicitly for both NumPy and Torch backends.
# pylint: disable=not-an-iterable,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals

from __future__ import annotations

from collections.abc import Sequence

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import ArrayLike, ClusterLabelBatch, IndexBatch
from spine.utils.conditional import torch
from spine.utils.jit import numbafy

__all__ = ["break_clusters", "form_clusters", "form_clusters_batch"]


def form_clusters_batch(
    data: ClusterLabelBatch,
    min_size: int = -1,
    column: str = "cluster",
    shapes: Sequence[int] | None = None,
) -> IndexBatch:
    """Batched version of :func:`form_clusters`.

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured cluster labels
    min_size : int, default -1
        Minimum size of a cluster to be included in the list
    column : str, default 'cluster'
        Named field used to fetch the voxel cluster IDs.
    shapes : List[int], optional
        List of semantic classes to include in the list of cluster

    Returns
    -------
    IndexBatch
        Object used to index clusters within a batch of data
    """
    # Resolve named association/particle fields before clustering
    ids = data.voxel_field(column)
    shape_values = data.shapes if shapes is not None else None
    clusts, counts, single_counts = [], [], []
    for batch_id in range(data.batch_size):
        ids_b = ids[batch_id]
        shape_b = None if shape_values is None else shape_values[batch_id]
        clusts_b, sizes_b = _form_clusters_from_fields(ids_b, min_size, shapes, shape_b)
        for i, clust in enumerate(clusts_b):
            clusts_b[i] = clust + data.data.edges[batch_id]
        clusts.extend(clusts_b)
        counts.append(len(sizes_b))
        single_counts.extend(sizes_b)

    return IndexBatch(clusts, data.counts, counts, single_counts)


def form_clusters(
    ids: ArrayLike,
    min_size: int = -1,
    shapes: Sequence[int] | None = None,
    shape_values: ArrayLike | None = None,
) -> tuple[list[ArrayLike], np.ndarray]:
    """Builds a list of indexes corresponding to each cluster in the event.

    The `data` tensor should only contain one entry.

    Parameters
    ----------
    ids : Union[np.ndarray, torch.Tensor]
        Cluster identifier associated with each voxel.
    min_size : int, default -1
        Minimum size of a cluster to be included in the list
    shapes : List[int], optional
        List of semantic classes to include in the list of cluster
    shape_values : Union[np.ndarray, torch.Tensor], optional
        Semantic class associated with each voxel. Required with ``shapes``.

    Returns
    -------
    List[Union[np.ndarray, torch.Tensor]]
        (C) List of arrays of voxel indexes in each cluster
    np.ndarray
        (C) Number of pixels in the mask for each cluster
    """
    if getattr(ids, "ndim", 1) == 2 and ids.shape[1] == 1:
        ids = ids[:, 0]
    if shapes is not None and shape_values is None:
        raise ValueError("`shape_values` is required when filtering by shape.")
    return _form_clusters_from_fields(ids, min_size, shapes, shape_values)


def _form_clusters_from_fields(
    ids: ArrayLike,
    min_size: int,
    shapes: Sequence[int] | None = None,
    shape_values: ArrayLike | None = None,
) -> tuple[list[ArrayLike], np.ndarray]:
    """Build cluster indexes from explicit ID and optional shape arrays."""
    if isinstance(ids, torch.Tensor):
        selection = torch.arange(len(ids), device=ids.device)
        if shapes is not None:
            assert shape_values is not None
            requested = torch.as_tensor(
                shapes, dtype=shape_values.dtype, device=ids.device
            )
            mask = torch.any(shape_values[:, None] == requested[None, :], dim=1)
            selection = selection[mask]
            ids = ids[mask]
        unique, counts = torch.unique(ids, return_counts=True)
        order = torch.argsort(ids, stable=True)
        order = selection[order]
        valid = torch.where((counts >= min_size) & (unique > -1))[0].cpu().numpy()
        sizes = counts.detach().cpu().numpy()
        clusters = torch.tensor_split(order, tuple(np.cumsum(sizes)[:-1]))
    else:
        selection = np.arange(len(ids))
        if shapes is not None:
            assert shape_values is not None
            mask = np.any(shape_values[:, None] == np.asarray(shapes)[None, :], axis=1)
            selection = selection[mask]
            ids = ids[mask]
        unique, counts = np.unique(ids, return_counts=True)
        order = selection[np.argsort(ids, stable=True)]
        valid = np.where((counts >= min_size) & (unique > -1))[0]
        sizes = counts
        clusters = list(np.split(order, tuple(np.cumsum(sizes)[:-1])))
    return [clusters[i] for i in valid], sizes[valid]


@numbafy(
    cast_args=["coords", "labels"],
    list_args=["clusts"],
    keep_torch=True,
    ref_arg="coords",
)
def break_clusters(
    coords: ArrayLike,
    labels: ArrayLike,
    clusts: Sequence[ArrayLike],
    eps: float,
    metric_id: int,
    p: float,
) -> ArrayLike:
    """Runs DBSCAN on each invididual cluster to segment them further if needed.

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    labels : np.ndarray
        Cluster identifier associated with each voxel.
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    eps : float
        DBSCAN clustering distance scale
    metric_id : int
        DBSCAN clustering distance metric enumerator
    p : float
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    np.ndarray
        New array of broken cluster labels
    """
    if len(clusts) == 0:
        return np.copy(labels)

    # Break labels
    break_labels = _break_clusters(coords, clusts, eps, metric_id, p)

    # Offset individual broken labels to prevent overlap
    labels = np.copy(labels)
    offset = np.max(labels) + 1
    for clust in clusts:
        # Update IDs, offset
        ids = break_labels[clust]
        labels[clust] = offset + ids
        offset += len(np.unique(ids))

    return labels


@nb.njit(cache=True, parallel=True, nogil=True)
def _break_clusters(
    coords: np.ndarray,
    clusts: Sequence[np.ndarray],
    eps: float,
    metric_id: int,
    p: float,
) -> np.ndarray:
    # Loop over clusters to break, run DBSCAN
    break_labels = np.full(len(coords), -1, dtype=coords.dtype)
    for k in nb.prange(len(clusts)):
        # Restrict the points to those in the cluster
        clust = clusts[k]
        points_c = coords[clust]

        # Run DBSCAN on the cluster, update labels
        clust_ids = sm.cluster.dbscan(points_c, eps=eps, metric_id=metric_id, p=p)

        # Store the breaking IDs
        break_labels[clust] = clust_ids

    return break_labels
