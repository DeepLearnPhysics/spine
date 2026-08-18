"""Cluster-wise truth-label reduction and point-label extraction."""

# Batched truth-label adapters necessarily coordinate several parallel label
# products, while low-level point extraction preserves its established API.
# pylint: disable=duplicate-code,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals

from __future__ import annotations

from collections.abc import Sequence

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import (
    ArrayLike,
    ClusterLabelBatch,
    IndexBatch,
    TensorBatch,
    TensorSchema,
)
from spine.utils.conditional import torch
from spine.utils.jit import numbafy

__all__ = [
    "get_cluster_closest_label_batch",
    "get_cluster_closest_primary_label_batch",
    "get_cluster_label",
    "get_cluster_label_batch",
    "get_cluster_points_label",
    "get_cluster_points_label_batch",
    "get_cluster_primary_label_batch",
]


def get_cluster_label_batch(
    data: ClusterLabelBatch,
    clusts: IndexBatch,
    column: str = "cluster",
) -> TensorBatch:
    """Batched version of :func:`get_cluster_label`.

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured cluster labels
    clusts : IndexBatch
        (C) List of cluster indexes
    column : str, default 'cluster'
        Named voxel field to reduce within each cluster.

    Returns
    -------
    TensorBatch
        (C) List of individual cluster labels
    """
    values = data.voxel_field(column).to_numpy().data
    if values.ndim == 1:
        labels = get_cluster_label(values, clusts.index_list)
    else:
        labels = np.column_stack(
            [
                get_cluster_label(values[:, index], clusts.index_list)
                for index in range(values.shape[1])
            ]
        )

    return TensorBatch(labels, clusts.counts)


def get_cluster_closest_label_batch(
    data: ClusterLabelBatch,
    coord_label: TensorBatch,
    clusts: IndexBatch,
    labels: TensorBatch,
    default: int | list[int] | np.ndarray,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_closest_label`.

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured cluster labels
    coord_label : TensorBatch
        Batch of particle end points labels
    clusts : IndexBatch
        (C) List of cluster indexes
    labels : TensorBatch
        (C) Existing list of cluster labels (the new labels will be a subset)
    default : Union[int, List[int]]
        Default value to assign to secondary clusters

    Returns
    -------
    TensorBatch
        (C) List of individual cluster labels
    """
    data_np = data.to_numpy()
    coord_np = coord_label.to_numpy()
    labels_np = labels.to_numpy()
    groups = get_cluster_label_batch(data, clusts, "group").to_numpy()
    particle_ids = data.particle_field("particle").to_numpy()
    output = labels_np.data.copy()
    default_values = np.asarray(default)
    for batch_id in range(data.batch_size):
        lower = clusts.edges[batch_id]
        voxels = data_np.coords[batch_id]
        points = coord_np.coordinates("start")[batch_id]
        particle_ids_b = particle_ids[batch_id].astype(np.int64, copy=False)
        event_offset = data_np.data.edges[batch_id]
        for group in np.unique(groups[batch_id].astype(np.int64)):
            indexes = np.where(groups[batch_id] == group)[0] + lower
            particle_index = np.where(particle_ids_b == group)[0]
            if len(particle_index) == 0 or particle_index[0] >= len(points):
                continue
            point = points[particle_index[0]]
            distances = []
            for cluster_index in indexes:
                local = clusts.index_list[cluster_index] - event_offset
                distances.append(np.min(np.linalg.norm(voxels[local] - point, axis=1)))
            closest = indexes[int(np.argmin(distances))]
            label = int(labels_np.data[closest])
            fallback = default_values[label] if 0 <= label < len(default_values) else -1
            output[indexes] = fallback
            output[closest] = label

    return TensorBatch(output, clusts.counts)


def get_cluster_primary_label_batch(
    data: ClusterLabelBatch,
    clusts: IndexBatch,
    column: str,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_primary_label`.

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured cluster labels
    clusts : IndexBatch
        (C) List of cluster indexes
    column : int
        Column in the label tensor which specifies the requested label

    Returns
    -------
    TensorBatch
        (C) List of cluster primary labels
    """
    values = data.voxel_field(column).data
    groups = data.group_ids.data
    particles = data.particle_ids.data
    # This slice preserves the NumPy/Torch backend and dtype. Every entry is
    # overwritten by the majority vote below, so its initial values are unused.
    labels = values[: len(clusts.index_list)] * 0

    for i, clust in enumerate(clusts.index_list):
        group_values = groups[clust]
        unique, counts = (
            torch.unique(group_values, return_counts=True)
            if isinstance(group_values, torch.Tensor)
            else np.unique(group_values, return_counts=True)
        )
        group = unique[counts.argmax()]
        primary = particles[clust] == group
        selected = values[clust][primary] if primary.any() else values[clust]
        unique, counts = (
            torch.unique(selected, return_counts=True)
            if isinstance(selected, torch.Tensor)
            else np.unique(selected, return_counts=True)
        )
        labels[i] = unique[counts.argmax()]

    return TensorBatch(labels, clusts.counts)


def get_cluster_closest_primary_label_batch(
    data: ClusterLabelBatch,
    coord_label: TensorBatch,
    clusts: IndexBatch,
    primary_ids: TensorBatch,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_cloest_primary_label`.

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured cluster labels
    coord_label : TensorBatch
        Batch of particle end points labels
    clusts : IndexBatch
        (C) List of cluster indexes
    primary_ids : TensorBatch
        (C) Existing list of primary IDs (the new labels will be a subset)

    Returns
    -------
    TensorBatch
        (C) List of cluster primary labels
    """
    data_np = data.to_numpy()
    coord_np = coord_label.to_numpy()
    primary_np = primary_ids.to_numpy()
    groups = get_cluster_label_batch(data, clusts, "group").to_numpy()
    particle_ids = data.particle_field("particle").to_numpy()
    output = primary_np.data.copy()
    for batch_id in range(data.batch_size):
        lower, upper = clusts.edges[batch_id : batch_id + 2]
        voxels = data_np.coords[batch_id]
        points = coord_np.coordinates("start")[batch_id]
        particle_ids_b = particle_ids[batch_id].astype(np.int64, copy=False)
        event_offset = data_np.data.edges[batch_id]
        primary_local = np.where(primary_np.data[lower:upper] == 1)[0]
        for group in np.unique(groups.data[lower:upper][primary_local].astype(int)):
            local_nodes = primary_local[
                groups.data[lower:upper][primary_local] == group
            ]
            particle_index = np.where(particle_ids_b == group)[0]
            if len(particle_index) == 0 or particle_index[0] >= len(points):
                continue
            point = points[particle_index[0]]
            indexes = local_nodes + lower
            distances = []
            for cluster_index in indexes:
                local = clusts.index_list[cluster_index] - event_offset
                distances.append(np.min(np.linalg.norm(voxels[local] - point, axis=1)))
            output[indexes] = 0
            output[indexes[int(np.argmin(distances))]] = 1

    return TensorBatch(output, clusts.counts)


def get_cluster_points_label_batch(
    data: ClusterLabelBatch,
    coord_label: TensorBatch,
    clusts: IndexBatch,
    random_order: bool = True,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_points_label`

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured cluster labels
    coord_label : TensorBatch
        Batch of particle end points labels
    clusts : IndexBatch
        (C) List of cluster indexes used to infer label identities
    random_order : bool, default True
        If `True`, randomize the order in which the start en end points of
        a track are stored in the output
    Returns
    -------
    np.ndarray
        (C, 6) Cluster-wise start and end points (in random order if requested)
    """
    data_np = data.to_numpy()
    coord_np = coord_label.to_numpy()
    particle_indexes = data.particle_indexes.to_numpy()
    points = np.empty((len(clusts.index_list), 6), dtype=data_np.dtype)
    starts = coord_np.coordinates("start")
    ends = coord_np.coordinates("end")
    times = coord_np.feature("time")
    for batch_id in range(data.batch_size):
        lower, upper = clusts.edges[batch_id : batch_id + 2]
        event_offset = data_np.data.edges[batch_id]
        event_voxels = data_np.coords[batch_id]
        starts_b = starts[batch_id]
        ends_b = ends[batch_id]
        times_b = times[batch_id]
        for cluster_index in range(lower, upper):
            local = clusts.index_list[cluster_index] - event_offset
            ids = np.unique(particle_indexes[batch_id][local]).astype(np.int64)
            valid_ids = ids[(ids >= 0) & (ids < len(starts_b))]
            if len(valid_ids) == 0:
                raise IndexError("Cluster has no valid particle coordinate label.")
            label_id = valid_ids[np.argmin(times_b[valid_ids])]
            start = starts_b[label_id].copy()
            end = ends_b[label_id].copy()
            if random_order and np.random.choice(2):
                start, end = end, start
            voxels = event_voxels[local]
            point_pair = np.stack((start, end))
            distances = np.linalg.norm(
                point_pair[:, None, :] - voxels[None, :, :], axis=2
            )
            points[cluster_index, :3] = voxels[np.argmin(distances[0])]
            points[cluster_index, 3:] = voxels[np.argmin(distances[1])]
    if not data.is_numpy:
        points = torch.as_tensor(points, dtype=data.dtype, device=data.device)

    return TensorBatch(
        points,
        clusts.counts,
        coord_cols=np.arange(6),
        schema=TensorSchema(coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)}),
    )


@numbafy(cast_args=["values"], list_args=["clusts"])
def get_cluster_label(values: ArrayLike, clusts: Sequence[ArrayLike]) -> ArrayLike:
    """Returns the majority label of each cluster, specified by the
    requested data column of the label tensor.

    Parameters
    ----------
    values : np.ndarray
        Voxel-level labels to reduce by majority vote.
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    Returns
    -------
    np.ndarray
        (C) List of individual cluster labels
    """
    if len(clusts) == 0:
        return np.empty(0, dtype=values.dtype)

    return _get_cluster_label(values, clusts)


@nb.njit(cache=True)
def _get_cluster_label(
    values: np.ndarray,
    clusts: Sequence[np.ndarray],
) -> np.ndarray:

    labels = np.empty(len(clusts), dtype=values.dtype)
    for i, c in enumerate(clusts):
        v, cts = sm.unique(values[c])
        labels[i] = v[np.argmax(cts)]

    return labels


@numbafy(
    cast_args=["coords", "particle_ids", "starts", "ends", "times"],
    list_args=["clusts"],
    keep_torch=True,
    ref_arg="coords",
)
def get_cluster_points_label(
    coords: ArrayLike,
    particle_ids: ArrayLike,
    starts: ArrayLike,
    ends: ArrayLike,
    times: ArrayLike,
    clusts: Sequence[ArrayLike],
    random_order: bool = True,
) -> ArrayLike:
    """Gets label points for each cluster.

    Returns start point of primary shower fragment twice if shower, delta or
    Michel and both end points of tracks if track.

    Parameters
    ----------
    coords : np.ndarray
        Voxel coordinates.
    particle_ids : np.ndarray
        Particle-table index associated with each voxel.
    starts, ends : np.ndarray
        Start and end coordinates associated with each particle.
    times : np.ndarray
        Creation time associated with each particle.
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    random_order : bool, default True
        If `True`, randomize the order in which the start en end points of
        a track are stored in the output
    Returns
    -------
    np.ndarray
        (C, 6) Cluster start and end points (in random order if requested)
    """
    if len(clusts) == 0:
        return np.empty((0, 6), dtype=coords.dtype)

    return _get_cluster_points_label(
        coords, particle_ids, starts, ends, times, clusts, random_order
    )


@nb.njit(cache=True)
def _get_cluster_points_label(
    coords: np.ndarray,
    particle_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    times: np.ndarray,
    clusts: Sequence[np.ndarray],
    random_order: bool = True,
) -> np.ndarray:

    # Get start and end points (one and the same for all but track class)
    points = np.empty((len(clusts), 6), dtype=coords.dtype)
    for i, c in enumerate(clusts):
        # Use the first constituent particle in time.
        part_ids = np.unique(particle_ids[c]).astype(np.int64)
        label_id = -1
        min_time = np.inf
        for part_id in part_ids:
            if part_id < 0 or part_id >= len(starts):
                raise IndexError("Invalid label index for coord_label.")
            time = times[part_id]
            if time < min_time:
                min_time = time
                label_id = part_id

        if label_id < 0 or label_id >= len(starts):
            raise IndexError("Invalid label index for coord_label.")
        start = starts[label_id]
        end = ends[label_id]
        if random_order and np.random.choice(2):
            start, end = end, start

        points[i, :3] = start
        points[i, 3:6] = end

    # Bring the start points to the closest point in the corresponding cluster
    for i, c in enumerate(clusts):
        point_pair = np.empty((2, 3), dtype=coords.dtype)
        point_pair[0] = points[i, :3]
        point_pair[1] = points[i, 3:6]
        dist_mat = sm.distance.cdist(point_pair, coords[c])
        argmins = sm.argmin(dist_mat, axis=1)
        for j, argmin in enumerate(argmins):
            points[i, 3 * j : 3 * (j + 1)] = coords[c[argmin]]

    return points
