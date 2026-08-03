"""Module with functions that operate on collections of pixels (clusters).

A cluster is typically represented as a list of row indexes pointing at the
voxels that up the cluster out of a tensor of pixels.
"""

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch, TensorSchema
from spine.utils.conditional import torch
from spine.utils.jit import numbafy


def form_clusters_batch(
    data: ClusterLabelBatch,
    min_size: int = -1,
    column: str = "cluster",
    shapes=None,
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
    default,
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
    default = np.asarray(default)
    for batch_id in range(data.batch_size):
        lower, upper = clusts.edges[batch_id : batch_id + 2]
        voxels = data_np.coords[batch_id]
        points = coord_np.coordinates("start")[batch_id]
        particle_ids_b = particle_ids[batch_id].astype(np.int64, copy=False)
        event_offset = data_np.data.edges[batch_id]
        for group in np.unique(groups[batch_id].astype(np.int64)):
            indexes = np.where(groups[batch_id] == group)[0] + lower
            particle_index = np.where(particle_ids_b == group)[0]
            if not len(particle_index) or particle_index[0] >= len(points):
                continue
            point = points[particle_index[0]]
            distances = []
            for cluster_index in indexes:
                local = clusts.index_list[cluster_index] - event_offset
                distances.append(np.min(np.linalg.norm(voxels[local] - point, axis=1)))
            closest = indexes[int(np.argmin(distances))]
            label = int(labels_np.data[closest])
            fallback = default[label] if 0 <= label < len(default) else -1
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
    labels = (
        values.new_empty(len(clusts.index_list))
        if isinstance(values, torch.Tensor)
        else np.empty(len(clusts.index_list), dtype=values.dtype)
    )
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
            if not len(particle_index) or particle_index[0] >= len(points):
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
            if not len(valid_ids):
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
    max_dist : float, default -1.0
        Neighborhood radius around the point used t compute the dE/dx

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

    if isinstance(data, ClusterLabelBatch):
        values = data.values.tensor if add_value else None
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
        values = data.values.tensor if add_value else None
        feats = get_cluster_features(data.coords.tensor, clusts.index_list, values)

    return TensorBatch(feats, clusts.counts)


def form_clusters(ids, min_size=-1, shapes=None, shape_values=None):
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


def _form_clusters_from_fields(ids, min_size, shapes=None, shape_values=None):
    """Build cluster indexes from explicit ID and optional shape arrays."""
    if isinstance(ids, torch.Tensor):
        selection = torch.arange(len(ids), device=ids.device)
        if shapes is not None:
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
def break_clusters(coords, labels, clusts, eps, metric_id, p):
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
    coords: nb.float32[:, :],
    clusts: nb.types.List(nb.int64[:]),
    eps: nb.float64,
    metric_id: nb.int64,
    p: nb.float64,
) -> nb.int64[:]:
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


@numbafy(cast_args=["values"], list_args=["clusts"])
def get_cluster_label(values, clusts):
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
    values: nb.float64[:],
    clusts: nb.types.List(nb.int64[:]),
) -> nb.float64[:]:

    labels = np.empty(len(clusts), dtype=values.dtype)
    for i, c in enumerate(clusts):
        v, cts = sm.unique(values[c])
        labels[i] = v[np.argmax(cts)]

    return labels


@numbafy(cast_args=["coords"], list_args=["clusts"], keep_torch=True, ref_arg="coords")
def get_cluster_centers(coords, clusts):
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
    coords: nb.float64[:, :], clusts: nb.types.List(nb.int64[:])
) -> nb.float64[:, :]:

    centers = np.empty((len(clusts), 3), dtype=coords.dtype)
    for i, c in enumerate(clusts):
        centers[i] = np.sum(coords[c], axis=0) / len(c)

    return centers


@numbafy(cast_args=["data"], list_args=["clusts"])
def get_cluster_sizes(data, clusts):
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
def _get_cluster_sizes(
    data: nb.float64[:, :], clusts: nb.types.List(nb.int64[:])
) -> nb.int64[:]:

    sizes = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        sizes[i] = len(c)

    return sizes


@numbafy(cast_args=["values"], list_args=["clusts"], keep_torch=True, ref_arg="values")
def get_cluster_energies(values, clusts):
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
    values: nb.float64[:], clusts: nb.types.List(nb.int64[:])
) -> nb.float64[:]:

    energies = np.empty(len(clusts), dtype=values.dtype)
    for i, c in enumerate(clusts):
        energies[i] = np.sum(values[c])

    return energies


def get_cluster_features(coords, clusts, values=None, shapes=None):
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
def get_cluster_features_base(coords, clusts):
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
    coords: nb.float64[:, :], clusts: nb.types.List(nb.int64[:])
) -> nb.float64[:, :]:

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
        A = np.cov(x.T, ddof=len(x) - 1).astype(x.dtype)

        # Center data
        x = x - center

        # Get eigenvectors, normalize orientation matrix and eigenvalues to
        # largest. If points are superimposed, i.e. if the largest eigenvalue
        # != 0, no need to keep going
        # TODO: get rid of casting, this is a complex LAPACK issue currently
        w, v = np.linalg.eigh(A.astype(np.float64))
        w, v = w.astype(x.dtype), v.astype(x.dtype)
        if w[2] == 0.0:
            feats[k, :3] = center
            feats[k, 3:15] = 0.0
            feats[k, 15] = len(clust)
            continue
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:, 2]

        # Projection all points, x, along the principal axis
        x0 = np.dot(x, v0)

        # Evaluate the distance from the points to the principal axis
        xp0 = x - np.outer(x0, v0)
        np0 = np.empty(len(xp0), dtype=coords.dtype)
        for i in range(len(xp0)):
            np0[i] = np.linalg.norm(xp0[i])

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
        feats[k, 3:12] = B.flatten()
        feats[k, 12:15] = v0
        feats[k, 15] = len(clust)

    return feats


@numbafy(cast_args=["values", "shapes"], list_args=["clusts"])
def get_cluster_features_extended(values, shapes, clusts):
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
    add_value = values is not None
    add_shape = shapes is not None
    if len(clusts) == 0:
        return np.empty((0, add_value * 2 + add_shape), dtype=reference.dtype)

    if values is None:
        values = np.empty(0, dtype=reference.dtype)
    if shapes is None:
        shapes = np.empty(0, dtype=reference.dtype)
    return _get_cluster_features_extended(values, shapes, clusts, add_value, add_shape)


@nb.njit(parallel=True, cache=True)
def _get_cluster_features_extended(
    values: nb.float64[:],
    shapes: nb.float64[:],
    clusts: nb.types.List(nb.int64[:]),
    add_value: bool = True,
    add_shape: bool = True,
) -> nb.float64[:, :]:
    dtype = values.dtype if add_value else shapes.dtype
    feats = np.empty((len(clusts), add_value * 2 + add_shape), dtype=dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get cluster
        clust = clusts[ids[k]]

        # Get mean and RMS energy in the cluster, if requested
        if add_value:
            mean_value = np.mean(values[clust])
            std_value = np.std(values[clust])
            feats[k, :2] = np.array([mean_value, std_value], dtype=dtype)

        # Get the cluster semantic class, if requested
        if add_shape:
            types, cnts = sm.unique(shapes[clust])
            major_sem_type = types[np.argmax(cnts)]
            feats[k, -1] = major_sem_type

    return feats


@numbafy(
    cast_args=["coords", "particle_ids", "starts", "ends", "times"],
    list_args=["clusts"],
    keep_torch=True,
    ref_arg="coords",
)
def get_cluster_points_label(
    coords, particle_ids, starts, ends, times, clusts, random_order=True
):
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
    coords: nb.float64[:, :],
    particle_ids: nb.float64[:],
    starts: nb.float64[:, :],
    ends: nb.float64[:, :],
    times: nb.float64[:],
    clusts: nb.types.List(nb.int64[:]),
    random_order: nb.boolean = True,
) -> nb.float64[:, :]:

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


@numbafy(
    cast_args=["coords", "starts"],
    list_args=["clusts"],
    keep_torch=True,
    ref_arg="coords",
)
def get_cluster_directions(coords, starts, clusts, max_dist=-1.0, optimize=False):
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
    voxels: nb.float64[:, :],
    starts: nb.float64[:, :],
    clusts: nb.types.List(nb.int64[:]),
    max_dist: nb.float64 = -1.0,
    optimize: nb.boolean = False,
) -> nb.float64[:, :]:

    dirs = np.empty(starts.shape, starts.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        dirs[k] = cluster_direction(
            voxels[clusts[ids[k]]], starts[k], max_dist, optimize
        )

    return dirs


@nb.njit(cache=True)
def cluster_direction(
    voxels: nb.float64[:, :],
    start: nb.float64[:],
    max_dist: nb.float64 = -1,
    optimize: nb.boolean = False,
) -> nb.float64[:]:
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

    if max_dist > 0:
        dist_mat = sm.distance.cdist(start.reshape(1, -1), voxels).flatten()
        voxels = voxels[dist_mat <= max(max_dist, np.min(dist_mat))]

    # If optimize is set, select the radius by minimizing the transverse spread
    if optimize and len(voxels) > 2:
        # Order the cluster points by increasing distance to the start point
        dist_mat = sm.distance.cdist(start.reshape(1, -1), voxels).flatten()
        order = np.argsort(dist_mat)
        voxels = voxels[order]
        dist_mat = dist_mat[order]

        # Find the PCA relative secondary spread for each point
        labels = -np.ones(len(voxels), dtype=voxels.dtype)
        meank = sm.mean(voxels[:3], 0)
        covk = (np.transpose(voxels[:3] - meank) @ (voxels[:3] - meank)) / 3
        for i in range(2, len(voxels)):
            # Get the eigenvalues, compute relative transverse spread
            # TODO: get rid of casting, this is complex LAPACK issue currently
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
    for i in range(len(voxels)):
        rel_voxels[i] = voxels[i] - start

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
def get_cluster_dedxs(coords, values, starts, clusts, max_dist=-1.0, anchor=False):
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
    voxels: nb.float64[:, :],
    values: nb.float64[:],
    starts: nb.float64[:, :],
    clusts: nb.types.List(nb.int64[:]),
    max_dist: nb.float64 = -1,
    anchor: nb.boolean = False,
) -> nb.float64[:, :]:

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
    voxels: nb.float64[:, :],
    values: nb.float64[:],
    start: nb.float64[:],
    max_dist: nb.float64 = 5.0,
    anchor: nb.boolean = False,
) -> nb.float64[:]:
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
    voxels: nb.float64[:, :],
    values: nb.float64[:],
    start: nb.float64[:],
    start_dir: nb.float64[:],
    max_dist: nb.float64 = 3.0,
    anchor: nb.boolean = False,
) -> nb.float64[:]:
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
    dE = np.sum(values)
    dx = np.max(voxels_proj) - np.min(voxels_proj)

    # Calculate the mean spread from the reco direction (quality metric)
    voxels_sp = voxels - start
    vectors_to_axis = voxels_sp - np.outer(voxels_proj, start_dir)
    spreads = np.sqrt(np.sum(vectors_to_axis**2, axis=1))
    spread = np.sum(spreads) / len(index)

    return dE / dx, dE, dx, spread, len(index)
