"""Module with functions that operate on collections of pixels (clusters).

A cluster is typically represented as a list of row indexes pointing at the
voxels that up the cluster out of a tensor of pixels.
"""

import numpy as np
import numba as nb
import torch
from typing import List

from spine.data import TensorBatch, IndexBatch

from spine.utils.decorators import numbafy
from spine.utils.globals import (
        BATCH_COL, COORD_COLS, VALUE_COL, CLUST_COL, PART_COL, GROUP_COL,
        MOM_COL, SHAPE_COL, COORD_START_COLS, COORD_END_COLS, COORD_TIME_COL)
import spine.utils.numba_local as nbl


def form_clusters_batch(data, min_size=-1, column=CLUST_COL, shapes=None,
                        batch_size=None):
    """Batched version of :func:`form_clusters`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    min_size : int, default -1
        Minimum size of a cluster to be included in the list
    column : int, default CLUST_COL
        Column of the label tensor to use to fetch the pixel cluster IDs
    shapes : List[int], optional
        List of semantic classes to include in the list of cluster

    Returns
    -------
    IndexBatch
        Object used to index clusters within a batch of data
    """
    # Loop over the individual entries
    clusts, counts, single_counts, offsets = [], [], [], [0]
    for b in range(data.batch_size):
        # Get the list of clusters and cluster sizes within this entry
        data_b = data[b]
        clusts_b, counts_b = form_clusters(data_b, min_size, column, shapes)

        # Offset the cluster indexes appropriately
        for i in range(len(clusts_b)):
            clusts_b[i] += offsets[-1]

        # Append
        clusts.extend(clusts_b)
        counts.append(len(counts_b))
        single_counts.extend(counts_b)
        if b < (data.batch_size - 1):
            offsets.append(offsets[-1] + len(data_b))

    # Make an IndexBatch out of the list
    return IndexBatch(
            clusts, offsets, counts, single_counts, is_numpy=data.is_numpy)


def get_cluster_label_batch(data, clusts, column=CLUST_COL):
    """Batched version of :func:`get_cluster_label`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    clusts : IndexBatch
        (C) List of cluster indexes
    column : int, default CLUST_COL
        Column in the label tensor which specifies the requested label

    Returns
    -------
    TensorBatch
        (C) List of individual cluster labels
    """
    labels = get_cluster_label(data.tensor, clusts.index_list, column)

    return TensorBatch(labels, clusts.counts)


def get_cluster_primary_label_batch(data, clusts, column):
    """Batched version of :func:`get_cluster_primary_label`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    clusts : IndexBatch
        (C) List of cluster indexes
    column : int
        Column in the label tensor which specifies the requested label

    Returns
    -------
    TensorBatch
        (C) List of cluster primary labels
    """
    labels = get_cluster_primary_label(data.tensor, clusts.index_list, column)

    return TensorBatch(labels, clusts.counts)


def get_cluster_closest_primary_label_batch(data, coord_label, clusts, primary_ids):
    """Batched version of :func:`get_cluster_cloest_primary_label`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
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
    labels = np.empty(len(clusts.index_list), dtype=np.int64)
    for b in range(data.batch_size):
        lower, upper = clusts.edges[b], clusts.edges[b+1]
        labels[lower:upper] = get_cluster_closest_primary_label(
                data[b], coord_label[b], clusts[b], primary_ids[b])

    return TensorBatch(labels, clusts.counts)


def get_cluster_points_label_batch(data, coord_label, clusts,
                                   random_order=True):
    """Batched version of :func:`get_cluster_points_label`

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    coord_label : TensorBatch
        Batch of particle end points labels
    clusts : IndexBatch
        (C) List of cluster indexes
    random_order : bool, default True
        If `True`, randomize the order in which the start en end points of
        a track are stored in the output

    Returns
    -------
    np.ndarray
        (C, 6) Cluster-wise start and end points (in random order if requested)
    """
    num_clusts = len(clusts.index_list)
    if isinstance(data.tensor, torch.Tensor):
        points = torch.empty(
                (num_clusts, 6), dtype=data.dtype, device=data.device)
    else:
        points = np.empty((num_clusts, 6), dtype=data.dtype)

    for b in range(data.batch_size):
        lower, upper = clusts.edges[b], clusts.edges[b+1]
        points[lower:upper] = get_cluster_points_label(
                data[b], coord_label[b], clusts[b], random_order)

    return TensorBatch(points, clusts.counts, coord_cols=points.shape[1])


def get_cluster_directions_batch(data, starts, clusts,
                                 max_dist=-1, optimize=False):
    """Batched version of :func:`get_cluster_directions`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    starts : TensorBatch
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : IndexBatch
        (C) List of cluster indexes
    max_dist : float, default -1
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
            data.tensor, starts.tensor, clusts.index_list)

    return TensorBatch(dirs, clusts.counts)


def get_cluster_dedxs_batch(data, starts, clusts, max_dist=-1):
    """Batched version of :func:`get_cluster_dedxs`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    starts : TensorBatch
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : IndexBatch
        (C) List of cluster indexes
    max_dist : float, default -1
        Neighborhood radius around the point used t compute the dE/dx

    Returns
    -------
    TensorBatch
        (C) List of cluster dE/dx value close to the start points
    """
    dedxs = get_cluster_dedxs(
            data.tensor, starts.tensor, clusts.index_list)

    return TensorBatch(dedxs, clusts.counts)


def get_cluster_features_batch(data, clusts, add_value=False, add_shape=False):
    """Batched version of :func:`get_cluster_features`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    starts : TensorBatch
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : IndexBatch
        (C) List of cluster indexes
    max_dist : float, default -1
        Neighborhood radius around the point used t compute the dE/dx

    Returns
    -------
    TensorBatch
        (C) List of cluster dE/dx value close to the start points
    """
    feats = get_cluster_features(
            data.tensor, clusts.index_list, add_value, add_shape)

    return TensorBatch(feats, clusts.counts)


def form_clusters(data, min_size=-1, column=CLUST_COL, shapes=None):
    """Builds a list of indexes corresponding to each cluster in the event.

    The `data` tensor should only contain one entry.

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        Cluster label data tensor
    min_size : int, default -1
        Minimum size of a cluster to be included in the list
    column : int, default CLUST_COL
        Column of the label tensor to use to fetch the pixel cluster IDs
    shapes : List[int], optional
        List of semantic classes to include in the list of cluster

    Returns
    -------
    List[Union[np.ndarray, torch.Tensor]]
        (C) List of arrays of voxel indexes in each cluster
    List[int]
        (C) Number of pixels in the mask for each cluster
    """
    # Fetch the right functions depending on input type
    if isinstance(data, torch.Tensor):
        zeros = lambda x: torch.zeros(x, dtype=torch.bool, device=data.device)
        where, unique = torch.where, torch.unique
    else:
        zeros = lambda x: np.zeros(x, dtype=bool)
        where, unique = np.where, np.unique

    # If requested, restrict data to a specific set of semantic classes
    if shapes is not None:
        mask = zeros(len(data))
        for s in shapes:
            mask |= (data[:, SHAPE_COL] == s)
        mask = where(mask)[0]
        data = data[mask]

    # Get the clusters in this entry
    clust_ids = data[:, column]
    clusts, counts = [], []
    for c in unique(clust_ids):
        # Skip if the cluster ID is invalid
        if c < 0:
            continue
        clust = where(clust_ids == c)[0]

        # Skip if the cluster size is below threshold
        if len(clust) < min_size:
            continue

        # If a mask was applied, get the appropriate IDs
        if shapes is not None:
            clust = mask[clust]

        clusts.append(clust)
        counts.append(len(clust))

    return clusts, counts


@numbafy(cast_args=['data'], list_args=['clusts'])
def get_cluster_label(data, clusts, column=CLUST_COL):
    """Returns the majority label of each cluster, specified by the
    requested data column of the label tensor.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    column : int, default CLUST_COL
        Column in the label tensor which specifies the requested label

    Returns
    -------
    np.ndarray
        (C) List of individual cluster labels
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_label(data, clusts, column)

@nb.njit(cache=True)
def _get_cluster_label(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       column: nb.int64 = CLUST_COL) -> nb.float64[:]:

    labels = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        v, cts = nbl.unique(data[c, column])
        labels[i] = v[np.argmax(cts)]

    return labels


@numbafy(cast_args=['data'], list_args=['clusts'])
def get_cluster_primary_label(data, clusts, column):
    """Returns the majority label of the primary cluster of the group each
    cluster belongs to, specified in the requested data column of the label
    tensor.

    The primary component is identified by picking the set of label voxels
    that have a `PART_COL` identical to the cluster `GROUP_COL`.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    column : int
        Column in the label tensor which specifies the requested label

    Returns
    -------
    np.ndarray
        (C) List of cluster primary labels
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_primary_label(data, clusts, column)

@nb.njit(cache=True)
def _get_cluster_primary_label(data: nb.float64[:,:],
                               clusts: nb.types.List(nb.int64[:]),
                               column: nb.int64) -> nb.float64[:]:

    labels = np.empty(len(clusts), dtype=data.dtype)
    group_ids = _get_cluster_label(data, clusts, GROUP_COL)
    for i in range(len(clusts)):
        part_ids = data[clusts[i], PART_COL]
        primary_mask = np.where(part_ids == group_ids[i])[0]
        if len(primary_mask):
            # Only use the primary component to label the cluster
            v, cts = nbl.unique(data[clusts[i][primary_mask], column])
        else: 
            # If there is no primary contribution, use the whole cluster
            v, cts = nbl.unique(data[clusts[i], column])
        labels[i] = v[np.argmax(cts)]

    return labels


@numbafy(cast_args=['data', 'coord_label'], list_args=['clusts'])
def get_cluster_closest_primary_label(data, coord_label, clusts, primary_ids):
    """Sets the primary label of clusters based on their proximity to the start
    point of the particle which created them.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    coord_label : np.ndarray
        Coordinate labels associated with each particle
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    primary_ids : np.ndarray
        (C) Existing list of primary IDs (the new labels will be a subset)

    Returns
    -------
    np.ndarray
        (C) List of cluster primary labels
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_closest_primary_label(
            data, coord_label, clusts, primary_ids)

@nb.njit(cache=True)
def _get_cluster_closest_primary_label(data: nb.float64[:,:],
                                       coord_label: nb.float64[:,:],
                                       clusts: nb.types.List(nb.int64[:]),
                                       primary_ids: nb.float64[:]) -> (
                                               nb.float64[:]):

    # Loop over the unique primary cluster groups
    primary_index = np.where(primary_ids == 1)[0]
    group_ids = _get_cluster_label(data, clusts, GROUP_COL)[primary_index]
    labels = primary_ids.copy()
    voxels = data[:, COORD_COLS]
    points = coord_label[:, COORD_COLS]
    for g in np.unique(group_ids.astype(np.int64)):
        # If the group index does not exist in the points, do not touch labels
        group_index = primary_index[group_ids == g]
        if g < 0 or g >= len(points):
            continue

        # Get the coordinates of the start point
        start_point = points[g].reshape(-1, 3)

        # Minimize the point-cluster distances
        dists = np.empty(len(group_index), dtype=data.dtype)
        for i, c in enumerate(group_index):
            dists[i] = np.min(nbl.cdist(start_point, voxels[clusts[c]]))

        # Label the closest cluster as the only primary cluster
        labels[group_index] = 0
        labels[group_index[np.argmin(dists)]] = 1

    return labels


@numbafy(cast_args=['data'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_centers(data, clusts):
    """Returns the coordinate of the centroid associated with each cluster.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C, 3) Tensor of cluster centers
    """
    if not len(clusts):
        return np.empty((0, 3), dtype=data.dtype)

    return _get_cluster_centers(data, clusts)

@nb.njit(cache=True)
def _get_cluster_centers(data: nb.float64[:,:],
                         clusts: nb.types.List(nb.int64[:])) -> (
                                 nb.float64[:,:]):

    centers = np.empty((len(clusts), 3), dtype=data.dtype)
    for i, c in enumerate(clusts):
        centers[i] = np.sum(data[c][:, COORD_COLS], axis=0)/len(c)

    return centers


@numbafy(cast_args=['data'], list_args=['clusts'])
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
    if not len(clusts):
        return np.empty(0, dtype=np.int64)

    return _get_cluster_sizes(data, clusts)

@nb.njit(cache=True)
def _get_cluster_sizes(data: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:])) -> nb.int64[:]:

    sizes = np.empty(len(clusts), dtype=np.int64)
    for i, c in enumerate(clusts):
        sizes[i] = len(c)

    return sizes


@numbafy(cast_args=['data'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_energies(data, clusts):
    """Returns the total charge/energy deposited by each cluster.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C) List of cluster pixel sums
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_energies(data, clusts)

@nb.njit(cache=True)
def _get_cluster_energies(data: nb.float64[:,:],
                          clusts: nb.types.List(nb.int64[:])) -> nb.float64[:]:

    energies = np.empty(len(clusts), dtype=data.dtype)
    for i, c in enumerate(clusts):
        energies[i] = np.sum(data[c, VALUE_COL])

    return energies


def get_cluster_features(data, clusts, add_value=False, add_shape=False):
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
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C, N_c) Tensor of cluster features
    """
    feats = get_cluster_features_base(data, clusts)
    if add_value or add_shape:
        feats_ext = get_cluster_features_extended(
                data, clusts, add_value, add_shape)
        if isinstance(data, np.ndarray):
            feats = np.hstack((feats, feats_ext))
        else:
            feats = torch.cat((feats, feats_ext), dim=1)

    return feats


@numbafy(cast_args=['data'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_features_base(data, clusts):
    """Returns an array of 16 geometric features for each of cluster.

    The 16 geometric features are composed of:
    - Center (3)
    - Covariance matrix (9)
    - Principal axis (3)
    - Voxel count (1)

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C, 16) Tensor of cluster features
    """
    if not len(clusts):
        return np.empty((0, 16), dtype=data.dtype) # Cannot type empty list

    return _get_cluster_features_base(data, clusts)

@nb.njit(parallel=True, cache=True)
def _get_cluster_features_base(data: nb.float64[:,:],
                               clusts: nb.types.List(nb.int64[:])) -> (
                                   nb.float64[:,:]):

    # Loop over the clusters (parallelize). The `prange` function creates a
    # uint64 iterator which is cast to int64 to access a list, and throws a
    # warning. To avoid this, use a separate counter to acces clusts.
    feats = np.empty((len(clusts), 16), dtype=data.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get list of voxels in the cluster
        clust = clusts[ids[k]]
        x = data[clust][:, COORD_COLS]

        # Get cluster center
        center = nbl.mean(x, 0)

        # Get orientation matrix
        A = np.cov(x.T, ddof = len(x) - 1).astype(x.dtype)

        # Center data
        x = x - center

        # Get eigenvectors, normalize orientation matrix and eigenvalues to
        # largest. If points are superimposed, i.e. if the largest eigenvalue
        # != 0, no need to keep going
        w, v = np.linalg.eigh(A)
        if w[2] == 0.:
            feats[k] = np.concatenate(
                    (center, np.zeros(12), np.array([len(clust)])))
            continue
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:,2]

        # Projection all points, x, along the principal axis
        x0 = np.dot(x, v0)

        # Evaluate the distance from the points to the principal axis
        xp0 = x - np.outer(x0, v0)
        np0 = np.empty(len(xp0), dtype=data.dtype)
        for i in range(len(xp0)):
            np0[i] = np.linalg.norm(xp0[i])

        # Flip the principal direction if it is not pointing towards the
        # maximum spread
        sc = np.dot(x0, np0)
        if sc < 0:
            # Numba does not support unary `-`, have to flip manually
            v0 = np.zeros(3, dtype=data.dtype) - v0

        # Weight direction
        v0 = dirwt * v0

        # Append
        feats[k] = np.concatenate(
                (center, B.flatten(), v0, np.array([len(clust)])))

    return feats


@numbafy(cast_args=['data'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_features_extended(data, clusts, add_value=True, add_shape=True):
    """Returns an array of 3 additional features for each of cluster.

    The flag `add_value` adds the following 2 features:
    - Mean energy (1)
    - RMS energy (1)

    The flag `add_shape` adds the particle shape information:
    - Semantic type (1), i.e. most represented type in cluster

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    add_value : bool, default True
        Whether to add the mean and std of the pixel values
    add_shape : bool, default True
        Whether to add the shape of the cluster

    Returns
    -------
    np.ndarray
        (C, 1/2/3) Tensor of additional cluster features
    """
    assert add_value or add_shape, (
            "Must add either value or shape for this function to do anything")
    if not len(clusts):
        return np.empty((0, add_value*2+add_shape), dtype=data.dtype)

    return _get_cluster_features_extended(data, clusts, add_value, add_shape)

@nb.njit(parallel=True, cache=True)
def _get_cluster_features_extended(data: nb.float64[:,:],
                                   clusts: nb.types.List(nb.int64[:]),
                                   add_value: bool = True,
                                   add_shape: bool = True) -> nb.float64[:,:]:
    feats = np.empty((len(clusts), add_value*2+add_shape), dtype=data.dtype)
    ids = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        # Get cluster
        clust = clusts[ids[k]]

        # Get mean and RMS energy in the cluster, if requested
        if add_value:
            mean_value = np.mean(data[clust, VALUE_COL])
            std_value = np.std(data[clust, VALUE_COL])
            feats[k, :2] = np.array([mean_value, std_value], dtype=data.dtype)

        # Get the cluster semantic class, if requested
        if add_shape:
            types, cnts = nbl.unique(data[clust, SHAPE_COL])
            major_sem_type = types[np.argmax(cnts)]
            feats[k, -1] = major_sem_type

    return feats


@numbafy(cast_args=['data', 'coord_label'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_points_label(data, coord_label, clusts, random_order=True):
    """Gets label points for each cluster.
    
    Returns start point of primary shower fragment twice if shower, delta or
    Michel and both end points of tracks if track.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    coord_label : np.ndarray
        (P, 9) Particle end points labels
        [batch_id, start_x, start_y, start_z, end_x, end_y, end_z, time, shape]
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
    if not len(clusts):
        return np.empty((0, 6), dtype=data.dtype)

    return _get_cluster_points_label(data, coord_label, clusts, random_order)

@nb.njit(cache=True)
def _get_cluster_points_label(data: nb.float64[:,:],
                              coord_label: nb.float64[:,:],
                              clusts: nb.types.List(nb.int64[:]),
                              random_order: nb.boolean = True) -> (
                                      nb.float64[:,:]):

    # Get start and end points (one and the same for all but track class)
    points = np.empty((len(clusts), 6), dtype=data.dtype)
    for i, c in enumerate(clusts):
        # Use the first cluster in time
        part_ids = np.unique(data[c, PART_COL]).astype(np.int64)
        min_id = part_ids[np.argmin(coord_label[part_ids, COORD_TIME_COL])]
        min_label = coord_label[min_id]
        start, end = min_label[COORD_START_COLS], min_label[COORD_END_COLS]
        if random_order and np.random.choice(2):
            start, end = end, start

        points[i,  :3] = start
        points[i, 3:6] = end

    # Bring the start points to the closest point in the corresponding cluster
    for i, c in enumerate(clusts):
        dist_mat = nbl.cdist(points[i].reshape(-1,3), data[c][:, COORD_COLS])
        argmins  = nbl.argmin(dist_mat, axis=1)
        points[i] = data[c][argmins][:, COORD_COLS].reshape(-1)

    return points


@numbafy(cast_args=['data', 'starts'], list_args=['clusts'], 
         keep_torch=True, ref_arg='data')
def get_cluster_directions(data, starts, clusts, max_dist=-1, optimize=False):
    """Estimates the direction of each cluster.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    starts : np.ndarray
        (C, 3) Start points w.r.t. which to estimate the direction
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    max_dist : float, default -1
        Neighborhood radius around the point used to estimate the direction
    optimize : bool, default False
        If `True`, the neighborhood radius is optimized on the fly for
        each cluster.

    Returns
    -------
    torch.tensor:
        (C, 3) Direction vector of each cluster
    """
    if not len(clusts):
        return np.empty(starts.shape, dtype=data.dtype)

    return _get_cluster_directions(
            data[:, COORD_COLS], starts, clusts, max_dist, optimize)

@nb.njit(parallel=True, cache=True)
def _get_cluster_directions(voxels: nb.float64[:,:],
                            starts: nb.float64[:,:],
                            clusts: nb.types.List(nb.int64[:]),
                            max_dist: nb.float64 = -1,
                            optimize: nb.boolean = False) -> nb.float64[:,:]:

    dirs = np.empty(starts.shape, voxels.dtype)
    ids  = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        dirs[k] = cluster_direction(
                voxels[clusts[ids[k]]], starts[k].astype(np.float64),
                max_dist, optimize)

    return dirs


@nb.njit(cache=True)
def cluster_direction(voxels: nb.float64[:,:],
                      start: nb.float64[:],
                      max_dist: nb.float64 = -1,
                      optimize: nb.boolean = False) -> nb.float64[:]:
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
    max_dist : float, default -1
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
    assert voxels.shape[1] == 3, (
            "The shape of the input is not compatible with voxel coordinates.")

    if max_dist > 0:
        dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
        voxels = voxels[dist_mat <= max(max_dist, np.min(dist_mat))]

    # If optimize is set, select the radius by minimizing the transverse spread
    if optimize and len(voxels) > 2:
        # Order the cluster points by increasing distance to the start point
        dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
        order = np.argsort(dist_mat)
        voxels = voxels[order]
        dist_mat = dist_mat[order]

        # Find the PCA relative secondary spread for each point
        labels = -np.ones(len(voxels), dtype=voxels.dtype)
        meank = nbl.mean(voxels[:3], 0)
        covk = (np.transpose(voxels[:3] - meank) @ (voxels[:3] - meank))/3
        for i in range(2, len(voxels)):
            # Get the eigenvalues, compute relative transverse spread
            w, _ = np.linalg.eigh(covk)
            labels[i] = np.sqrt(w[2] / (w[0] + w[1])) \
                    if (w[0] + w[1]) / w[2] > 1e-9 else 0.

            # If the value is the same as the previous, choose this one
            if dist_mat[i] == dist_mat[i-1]:
                labels[i-1] = -1.

            # Increment mean and matrix
            if i != len(voxels) - 1:
                meank = ((i + 1) * meank + voxels[i+1]) / (i + 2)
                covk = (i + 1) * covk / (i + 2) \
                        + (voxels[i+1] - meank).reshape(-1,1) \
                        * (voxels[i+1] - meank) / (i + 1)

        # Subselect voxels that are most track-like
        max_id = np.argmax(labels)
        voxels = voxels[:max_id+1]

    # If no voxels were selected, return dummy value
    if not len(voxels) or (len(voxels) == 1 and np.all(voxels[0] == start)):
        return np.array([1., 0., 0.], dtype=voxels.dtype)

    # Compute mean direction with respect to start point, normalize it
    rel_voxels = np.empty((len(voxels), 3), dtype=voxels.dtype)
    for i in range(len(voxels)):
        rel_voxels[i] = voxels[i] - start

    mean = nbl.mean(rel_voxels, 0)
    norm = np.sqrt(np.sum(mean**2))
    if norm:
        return mean/norm

    return mean


@numbafy(cast_args=['data', 'starts'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_dedxs(data, starts, clusts, max_dist=-1):
    """Computes the initial local dE/dxs of each cluster.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    starts : np.ndarray
        (C, 3) Start points w.r.t. which to estimate the local dE/dxs
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    max_dist : float, default -1
        Neighborhood radius around the point used to compute the dE/dx

    Returns
    -------
    np.ndarray
        (C) Local dE/dx values for each cluster
    """
    if not len(clusts):
        return np.empty(0, dtype=data.dtype)

    return _get_cluster_dedxs(
            data[:, COORD_COLS], data[:, VALUE_COL], starts, clusts, max_dist)

@nb.njit(parallel=True, cache=True)
def _get_cluster_dedxs(voxels: nb.float64[:,:],
                       values: nb.float64[:],
                       starts: nb.float64[:,:],
                       clusts: nb.types.List(nb.int64[:]),
                       max_dist: nb.float64 = -1) -> nb.float64[:,:]:

    dedxs = np.empty(len(clusts), voxels.dtype)
    ids   = np.arange(len(clusts)).astype(np.int64)
    for k in nb.prange(len(clusts)):
        dedxs[k] = cluster_dedx(
                voxels[clusts[ids[k]]], values[clusts[ids[k]]],
                starts[k].astype(np.float64), max_dist)

    return dedxs


@nb.njit(cache=True)
def cluster_dedx(voxels: nb.float64[:,:],
                 values: nb.float64[:],
                 start: nb.float64[:],
                 max_dist: nb.float64=5.0) -> nb.float64[:]:
    """Computes the initial local dE/dx of a cluster.

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates
    values : np.ndarray
        (N) Voxel values
    starts : np.ndarray
        (3) Start point w.r.t. which to compute the local dE/dx
    max_dist : float, default 5.0
        Neighborhood radius around the point used to compute the dE/dx

    Returns
    -------
    float
        Local dE/dx value around the start point
    """
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist
    assert voxels.shape[1] == 3, (
            "The shape of the input is not compatible with voxel coordinates.")

    dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
    if max_dist > 0:
        voxels = voxels[dist_mat <= max_dist]
        if len(voxels) < 2:
            return 0.
        values = values[dist_mat <= max_dist]
        dist_mat = dist_mat[dist_mat <= max_dist]

    # Compute the total energy in the neighborhood and the max distance, return ratio
    if np.max(dist_mat) == 0.:
        return 0.

    return np.sum(values)/np.max(dist_mat)


@nb.njit(cache=True)
def cluster_dedx2(voxels: nb.float64[:,:],
                 values: nb.float64[:],
                 start: nb.float64[:],
                 max_dist: nb.float64=5.0) -> nb.types.Tuple((nb.float64, nb.float64)):
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist                                                                                     
    assert voxels.shape[1] == 3, (
            "The shape of the input is not compatible with voxel coordinates.")

    dist_mat = nbl.cdist(start.reshape(1,-1), voxels).flatten()
    if max_dist > 0:
        voxels = voxels[dist_mat <= max_dist]
        if len(voxels) < 2:
            return 0., 0.
        values = values[dist_mat <= max_dist]
        dist_mat = dist_mat[dist_mat <= max_dist]

    if np.max(dist_mat) == 0.:
        return 0., 0.

        # Calculate sum of values
    sum_values = float(np.sum(values))
    
    # Calculate max distance
    max_distance = float(np.max(dist_mat))
    
    return sum_values, max_distance

        

@numbafy(cast_args=['data'], list_args=['clusts'],
         keep_torch=True, ref_arg='data')
def get_cluster_start_points(data, clusts):
    """Estimates the start point of clusters based on their PCA and the 
    local curvature at each of the PCA extrema.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    clusts : List[np.ndarray]
        (C) List of cluster indexes

    Returns
    -------
    np.ndarray
        (C, 3) Cluster start points
    """
    if not len(clusts):
        return np.empty((0, 3), dtype=data.dtype)

    return _get_cluster_start_points(data, clusts)

@nb.njit(parallel=True, cache=True)
def _get_cluster_start_points(data: nb.float64[:,:],
                              clusts: nb.types.List(nb.int64[:])) -> (
                                      nb.float64[:,:]):

    points = np.empty((len(clusts), 3))
    for k in nb.prange(len(clusts)):
        vid = cluster_end_points(data[clusts[k]][:, COORD_COLS])[-1]

    return points


@nb.njit(cache=True)
def cluster_end_points(voxels: nb.float64[:,:]) -> (
        nb.float64[:], nb.float64[:]):
    """Estimates the end points of a clusters using PCA and curvature.

    It proceeds in the following fashion:
    1. Find the principal axis a of the point cloud
    2. Find the coordinate a_i of each point along this axis
    3. Find the points with minimum and maximum coordinate
    4. Find the point that has the largest umbrella curvature

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates

    Returns
    -------
    int
        Index of the start voxel
    int
        Index of the end voxel
    """
    # Get the axis of maximum spread
    axis = nbl.principal_components(voxels)[0]

    # Compute coord values along that axis
    coords = np.empty(len(voxels))
    for i in range(len(coords)):
        coords[i] = np.dot(voxels[i], axis)

    # Compute curvature of the extremities
    ids = [np.argmin(coords), np.argmax(coords)]

    # Sort the voxel IDs by increasing order of curvature order
    curvs = [umbrella_curv(voxels, ids[0]), umbrella_curv(voxels, ids[1])]
    curvs = np.array(curvs, dtype=np.int64)
    ids = np.array(ids, dtype=np.int64)
    ids[np.argsort(curvs)]

    # Return extrema
    return voxels[ids[0]], voxels[ids[1]]


@nb.njit(cache=True)
def umbrella_curv(voxels: nb.float64[:,:],
                  vox_id: nb.int64) -> nb.float64:
    """Computes the umbrella curvature as in equation 9 of "Umbrella Curvature:
    A New Curvature Estimation Method for Point Clouds" by A.Foorginejad and
    K.Khalili
    (https://www.sciencedirect.com/science/article/pii/S2212017313006828)

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates
    vox_id : int
        Index of the voxel w.r.t. which to compute the curvature

    Returns
    -------
    float
        Value of the umbrella curvature at `vox_id`
    """
    # Find the mean direction from that point
    refvox = voxels[vox_id]
    diffs = voxels - refvox
    axis = nbl.mean(voxels - refvox, axis=0)
    axis /= np.linalg.norm(axis)

    # Compute the dot product of every displacement vector w.r.t. the axis
    dots = np.zeros(len(diffs), dtype=diffs.dtype)
    for i, diff in enumerate(diffs):
        if i != vox_id:
            dots[i] = np.dot(diff/np.linalg.norm(diff), axis)

    # Find the umbrella curvature (mean angle from the mean direction)
    return abs(np.mean(dots))
