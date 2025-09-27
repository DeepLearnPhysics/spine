"""Module which contains functions used to clean up cluster data.

When loading :class:`larcv.Cluster3DVoxelTensor` objects into tensors,
there can be duplicate voxels. These routines are used to remove these
duplicates and ensure the ordering of the output.
"""

import numba as nb
import numpy as np

from spine.utils.globals import SHAPE_COL, SHAPE_PREC


def clean_sparse_data(
    cluster_voxels,
    cluster_data,
    sparse_voxels=None,
    sum_cols=None,
    prec_col=SHAPE_COL,
    precedence=SHAPE_PREC,
):
    """Helper that factorizes common cleaning operations required when trying
    to match cluster3d data products to sparse3d data products.

    This function does the following:
    1. Lexicographically sort group data (images are lexicographically sorted)
    2. Choose only one group per voxel (by lexicographic order or precedence)
    3. Remove voxels from cluster data that are not in the image data (optional)

    The set of sparse voxels must be a subset of the set of cluster voxels and
    it must not contain any duplicates. If not provided, this function can also
    be used to remove duplicates when overlaying multiple images together.

    Parameters
    ----------
    cluster_voxels : np.ndarray
        (N, 3) Matrix of voxel coordinates in the cluster3d tensor
    cluster_data : np.ndarray
        (N, F) Matrix of voxel values corresponding to each voxel
        in the cluster3d tensor
    sparse_voxels : np.ndarray, optional
        (M, 3) Matrix of voxel coordinates in the reference sparse tensor
    sum_cols : np.ndarray, optional
        List of feature columns to sum when removing duplicates
    prec_col : int, default SHAPE_COL
        Column in the input feature tensor to use as a precdence source
    precedence: list, default SHAPE_PREC
        (C) Array of classes in the reference array, ordered by precedence

    Returns
    -------
    cluster_voxels: np.ndarray
        (M, 3) Ordered and filtered set of voxel coordinates
    cluster_data: np.ndarray
        (M, F) Ordered and filtered set of voxel values
    """
    # Lexicographically sort cluster and sparse data
    perm = np.lexsort(cluster_voxels.T)
    cluster_voxels = cluster_voxels[perm]
    cluster_data = cluster_data[perm]

    # Find duplicates (and the groups they belong to)
    if prec_col is not None or sum_cols is not None:
        reference = cluster_data[:, prec_col] if prec_col else None
        duplicate_mask, groups = filter_duplicate_voxels_group(
            cluster_voxels, reference, nb.typed.List(precedence)
        )

    else:
        duplicate_mask = filter_duplicate_voxels(cluster_voxels)

    # Sum the values of duplicate voxels, if requested
    if sum_cols is not None and len(groups) > 0:
        cluster_data = aggregate_features(cluster_data, groups, sum_cols)

    # Remove duplicates
    duplicate_index = np.where(duplicate_mask)[0]
    cluster_voxels = cluster_voxels[duplicate_index]
    cluster_data = cluster_data[duplicate_index]

    # Remove voxels not present in the sparse matrix, if needed
    if sparse_voxels is not None:
        perm = np.lexsort(sparse_voxels.T)
        sparse_voxels = sparse_voxels[perm]

        non_ref_mask = filter_voxels_ref(cluster_voxels, sparse_voxels)
        non_ref_index = np.where(non_ref_mask)[0]
        cluster_voxels = cluster_voxels[non_ref_index]
        cluster_data = cluster_data[non_ref_index]

    return cluster_voxels, cluster_data


@nb.njit(cache=True)
def filter_duplicate_voxels(data: nb.int32[:, :]) -> nb.boolean[:]:
    """Returns a mask of non-duplicate voxels.

    If there are multiple voxels with the same coordinates, this algorithm
    simply keeps the last one in the list.

    Parameters
    ----------
    data: np.ndarray
        (N, 3) Lexicographically sorted matrix of voxel coordinates

    Returns
    -------
    np.ndarray
        (N) Boolean mask which is False for pixels to remove
    """
    # For each voxel, check if the next one shares its coordinates
    num_voxels = data.shape[0]
    mask = np.ones(num_voxels, dtype=np.bool_)
    for i in range(1, num_voxels):
        if np.all(data[i - 1] == data[i]):
            mask[i - 1] = False

    return mask


@nb.njit(cache=True)
def filter_duplicate_voxels_group(
    data: nb.int32[:, :],
    reference: nb.int32[:] = None,
    precedence: nb.types.List(nb.int32) = None,
) -> nb.boolean[:]:
    """Returns a mask of non-duplicate voxels and a list of duplicate groups.

    If there are multiple voxels with the same coordinates, this algorithm
    simply keeps the last one in the list.

    If a precedence is defined and there are multiple voxels with the same
    coordinates, this algorithm picks the voxel which has the label that comes
    first in order of precedence. If multiple voxels with the same precedence
    index share voxel coordinates, the last one is picked.

    The duplicate voxel groups map the chosen voxel indices to the set of voxels
    which share voxel coordinates.

    Parameters
    ----------
    data: np.ndarray
        (N, 3) Lexicographically sorted matrix of voxel coordinates
    reference: np.ndarray, optional
        (N) Array of values which have to follow the precedence order
    precedence: list, optional
        (C) Array of classes in the reference array, ordered by precedence

    Returns
    -------
    np.ndarray
        (N) Boolean mask which is False for pixels to remove
    Dict[int, np.ndarray]
        Map between kept voxel indexes onto voxels which share the same coordinates
    """
    # Find all the voxels which are duplicated and organize them in groups
    num_voxels = data.shape[0]
    mask = np.ones(num_voxels, dtype=np.bool_)
    tmp_list = nb.typed.List.empty_list(nb.int64)
    groups = []
    for i in range(1, num_voxels):
        # Check if the current voxel matches the previous one
        same = np.all(data[i - 1] == data[i])

        # If it does, create/append the duplicate group
        if same:
            if not tmp_list:
                tmp_list.extend([i - 1, i])
            else:
                tmp_list.append(i)

        # If it does not, store an any existing duplicate group
        if tmp_list and (not same or i == num_voxels - 1):
            groups.append(np.asarray(tmp_list))
            tmp_list = nb.typed.List.empty_list(nb.int64)

    # For each group, pick the voxel with the label that comes first
    # in order of precedence, track duplicate groups
    merge = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64[:])
    for group in groups:
        if reference is not None:
            # Order the voxels in the group by precedence
            ref = np.array([precedence.index(int(r)) for r in reference[group]])
            args = np.argsort(-ref, kind="mergesort")  # Preserve duplicate order

            # Modify the mask and store the group indexes
            mask[group[args[:-1]]] = False
            merge[group[args[-1]]] = group

        else:
            # Pick the last element in the group
            mask[group[:-1]] = False
            merge[group[-1]] = group

    return mask, merge


@nb.njit(cache=True)
def filter_voxels_ref(data: nb.int32[:, :], reference: nb.int32[:, :]) -> nb.boolean[:]:
    """Removes voxels thsat do not appear in a reference tensor.

    Returns an array which does not contain any voxels which do not belong to
    the reference array. The reference array must contain a subset of the
    voxels in the array to be filtered.

    Assumes both arrays are lexicographically sorted, the reference matrix
    contains no duplicates and is a subset of the matrix to be filtered.

    Parameters
    ----------
    data: np.ndarray
        (N, 3) Lexicographically sorted matrix of voxel coordinates to filter
    reference: np.ndarray
        (N, 3) Lexicographically sorted matrix of voxel coordinates to match

    Returns
    -------
    np.ndarray
        (N) Boolean mask which is False for pixels to remove
    """
    # Try to match each voxel in the data tensor to one in the reference tensor
    n_data, n_ref = data.shape[0], reference.shape[0]
    d, r = 0, 0
    mask = np.ones(n_data, dtype=np.bool_)
    while d < n_data and r < n_ref:
        if np.all(data[d] == reference[r]):
            # Voxel is in both matrices
            d += 1
            r += 1
        else:
            # Voxel is in data, but not reference
            mask[d] = False
            d += 1

    # Need to go through rest of data, if any is left
    while d < n_data:
        mask[d] = False
        d += 1

    return mask


@nb.njit(cache=True)
def aggregate_features(
    data: nb.float32[:, :], groups: nb.typed.Dict, cols: nb.int64[:]
):
    """Aggregate the information in pre-defined voxel groups.

    Parameters
    ----------
    data : np.ndarray
        (N, F) Matrix of voxel features to aggregate
    groups : Dict[int, np.ndarray]
        Map between kept voxel indexes onto voxels which share the same coordinates
    cols : np.ndarray
        List of feature columns to modify

    Returns
    -------
    np.ndarray
        (N, F) Matrix of aggregated voxel features
    """
    for col in cols:
        for idx, group in groups.items():
            data[idx, col] = np.sum(data[group, col])

    return data
