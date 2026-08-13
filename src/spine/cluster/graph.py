"""Functions used to manipulate a graph of nodes and edges."""

# Graph kernels expose distance-algorithm controls directly and retain the
# intermediate closest-point arrays needed by their Numba implementations.
# pylint: disable=not-an-iterable,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals

from __future__ import annotations

from collections.abc import Sequence

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import ArrayLike, EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.jit import numbafy

from .topology import complete_graph

__all__ = [
    "get_cluster_edge_features",
    "get_cluster_edge_features_batch",
    "get_edge_distances",
    "inter_cluster_distance",
]


def get_cluster_edge_features_batch(
    data: TensorBatch,
    clusts: IndexBatch,
    edge_index: EdgeIndexBatch,
    closest_index: ArrayLike | None = None,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> TensorBatch:
    """Batched version of :func:`get_cluster_edge_features`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    clusts : IndexBatch
        (C) List of cluster indexes
    edge_index : EdgeIndexBatch
        (2, E) Sparse incidence matrix
    closest_index : Union[np.ndarray, torch.Tensor], optional
        (C, C) : Combined index of the closest pair of voxels per edge
    iterative : bool, default False
        If `True`, uses an iterative, fast approximation for distance computations
    use_legacy_distance : bool, default False
        If `True`, preserves the historical iterative closest-pair behavior

    Returns
    -------
    TensorBatch
        (E, N_e) List of edge features between clusters
    """
    directed = edge_index.directed
    index = edge_index.index_t if directed else edge_index.directed_index_t
    counts = edge_index.counts if directed else edge_index.directed_counts
    feats = get_cluster_edge_features(
        data.coords.tensor,
        clusts.index_list,
        index,
        closest_index,
        iterative,
        use_legacy_distance,
    )

    return TensorBatch(feats, counts)


@numbafy(cast_args=["data"], list_args=["clusts"], keep_torch=True, ref_arg="data")
def get_cluster_edge_features(
    data: ArrayLike,
    clusts: Sequence[ArrayLike],
    edge_index: ArrayLike,
    closest_index: ArrayLike | None = None,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> ArrayLike:
    """Returns a tensor of edge features for each edge connecting
    point clusters in the graph.

    The edge features (N_e = 19) include (in that order):
    - Coordinates of the voxel in the first cluster closest to the second (3)
    - Coordinates of the voxel in the second cluster closest to the first (3)
    - Displacement vector between the aforementioned voxels (3)
    - Magnitude of the displacement vector (1)
    - Outer product of the displacement vector (9)

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        Either an ``(N, 3)`` spatial-coordinate array or a legacy
        ``(N, 1 + D + N_f)`` sparse table with batch ID in the first column
    clusts : List[np.ndarray]
        (C) List of arrays of voxels IDs in each cluster
    edge_index : Union[np.ndarray, torch.Tensor]
        (2, E) Incidence map between voxels
    closest_index : Union[np.ndarray, torch.Tensor], optional
        (C, C) : Combined index of the closest pair of voxels per edge
    iterative : bool, default False
        If `True`, uses an iterative, fast approximation for distance computations
    use_legacy_distance : bool, default False
        If `True`, preserves the historical iterative closest-pair behavior

    Returns
    -------
    np.ndarray
        (E, N_e) Tensor of edge features
    """
    if len(clusts) == 0:
        return np.empty((0, 19), dtype=data.dtype)  # Cannot type empty list

    # Typed batch callers provide spatial coordinates directly. Preserve the
    # historical raw-array interface by stripping its batch/features columns.
    coordinates = data if data.shape[1] == 3 else data[:, 1:4]
    return _get_cluster_edge_features(
        coordinates,
        clusts,
        edge_index,
        closest_index,
        iterative,
        use_legacy_distance,
    )
    # return _get_cluster_edge_features_vec(
    #         data, clusts, edge_index, closest_index, iterative)


@nb.njit(parallel=True, cache=True)
def _get_cluster_edge_features(
    data: np.ndarray,
    clusts: Sequence[np.ndarray],
    edge_index: np.ndarray,
    closest_index: np.ndarray | None = None,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> np.ndarray:

    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxels in the clusters connected by the edge
        c1, c2 = edge_index[k]
        x1 = data[clusts[c1]]
        x2 = data[clusts[c2]]

        # Find the closest set point in each cluster
        if closest_index is not None:
            imin = closest_index[c1, c2]
            i1, i2 = imin // len(x2), imin % len(x2)
        else:
            if use_legacy_distance:
                i1, i2, _ = sm.distance.closest_pair_legacy(x1, x2, iterative)
            else:
                i1, i2, _ = sm.distance.closest_pair(x1, x2, iterative)
        v1 = x1[i1, :]
        v2 = x2[i2, :]

        # Displacement
        disp = v1 - v2

        # Distance
        lend = np.linalg.norm(disp)
        if lend > 0:
            disp = disp / lend

        # Outer product
        outer = np.outer(disp, disp).flatten()

        feats[k] = np.concatenate((v1, v2, disp, np.array([lend]), outer))

    return feats


@nb.njit(cache=True)
def _get_cluster_edge_features_vec(
    data: np.ndarray,
    clusts: Sequence[np.ndarray],
    edge_index: np.ndarray,
    closest_index: np.ndarray | None = None,
    iterative: bool = False,
) -> np.ndarray:

    # Get the closest points of approach IDs for each edge
    edge_lengths = np.empty(0, dtype=data.dtype)
    if closest_index is None:
        edge_lengths, idxs1, idxs2 = _get_edge_distances(
            data, clusts, edge_index, iterative
        )
    else:
        idxs1 = np.empty(edge_index.shape[1], dtype=np.int64)
        idxs2 = np.empty(edge_index.shape[1], dtype=np.int64)
        for k in range(edge_index.shape[1]):
            c1, c2 = edge_index[0, k], edge_index[1, k]
            combined = closest_index[c1, c2]
            idxs1[k] = clusts[c1][combined // len(clusts[c2])]
            idxs2[k] = clusts[c2][combined % len(clusts[c2])]

    # Get the points that correspond to the first voxels
    v1 = data[idxs1]

    # Get the points that correspond to the second voxels
    v2 = data[idxs2]

    # Get the displacement
    disp = v1 - v2

    # Reshape the distance vector to a column vector
    if closest_index is None:
        lend = edge_lengths.reshape(-1, 1)
    else:
        lend = sm.linalg.norm(disp, 1).reshape(-1, 1)

    # Normalize the displacement vector
    disp = disp / (lend + (lend == 0))

    # Compute the outer product of the displacement
    outer = np.empty((len(disp), 9), dtype=data.dtype)
    for k, displacement in enumerate(disp):
        outer[k] = np.outer(displacement, displacement).flatten()

    return np.hstack((v1, v2, disp, lend, outer))


@numbafy(cast_args=["voxels"], list_args=["clusts"])
def get_edge_distances(
    voxels: ArrayLike,
    clusts: Sequence[ArrayLike],
    edge_index: ArrayLike,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """For each edge, finds the closest points of approach (CPAs) between the
    two voxel clusters it connects, and the distance that separates them.

    Notes
    -----
    The voxel IDs correspond to the voxel list, not an index within a cluster.

    Parameters
    ----------
    voxels : Union[np.ndarray, torch.Tensor
        (N,3) Tensor of voxel coordinates
    clusts : List[np.ndarray]
        (C) List of arrays of voxel IDs in each cluster
    edge_index : Union[np.ndarray, torch.Tensor]
        (2, E) Incidence matrix
    iterative : bool, default False
        If `True`, uses an iterative, fast approximation for distance computations
    use_legacy_distance : bool, default False
        If `True`, preserves the historical iterative closest-pair behavior

    Returns
    -------
    np.ndarray
        (E) List of edge lengths
    np.ndarray
        (E) List of voxel IDs corresponding to the first edge cluster CPA
    np.ndarray
        (E) List of voxel IDs corresponding to the second edge cluster CPA
    """
    return _get_edge_distances(
        voxels, clusts, edge_index, iterative, use_legacy_distance
    )


@nb.njit(parallel=True, cache=True)
def _get_edge_distances(
    voxels: np.ndarray,
    clusts: Sequence[np.ndarray],
    edge_index: np.ndarray,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Loop over the provided edges
    lend = np.empty(len(edge_index), dtype=voxels.dtype)
    resi = np.empty(len(edge_index), dtype=np.int64)
    resj = np.empty(len(edge_index), dtype=np.int64)
    indxi, indxj = edge_index
    for k in nb.prange(len(indxi)):
        i, j = indxi[k], indxj[k]
        if i == j:
            ii = jj = 0
            dist = 0.0
        else:
            if use_legacy_distance:
                ii, jj, dist = sm.distance.closest_pair_legacy(
                    voxels[clusts[i]], voxels[clusts[j]], iterative
                )
            else:
                ii, jj, dist = sm.distance.closest_pair(
                    voxels[clusts[i]], voxels[clusts[j]], iterative
                )

        lend[k] = dist
        resi[k] = clusts[i][ii]
        resj[k] = clusts[j][jj]

    return lend, resi, resj


@numbafy(cast_args=["voxels"], list_args=["clusts"])
def inter_cluster_distance(
    voxels: ArrayLike,
    clusts: Sequence[ArrayLike],
    counts: np.ndarray | None = None,
    centroid: bool = False,
    iterative: bool = False,
    return_index: bool = False,
    use_legacy_distance: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Finds the inter-cluster distance between every pair of clusters within
    each batch, returned as a block-diagonal matrix.

    Parameters
    ----------
    voxels : Union[np.ndarray, torch.Tensor]
        (N, D) Tensor of voxel coordinates
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    counts : np.ndarray, optional
        (B) Number of clusters in each entry of the batch
    centroid : bool, default False
        If `True`, use the centroid distance as a fast, approximate proxy
    iterative : bool, default False
        If `True`, uses an iterative, fast approximation to compute voxel distance
    use_legacy_distance : bool, default False
        If `True`, preserves the historical iterative closest-pair behavior
    return_index : bool, default True
        Returns a combined index of the closest pair of voxels for each
        cluster, if the 'centroid' distance method is not used

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (C, C) Tensor of pair-wise cluster distances
    Union[np.ndarray, torch.Tensor], optional
        (C, C) Tensor of pair-wise closest voxel pair
    """
    # If there is no counts provided, assume all clusters are in one entry
    if counts is None:
        counts = np.array([len(clusts)], dtype=np.int64)

    if not return_index:
        # If there are no clusters, return empty
        if len(clusts) == 0:
            return np.empty((0, 0), dtype=voxels.dtype)

        return _inter_cluster_distance(
            voxels, clusts, counts, centroid, iterative, use_legacy_distance
        )

    # If there are no clusters, return empty
    assert not centroid, "Cannot return index for centroid method."
    if len(clusts) == 0:
        return (
            np.empty((0, 0), dtype=voxels.dtype),
            np.empty((0, 0), dtype=np.int64),
        )

    return _inter_cluster_distance_index(
        voxels, clusts, counts, iterative, use_legacy_distance
    )


@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance(
    voxels: np.ndarray,
    clusts: Sequence[np.ndarray],
    counts: np.ndarray,
    centroid: bool = False,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> np.ndarray:

    # Loop over the upper diagonal elements of each block on the diagonal
    dist_mat = np.zeros((len(clusts), len(clusts)), dtype=voxels.dtype)
    indxi, indxj = complete_graph(counts)
    if not centroid:
        for k in nb.prange(len(indxi)):
            # Identifiy the two voxels closest to each other in each cluster
            i, j = indxi[k], indxj[k]
            if use_legacy_distance:
                dist_mat[i, j] = dist_mat[j, i] = sm.distance.closest_pair_legacy(
                    voxels[clusts[i]], voxels[clusts[j]], iterative
                )[-1]
            else:
                dist_mat[i, j] = dist_mat[j, i] = sm.distance.closest_pair(
                    voxels[clusts[i]], voxels[clusts[j]], iterative
                )[-1]

    else:
        # Compute the centroid of each cluster
        dtype = voxels.dtype
        centroids = np.empty((len(clusts), voxels.shape[1]), dtype=dtype)
        for i in nb.prange(len(clusts)):
            centroids[i] = sm.mean(voxels[clusts[i]], axis=0)

        # Measure the distance between cluster centroids
        for k in nb.prange(len(indxi)):
            i, j = indxi[k], indxj[k]
            dist_mat[i, j] = dist_mat[j, i] = np.sqrt(
                np.sum((centroids[j] - centroids[i]) ** 2)
            )

    return dist_mat


@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance_index(
    voxels: np.ndarray,
    clusts: Sequence[np.ndarray],
    counts: np.ndarray,
    iterative: bool = False,
    use_legacy_distance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    # Loop over the upper diagonal elements of each block on the diagonal
    dist_mat = np.zeros((len(clusts), len(clusts)), dtype=voxels.dtype)
    closest_index = np.zeros((len(clusts), len(clusts)), dtype=np.int64)
    indxi, indxj = complete_graph(counts)
    for k in nb.prange(len(indxi)):
        # Identify the two voxels closest to each other in each cluster
        i, j = indxi[k], indxj[k]
        if use_legacy_distance:
            ii, jj, dist = sm.distance.closest_pair_legacy(
                voxels[clusts[i]], voxels[clusts[j]], iterative
            )
        else:
            ii, jj, dist = sm.distance.closest_pair(
                voxels[clusts[i]], voxels[clusts[j]], iterative
            )
        index = ii * len(clusts[j]) + jj

        # Store the index and the distance in a matrix
        closest_index[i, j] = closest_index[j, i] = index
        dist_mat[i, j] = dist_mat[j, i] = dist

    return dist_mat, closest_index
