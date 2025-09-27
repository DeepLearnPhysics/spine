"""Functions used to manipulate a graph of nodes and edges."""

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import TensorBatch
from spine.utils.globals import COORD_COLS
from spine.utils.jit import numbafy


def get_cluster_edge_features_batch(
    data, clusts, edge_index, closest_index=None, iterative=False
):
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

    Returns
    -------
    TensorBatch
        (E, N_e) List of edge features between clusters
    """
    directed = edge_index.directed
    index = edge_index.index_t if directed else edge_index.directed_index_t
    counts = edge_index.counts if directed else edge_index.directed_counts
    feats = get_cluster_edge_features(
        data.tensor, clusts.index_list, index, closest_index, iterative
    )

    return TensorBatch(feats, counts)


def get_voxel_edge_features_batch(data, edge_index, max_dist=5.0):
    """Batched version of :func:`get_voxel_edge_features`.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    edge_index : EdgeIndexBatch
        (2, E) Sparse incidence matrix

    Returns
    -------
    TensorBatch
        (E, N_e) List of edge features between voxels.
    """
    directed = edge_index.directed
    index = edge_index.index_t if directed else edge_index.directed_index_t
    counts = edge_index.counts if directed else edge_index.directed_counts
    feats = get_voxel_edge_features(data.tensor, index)

    return TensorBatch(feats, counts)


@numbafy(cast_args=["data"], list_args=["clusts"], keep_torch=True, ref_arg="data")
def get_cluster_edge_features(
    data, clusts, edge_index, closest_index=None, iterative=False
):
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
        (N, 1 + D + N_f) Batched sparse tensors
    clusts : List[np.ndarray]
        (C) List of arrays of voxels IDs in each cluster
    edge_index : Union[np.ndarray, torch.Tensor]
        (2, E) Incidence map between voxels
    closest_index : Union[np.ndarray, torch.Tensor], optional
        (C, C) : Combined index of the closest pair of voxels per edge
    iterative : bool, default False
        If `True`, uses an iterative, fast approximation for distance computations

    Returns
    -------
    np.ndarray
        (E, N_e) Tensor of edge features
    """
    if not len(clusts):
        return np.empty((0, 19), dtype=data.dtype)  # Cannot type empty list

    return _get_cluster_edge_features(
        data, clusts, edge_index, closest_index, iterative
    )
    # return _get_cluster_edge_features_vec(
    #         data, clusts, edge_index, closest_index, iterative)


@nb.njit(parallel=True, cache=True)
def _get_cluster_edge_features(
    data: nb.float32[:, :],
    clusts: nb.types.List(nb.int64[:]),
    edge_index: nb.int64[:, :],
    closest_index: nb.int64[:] = None,
    iterative: nb.boolean = False,
) -> nb.float32[:, :]:

    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxels in the clusters connected by the edge
        c1, c2 = edge_index[k]
        x1 = data[clusts[c1]][:, COORD_COLS]
        x2 = data[clusts[c2]][:, COORD_COLS]

        # Find the closest set point in each cluster
        if closest_index is not None:
            imin = closest_index[c1, c2]
            i1, i2 = imin // len(x2), imin % len(x2)
        else:
            i1, j2, _ = sm.distance.closest_pair(x1, x2, iterative)
        v1 = x1[i1, :]
        v2 = x2[i2, :]

        # Displacement
        disp = v1 - v2

        # Distance
        lend = np.linalg.norm(disp)
        if lend > 0:
            disp = disp / lend

        # Outer product
        B = np.outer(disp, disp).flatten()

        feats[k] = np.concatenate((v1, v2, disp, np.array([lend]), B))

    return feats


@nb.njit(cache=True)
def _get_cluster_edge_features_vec(
    data: nb.float32[:, :],
    clusts: nb.types.List(nb.int64[:]),
    edge_index: nb.int64[:, :],
    closest_index: nb.int64[:] = None,
    iterative: nb.boolean = False,
) -> nb.float32[:, :]:

    # Get the closest points of approach IDs for each edge
    if closest_index is None:
        lend, idxs1, idxs2 = _get_edge_distances(
            data[:, COORD_COLS], clusts, edge_index, iterative
        )
    else:
        idxs1, idxs2 = closest_index[(edge_index[0], edge_index[1])]

    # Get the points that correspond to the first voxels
    v1 = data[idxs1][:, COORD_COLS]

    # Get the points that correspond to the second voxels
    v2 = data[idxs2][:, COORD_COLS]

    # Get the displacement
    disp = v1 - v2

    # Reshape the distance vector to a column vector
    if closest_index is None:
        lend = lend.reshape(-1, 1)
    else:
        lend = np.linalg.norm(disp, axis=1)

    # Normalize the displacement vector
    disp = disp / (lend + (lend == 0))

    # Compute the outer product of the displacement
    B = np.empty((len(disp), 9), dtype=data.dtype)
    for k in range(len(disp)):
        B[k] = np.outer(disp, disp).flatten()
    # B = np.dot(disp.reshape(len(disp),-1,1),
    #            disp.reshape(len(disp),1,-1)).reshape(len(disp),-1)

    return np.hstack((v1, v2, disp, lend, B))


@numbafy(cast_args=["data"], keep_torch=True, ref_arg="data")
def get_voxel_edge_features(data, edge_index):
    """Returns a tensor of edge features for each edge connecting
    point individual voxels in the graph.

    The edge features (N_e = 19) include (in that order):
    - Coordinates of the source voxel (3)
    - Coordinates of the target voxel (3)
    - Displacement vector between the two aforementioned voxels (3)
    - Magnitude of the displacement vector (1)
    - Outer product of the displacement vector (9)

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        (N, 1 + D + N_f) Batched sparse tensors
    clusts : List[np.ndarray]
        (C) List of arrays of voxels IDs in each cluster
    edge_index : Union[np.ndarray, torch.Tensor]
        (2, E) Incidence map between voxels

    Returns
    -------
    np.ndarray
        (E, N_e) Tensor of edge features
    """
    return _get_voxel_edge_features(data, edge_index)


@nb.njit(parallel=True, cache=True)
def _get_voxel_edge_features(
    data: nb.float32[:, :], edge_index: nb.int64[:, :]
) -> nb.float32[:, :]:
    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxel coordinates
        xi = data[edge_index[k, 0]][:, COORD_COLS]
        xj = data[edge_index[k, 1]][:, COORD_COLS]

        # Displacement
        disp = xj - xi

        # Distance
        lend = np.linalg.norm(disp)
        if lend > 0:
            disp = disp / lend

        # Outer product
        B = np.outer(disp, disp).flatten()

        feats[k] = np.concatenate([xi, xj, disp, np.array([lend]), B])

    return feats


@numbafy(cast_args=["voxels"], list_args=["clusts"])
def get_edge_distances(voxels, clusts, edge_index, iterative):
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

    Returns
    -------
    np.ndarray
        (E) List of edge lengths
    np.ndarray
        (E) List of voxel IDs corresponding to the first edge cluster CPA
    np.ndarray
        (E) List of voxel IDs corresponding to the second edge cluster CPA
    """
    return _get_edge_distances(voxels, clusts, edge_index, iterative)


@nb.njit(parallel=True, cache=True)
def _get_edge_distances(
    voxels: nb.float32[:, :],
    clusts: nb.types.List(nb.int64[:]),
    edge_index: nb.int64[:, :],
    iterative: nb.boolean = False,
) -> (nb.float32[:], nb.int64[:], nb.int64[:]):

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
            ii, jj, dist = sm.distance.closest_pair(
                voxels[clusts[i]], voxels[clusts[j]], iterative
            )

        lend[k] = dist
        resi[k] = clusts[i][ii]
        resj[k] = clusts[j][jj]

    return lend, resi, resj


@numbafy(cast_args=["voxels"], list_args=["clusts"])
def inter_cluster_distance(
    voxels, clusts, counts=None, centroid=False, iterative=False, return_index=False
):
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

        return _inter_cluster_distance(voxels, clusts, counts, centroid, iterative)

    else:
        # If there are no clusters, return empty
        assert not centroid, "Cannot return index for centroid method."
        if len(clusts) == 0:
            return (
                np.empty((0, 0), dtype=voxels.dtype),
                np.empty((0, 0), dtype=np.int64),
            )

        return _inter_cluster_distance_index(voxels, clusts, counts, iterative)


@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance(
    voxels: nb.float32[:, :],
    clusts: nb.types.List(nb.int64[:]),
    counts: nb.int64[:],
    centroid: nb.boolean = False,
    iterative: nb.boolean = False,
) -> nb.float32[:, :]:

    # Loop over the upper diagonal elements of each block on the diagonal
    dist_mat = np.zeros((len(clusts), len(clusts)), dtype=voxels.dtype)
    indxi, indxj = complete_graph(counts)
    if not centroid:
        for k in nb.prange(len(indxi)):
            # Identifiy the two voxels closest to each other in each cluster
            i, j = indxi[k], indxj[k]
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
    voxels: nb.float32[:, :],
    clusts: nb.types.List(nb.int64[:]),
    counts: nb.int64[:],
    iterative: nb.boolean = False,
) -> (nb.float32[:, :], nb.int64[:, :]):

    # Loop over the upper diagonal elements of each block on the diagonal
    dist_mat = np.zeros((len(clusts), len(clusts)), dtype=voxels.dtype)
    closest_index = np.zeros((len(clusts), len(clusts)), dtype=nb.int64)
    indxi, indxj = complete_graph(counts)
    for k in nb.prange(len(indxi)):
        # Identify the two voxels closest to each other in each cluster
        i, j = indxi[k], indxj[k]
        ii, jj, dist = sm.distance.closest_pair(
            voxels[clusts[i]], voxels[clusts[j]], iterative
        )
        index = ii * len(clusts[j]) + jj

        # Store the index and the distance in a matrix
        closest_index[i, j] = closest_index[j, i] = index
        dist_mat[i, j] = dist_mat[j, i] = dist

    return dist_mat, closest_index


@numbafy(cast_args=["graph"])
def get_fragment_edges(graph, clust_ids):
    """Function that converts a set of edges between cluster ids
    to a set of edges between fragment ids (ordering in list).

    Parameters
    ----------
    graph : Union[np.ndarray, torch.Tensor]
        (E, 2) Tensor of [clust_id_1, clust_id_2]
    clust_ids : np.ndarray
        (C) List of fragment cluster ids

    Returns
    -------
    np.ndarray
        (E,2) Tensor of true edges [frag_id_1, frag_id2]
    """
    return _get_fragment_edges(graph, clust_ids)


@nb.njit(cache=True)
def _get_fragment_edges(
    graph: nb.int64[:, :], clust_ids: nb.int64[:]
) -> nb.int64[:, :]:
    # Loop over the graph edges, find the fragment ids, append
    true_edges = np.empty((0, 2), dtype=np.int64)
    for e in graph:
        n1 = np.where(clust_ids == e[0])[0]
        n2 = np.where(clust_ids == e[1])[0]
        if len(n1) and len(n2):
            true_edges = np.vstack(
                (true_edges, np.array([[n1[0], n2[0]]], dtype=np.int64))
            )

    return true_edges


@nb.njit(cache=True)
def complete_graph(counts: nb.int64[:]) -> nb.int64[:, :]:
    """Creates a list of edges corresponding to a directed complete graph
    in a batch of nodes (nodes from separate entries).

    Parameters
    ----------
    counts : np.ndarray, optional
        (B) Number of nodes in each entry of the batch
    """
    # Loop over the batches, define the adjacency matrix for each
    num_edges = np.sum((counts * (counts - 1)) // 2)
    edge_index = np.empty((2, num_edges), dtype=np.int64)
    offset, index = 0, 0
    for b in range(len(counts)):
        # Build a list of edges
        c = counts[b]
        adj_mat = np.triu(np.ones((c, c)), k=1)
        edges = np.vstack(np.where(adj_mat))
        num_edges_b = edges.shape[1]

        # Append
        edge_index[:, index : index + num_edges_b] = offset + edges
        index += num_edges_b
        offset += c

    return edge_index


@nb.njit(cache=True)
def filter_invalid_nodes(
    edge_index: nb.int64[:, :], invalid_nodes: nb.int64[:]
) -> nb.int64[:, :]:
    """Remove invalid node from a graph, bridge gaps formed by the filtering.

    Each time a node is removed, the function proceeds as follows:
    - If the node has no children, remove any edges that connect to it
    - If the node has children:
      - If it does not have a parent, remove any edges from it
      - If it has a parent, connect the parent to its children

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) Original graph incidence map
    invalid_nodes : np.ndarray
        (N) List of nodes to remove from the original graph incidence map

    Returns
    -------
    np.ndarray
        (E', 2) Filtered graph incidence map
    """
    # Loop over the list of invalid nodes
    edges = edge_index.copy()
    for node in invalid_nodes:
        # If the node has no children, remove edges to the node
        children = np.where(edges[:, 0] == node)[0]
        if len(children) == 0:
            edges = edges[edges[:, 1] != node]
            continue

        # If it has children, find its parent and reassign its children
        parent = np.where(edges[:, 1] == node)[0]
        assert len(parent) <= 1, "Found a particle with multiple parents, not allowed."

        if len(parent) == 1:
            # If it has a parent, then assign children to that parent
            parent_id = edges[parent][0][0]
            edges[:, 0][children] = parent_id
        else:
            # If it has no parent, remove the edges to it
            edges = edges[edges[:, 0] != node]

        # Remove edges to the node
        edges = edges[edges[:, 1] != node]

    return edges
