import numpy as np
import numba as nb

import mlreco.utils.numba_local as nbl
from mlreco.utils.decorators import numbafy
from mlreco.utils.globals import COORD_COLS
from mlreco.utils.factory import module_dict, instantiate


@numbafy(cast_args=['data'], list_args=['clusts'], keep_torch=True, ref_arg='data')
def get_cluster_edge_features(data, clusts, edge_index, closest_index=None):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting clusters in the graph.

    Parameters
    ----------
        data (np.ndarray)         : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        clusts ([np.ndarray])     : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray)   : (2,E) Incidence matrix
        closest_index (np.ndarray): (E) Index of closest pair of voxels for each edge
    Returns
    -------
        np.ndarray: (E,19) Tensor of edge features (point1, point2, displacement, distance, orientation)
    """
    return _get_cluster_edge_features(data, clusts, edge_index, closest_index)
    #return _get_cluster_edge_features_vec(data, clusts, edge_index)

@nb.njit(parallel=True, cache=True)
def _get_cluster_edge_features(data: nb.float32[:,:],
                               clusts: nb.types.List(nb.int64[:]),
                               edge_index: nb.int64[:,:],
                               closest_index: nb.int64[:] = None) -> nb.float32[:,:]:

    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxels in the clusters connected by the edge
        c1, c2 = edge_index[k]
        x1 = data[clusts[c1]][:, COORD_COLS]
        x2 = data[clusts[c2]][:, COORD_COLS]

        # Find the closest set point in each cluster
        imin = np.argmin(nbl.cdist(x1, x2)) if closest_index is None else closest_index[k]
        i1, i2 = imin//len(x2), imin%len(x2)
        v1 = x1[i1,:]
        v2 = x2[i2,:]

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
def _get_cluster_edge_features_vec(data: nb.float32[:,:],
                                   clusts: nb.types.List(nb.int64[:]),
                                   edge_index: nb.int64[:,:]) -> nb.float32[:,:]:

    # Get the closest points of approach IDs for each edge
    lend, idxs1, idxs2 = _get_edge_distances(data[:,COORD_COLS], clusts, edge_index)

    # Get the points that correspond to the first voxels
    v1 = data[idxs1][:, COORD_COLS]

    # Get the points that correspond to the second voxels
    v2 = data[idxs2][:, COORD_COLS]

    # Get the displacement
    disp = v1 - v2

    # Reshape the distance vector to a column vector
    lend = lend.reshape(-1,1)

    # Normalize the displacement vector
    disp = disp/(lend + (lend == 0))

    # Compute the outer product of the displacement
    B = np.empty((len(disp), 9), dtype=data.dtype)
    for k in range(len(disp)):
        B[k] = np.outer(disp, disp).flatten()
    #B = np.dot(disp.reshape(len(disp),-1,1), disp.reshape(len(disp),1,-1)).reshape(len(disp),-1)

    return np.hstack((v1, v2, disp, lend, B))


@numbafy(cast_args=['data'], keep_torch=True, ref_arg='data')
def get_voxel_edge_features(data, edge_index):
    """
    Function that returns a tensor of edge features for each of the
    edges connecting voxels in the graph.

    Parameters
    ----------
        data (np.ndarray)      : (N,8) [x, y, z, batchid, value, id, groupid, shape]
        edge_index (np.ndarray): (2,E) Incidence matrix
    Returns
    -------
        np.ndarray: (E,19) Tensor of edge features (displacement, orientation)
    """
    return _get_voxel_edge_features(data, edge_index)


@nb.njit(parallel=True, cache=True)
def _get_voxel_edge_features(data: nb.float32[:,:],
                         edge_index: nb.int64[:,:]) -> nb.float32[:,:]:
    feats = np.empty((len(edge_index), 19), dtype=data.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxel coordinates
        xi = data[edge_index[k,0]][:, COORD_COLS]
        xj = data[edge_index[k,1]][:, COORD_COLS]

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


@numbafy(cast_args=['voxels'], list_args=['clusts'])
def get_edge_distances(voxels, clusts, edge_index):
    """
    For each edge, finds the closest points of approach (CPAs) between the
    the two voxel clusters it connects, and the distance that separates them.

    Parameters
    ----------
        voxels (np.ndarray)    : (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray])  : (C) List of arrays of voxel IDs in each cluster
        edge_index (np.ndarray): (E,2) Incidence matrix
    Returns
    -------
        np.ndarray: (E) List of edge lengths
        np.ndarray: (E) List of voxel IDs corresponding to the first edge cluster CPA
        np.ndarray: (E) List of voxel IDs corresponding to the second edge cluster CPA
    """
    return _get_edge_distances(voxels, clusts, edge_index)

@nb.njit(parallel=True, cache=True)
def _get_edge_distances(voxels: nb.float32[:,:],
                        clusts: nb.types.List(nb.int64[:]),
                        edge_index:  nb.int64[:,:]) -> (nb.float32[:], nb.int64[:], nb.int64[:]):

    resi, resj = np.empty(len(edge_index), dtype=np.int64), np.empty(len(edge_index), dtype=np.int64)
    lend = np.empty(len(edge_index), dtype=np.float32)
    for k in nb.prange(len(edge_index)):
        i, j = edge_index[k]
        if i == j:
            ii = jj = 0
            lend[k] = 0.
        else:
            dist_mat = nbl.cdist(voxels[clusts[i]], voxels[clusts[j]])
            idx = np.argmin(dist_mat)
            ii, jj = idx//len(clusts[j]), idx%len(clusts[j])
            lend[k] = dist_mat[ii, jj]
        resi[k] = clusts[i][ii]
        resj[k] = clusts[j][jj]

    return lend, resi, resj


@numbafy(cast_args=['voxels'], list_args=['clusts'])
def inter_cluster_distance(voxels, clusts, batch_ids=None, mode='voxel', algorithm='brute', return_index=False):
    """
    Finds the inter-cluster distance between every pair of clusters within
    each batch, returned as a block-diagonal matrix.

    Parameters
    ----------
        voxels (torch.tensor) : (N,3) Tensor of voxel coordinates
        clusts ([np.ndarray]) : (C) List of arrays of voxel IDs in each cluster
        batch_ids (np.ndarray): (C) List of cluster batch IDs
        mode (str)            : Eiher use closest voxel distance (`voxel`) or centroid distance (`centroid`)
        algorithm (str)       : `brute` is exact but slow, `recursive` uses a fast but approximate proxy
        return_index (bool)   : If True, returns the combined index of the closest voxel pair
    Returns
    -------
        torch.tensor: (C,C) Tensor of pair-wise cluster distances
    """
    # If there is no batch_ids provided, assign 0 to all clusters
    if batch_ids is None:
        batch_ids = np.zeros(len(clusts), dtype=np.int64) 

    if not return_index:
        return _inter_cluster_distance(voxels, clusts, batch_ids, mode, algorithm)
    else:
        assert mode == 'voxel', 'Cannot return index for centroid method'
        return _inter_cluster_distance_index(voxels, clusts, batch_ids, algorithm)

@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance(voxels: nb.float32[:,:],
                            clusts: nb.types.List(nb.int64[:]),
                            batch_ids: nb.int64[:],
                            mode: str = 'voxel',
                            algorithm: str = 'brute') -> nb.float32[:,:]:

    assert len(clusts) == len(batch_ids)
    dist_mat = np.zeros((len(batch_ids), len(batch_ids)), dtype=voxels.dtype)
    indxi, indxj = complete_graph(batch_ids, directed=True)
    if mode == 'voxel':
        for k in nb.prange(len(indxi)):
            i, j = indxi[k], indxj[k]
            dist_mat[i,j] = dist_mat[j,i] = nbl.closest_pair(voxels[clusts[i]], voxels[clusts[j]], algorithm)[-1]
    elif mode == 'centroid':
        centroids = np.empty((len(batch_ids), voxels.shape[1]), dtype=voxels.dtype)
        for i in nb.prange(len(batch_ids)):
            centroids[i] = nbl.mean(voxels[clusts[i]], axis=0)
        for k in nb.prange(len(indxi)):
            i, j = indxi[k], indxj[k]
            dist_mat[i,j] = dist_mat[j,i] = np.sqrt(np.sum((centroids[j]-centroids[i])**2))
    else:
        raise ValueError('Inter-cluster distance mode not supported')

    return dist_mat


@nb.njit(parallel=True, cache=True)
def _inter_cluster_distance_index(voxels: nb.float32[:,:],
                                  clusts: nb.types.List(nb.int64[:]),
                                  batch_ids: nb.int64[:],
                                  algorithm: str = 'brute') -> (nb.float32[:,:], nb.int64[:,:]):

    assert len(clusts) == len(batch_ids)
    dist_mat = np.zeros((len(batch_ids), len(batch_ids)), dtype=voxels.dtype)
    closest_index = np.empty((len(batch_ids), len(batch_ids)), dtype=nb.int64)
    for i in range(len(clusts)):
        closest_index[i,i] = i
    indxi, indxj = complete_graph(batch_ids, directed=True)
    for k in nb.prange(len(indxi)):
        i, j = indxi[k], indxj[k]
        ii, jj, dist = nbl.closest_pair(voxels[clusts[i]], voxels[clusts[j]], algorithm)
        index = ii*len(clusts[j]) + jj

        closest_index[i,j] = closest_index[j,i] = index
        dist_mat[i,j] = dist_mat[j,i] = dist

    return dist_mat, closest_index


@numbafy(cast_args=['graph'])
def get_fragment_edges(graph, clust_ids):
    """
    Function that converts a set of edges between cluster ids
    to a set of edges between fragment ids (ordering in list)

    Parameters
    ----------
        graph (np.ndarray)    : (E,2) Tensor of [clust_id_1, clust_id_2]
        clust_ids (np.ndarray): (C) List of fragment cluster ids
        batch_ids (np.ndarray): (C) List of fragment batch ids
    Returns
    -------
        np.ndarray: (E,2) Tensor of true edges [frag_id_1, frag_id2]
    """
    return _get_fragment_edges(graph, clust_ids)

@nb.njit(cache=True)
def _get_fragment_edges(graph: nb.int64[:,:],
                        clust_ids: nb.int64[:]) -> nb.int64[:,:]:
    # Loop over the graph edges, find the fragment ids, append
    true_edges = np.empty((0,2), dtype=np.int64)
    for e in graph:
        n1 = np.where(clust_ids == e[0])[0]
        n2 = np.where(clust_ids == e[1])[0]
        if len(n1) and len(n2):
            true_edges = np.vstack((true_edges, np.array([[n1[0], n2[0]]], dtype=np.int64)))

    return true_edges
