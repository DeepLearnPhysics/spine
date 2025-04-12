"""Extensions to the basic Numba package."""

import numpy as np
import numba as nb


@nb.njit
def seed(seed: int) -> None:
    """Sets the numpy random seed for all Numba jitted functions.

    Note that setting the seed using `np.random.seed` outside a Numba jitted
    function does *not* set the seed of Numba functions.

    Parameters
    ----------
    seed : int
        Random number generator seed
    """
    np.random.seed(seed)


@nb.njit(cache=True)
def submatrix(x: nb.float32[:,:],
              index1: nb.int32[:],
              index2: nb.int32[:]) -> nb.float32[:,:]:
    """Numba implementation of matrix subsampling.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    index1 : np.ndarray
        (N') array of indices along axis 0 in the input matrix
    index2 : np.ndarray
        (M') array of indices along axis 1 in the input matrix

    Returns
    -------
    np.ndarray
        (N',M') array of values from the original matrix
    """
    subx = np.empty((len(index1), len(index2)), dtype=x.dtype)
    for i, i1 in enumerate(index1):
        for j, i2 in enumerate(index2):
            subx[i,j] = x[i1,i2]
    return subx


@nb.njit(cache=True)
def unique(x: nb.int32[:]) -> (nb.int32[:], nb.int64[:]):
    """Numba implementation of `np.unique(x, return_counts=True)`.

    Parameters
    ----------
    x : np.ndarray
        (N) array of values

    Returns
    -------
    np.ndarray
        (U) array of unique values
    np.ndarray
        (U) array of counts of each unique value in the original array
    """
    b = np.sort(x.flatten())
    unique = list(b[:1])
    counts = [1 for _ in unique]
    for v in b[1:]:
        if v != unique[-1]:
            unique.append(v)
            counts.append(1)
        else:
            counts[-1] += 1

    unique_np = np.empty(len(unique), dtype=x.dtype)
    counts_np = np.empty(len(counts), dtype=np.int32)
    for i in range(len(unique)):
        unique_np[i] = unique[i]
        counts_np[i] = counts[i]

    return unique_np, counts_np


@nb.njit(cache=True)
def mean(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """Numba implementation of `np.mean(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `mean` values
    """
    assert axis == 0 or axis == 1
    mean = np.empty(x.shape[1-axis], dtype=x.dtype)
    if axis == 0:
        for i in range(len(mean)):
            mean[i] = np.mean(x[:,i])
    else:
        for i in range(len(mean)):
            mean[i] = np.mean(x[i])
    return mean


@nb.njit(cache=True)
def norm(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """Numba implementation of `np.linalg.norm(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `norm` values
    """
    assert axis == 0 or axis == 1
    xnorm = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(xnorm)):
            xnorm[i] = np.linalg.norm(x[:,i])
    else:
        for i in range(len(xnorm)):
            xnorm[i] = np.linalg.norm(x[i])
    return xnorm


@nb.njit(cache=True)
def argmin(x: nb.float32[:,:],
           axis: nb.int32) -> nb.int32[:]:
    """Numba implementation of `np.argmin(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `argmin` values
    """
    assert axis == 0 or axis == 1
    argmin = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(argmin)):
            argmin[i] = np.argmin(x[:,i])
    else:
        for i in range(len(argmin)):
            argmin[i] = np.argmin(x[i])
    return argmin


@nb.njit(cache=True)
def argmax(x: nb.float32[:,:],
           axis: nb.int32) -> nb.int32[:]:
    """Numba implementation of `np.argmax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `argmax` values
    """
    assert axis == 0 or axis == 1
    argmax = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(argmax)):
            argmax[i] = np.argmax(x[:,i])

    else:
        for i in range(len(argmax)):
            argmax[i] = np.argmax(x[i])

    return argmax


@nb.njit(cache=True)
def amin(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """Numba implementation of `np.amin(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `min` values
    """
    assert axis == 0 or axis == 1
    xmin = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(xmin)):
            xmin[i] = np.min(x[:, i])

    else:
        for i in range(len(xmin)):
            xmin[i] = np.min(x[i])

    return xmin


@nb.njit(cache=True)
def amax(x: nb.float32[:,:],
         axis: nb.int32) -> nb.float32[:]:
    """Numba implementation of `np.amax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `max` values
    """
    assert axis == 0 or axis == 1
    xmax = np.empty(x.shape[1-axis], dtype=np.int32)
    if axis == 0:
        for i in range(len(xmax)):
            xmax[i] = np.max(x[:, i])

    else:
        for i in range(len(xmax)):
            xmax[i] = np.max(x[i])

    return xmax


@nb.njit(cache=True)
def all(x: nb.float32[:,:],
        axis: nb.int32) -> nb.boolean[:]:
    """Numba implementation of `np.all(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N, M) Array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N) or (M) array of `all` outputs
    """
    assert axis == 0 or axis == 1
    all = np.empty(x.shape[1-axis], dtype=np.bool_)
    if axis == 0:
        for i in range(len(all)):
            all[i] = np.all(x[:,i])

    else:
        for i in range(len(all)):
            all[i] = np.all(x[i])

    return all


@nb.njit(cache=True)
def contingency_table(x: nb.int32[:],
                      y: nb.int32[:],
                      nx: nb.int32=None,
                      ny: nb.int32=None) -> nb.int64[:, :]:
    """Build a contingency table for two sets of labels.

    Parameters
    ----------
    x : np.ndarray
        (N) Array of integrer values
    y : np.ndarray
        (M) Array of integrer values
    nx : int, optional
        Number of integer values allowed in `x`, N
    ny : int, optional
        Number of integer values allowd in `y`, M

    Returns
    -------
    np.ndarray
        (N, M) Contingency table
    """
    # If not provided, assume that the max label is the max of the label array
    if not nx:
        nx = np.max(x) + 1
    if not ny:
        ny = np.max(y) + 1

    # Bin the table
    table = np.zeros((nx, ny), dtype=np.int64)
    for i, j in zip(x, y):
        table[i, j] += 1

    return table


@nb.njit(cache=True)
def softmax(x: nb.float32[:,:],
            axis: nb.int32) -> nb.float32[:,:]:
    """
    Numba implementation of `scipy.special.softmax(x, axis)`.

    Parameters
    ----------
    x : np.ndarray
        (N,M) array of values
    axis : int
        Array axis ID

    Returns
    -------
    np.ndarray
        (N,M) Array of softmax scores
    """
    assert axis == 0 or axis == 1
    if axis == 0:
        xmax = amax(x, axis=0)
        logsumexp = np.log(np.sum(np.exp(x-xmax), axis=0)) + xmax
        return np.exp(x - logsumexp)
    else:
        xmax = amax(x, axis=1).reshape(-1,1)
        logsumexp = np.log(np.sum(np.exp(x-xmax), axis=1)).reshape(-1,1) + xmax
        return np.exp(x - logsumexp)


@nb.njit(cache=True)
def log_loss(label: nb.boolean[:],
             pred: nb.float32[:]) -> nb.float32:
    """Numba implementation of cross-entropy loss.

    Parameters
    ----------
    label : np.ndarray
        (N) array of boolean labels (0 or 1)
    pred : np.ndarray
        (N) array of float scores (between 0 and 1)

    Returns
    -------
    float
        Cross-entropy loss
    """
    if len(label) > 0:
        return -(np.sum(np.log(pred[label])) + np.sum(np.log(1.-pred[~label])))/len(label)
    else:
        return 0.


@nb.njit(cache=True)
def pdist(x: nb.float32[:,:],
          metric: str = 'euclidean') -> nb.float32[:,:]:
    """Numba implementation of
    `scipy.spatial.distance.pdist(x, p=2)` in 3D.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of point coordinates in the set
    metric : str, default 'euclidean'
        Distance metric

    Returns
    -------
    np.ndarray
        (N, N) array of pair-wise Euclidean distances
    """
    # Initialize the return matrix
    assert x.shape[1] == 3, "Only supports 3D points for now."
    res = np.empty((len(x), len(x)), dtype=x.dtype)

    if metric == 'euclidean':
        for i in range(x.shape[0]):
            res[i, i] = 0.
            for j in range(i+1, x.shape[0]):
                res[i, j] = res[j, i] = np.sqrt(
                        (x[i][0] - x[j][0])**2 +
                        (x[i][1] - x[j][1])**2 +
                        (x[i][2] - x[j][2])**2)

    elif metric == 'cityblock':
        for i in range(x.shape[0]):
            res[i, i] = 0.
            for j in range(i+1, x.shape[0]):
                res[i, j] = res[j, i] = (
                        abs(x[i][0] - x[j][0]) +
                        abs(x[i][1] - x[j][1]) +
                        abs(x[i][2] - x[j][2]))

    elif metric == 'chebyshev':
        for i in range(x.shape[0]):
            res[i, i] = 0.
            for j in range(i+1, x.shape[0]):
                res[i, j] = res[j, i] = max(
                        abs(x[i][0] - x[j][0]),
                        abs(x[i][1] - x[j][1]),
                        abs(x[i][2] - x[j][2]))

    else:
        raise ValueError("Distance metric not recognized.")

    return res


@nb.njit(cache=True)
def cdist(x1: nb.float32[:,:],
          x2: nb.float32[:,:]) -> nb.float32[:,:]:
    """Numba implementation of Euclidean
    `scipy.spatial.distance.cdist(x, p=2)` in 1D, 2D or 3D.

    Parameters
    ----------
    x1 : np.ndarray
        (N,d) array of point coordinates in the first set
    x2 : np.ndarray
        (M,d) array of point coordinates in the second set

    Returns
    -------
    np.ndarray
        (N,M) array of pair-wise Euclidean distances
    """
    dim = x1.shape[1]
    assert dim and dim < 4, 'Only supports point dimensions up to 3'
    res = np.empty((x1.shape[0], x2.shape[0]), dtype=x1.dtype)
    if dim == 1:
        for i1 in range(x1.shape[0]):
            for i2 in range(x2.shape[0]):
                res[i1,i2] = abs(x1[i1][0] - x2[i2][0])

    elif dim == 2:
        for i1 in range(x1.shape[0]):
            for i2 in range(x2.shape[0]):
                res[i1,i2] = np.sqrt(
                        (x1[i1][0] - x2[i2][0])**2 +
                        (x1[i1][1] - x2[i2][1])**2)

    elif dim == 3:
        for i1 in range(x1.shape[0]):
            for i2 in range(x2.shape[0]):
                res[i1,i2] = np.sqrt(
                        (x1[i1][0]-x2[i2][0])**2 +
                        (x1[i1][1]-x2[i2][1])**2 +
                        (x1[i1][2]-x2[i2][2])**2)

    return res


@nb.njit(cache=True)
def radius_graph(x: nb.float32[:,:],
                 radius: nb.float32,
                 metric: str = 'euclidean') -> nb.float32[:,:]:
    """Numba implementation of a radius-graph construction.

    This function generates a list of edges in a graph which connects all nodes
    within some radius R of each other.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of node coordinates
    radius : float
        Radius within which to build connections in the graph
    metric : str, default 'euclidean'
        Distance metric

    Returns
    -------
    np.ndarray
        (E, 2) array of edges in the radius graph
    """
    # Initialize an empty list of edges to add to the graph
    assert x.shape[1] == 3, "Only supports 3D points for now."
    edges = nb.typed.List.empty_list(nb.int64[:])

    if metric == 'euclidean':
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                dist = np.sqrt(
                        (x[i][0] - x[j][0])**2 +
                        (x[i][1] - x[j][1])**2 +
                        (x[i][2] - x[j][2])**2)
                if dist <= radius:
                    edges.append(np.array([i, j]))

    elif metric == 'cityblock':
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                dist = (
                        abs(x[i][0] - x[j][0]) +
                        abs(x[i][1] - x[j][1]) +
                        abs(x[i][2] - x[j][2]))
                if dist <= radius:
                    edges.append(np.array([i, j]))

    elif metric == 'chebyshev':
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                dist = max(
                        abs(x[i][0] - x[j][0]),
                        abs(x[i][1] - x[j][1]),
                        abs(x[i][2] - x[j][2]))
                if dist <= radius:
                    edges.append(np.array([i, j]))

    else:
        raise ValueError("Distance metric not recognized.")

    edge_index = np.empty((len(edges), 2), dtype=np.int64)
    for i, e in enumerate(edges):
        edge_index[i] = e

    return edge_index


@nb.njit(cache=True)
def union_find(edge_index: nb.int64[:,:],
               count: nb.int64,
               return_inverse: bool = True) -> nb.int64[:]:
    """Numba implementation of the Union-Find algorithm.

    This function assigns a group to each node in a graph, provided
    a set of edges connecting the nodes together.

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) List of edges (sparse adjacency matrix)
    count : int
        Number of nodes in the graph, C
    return_inverse : bool, default True
        Make sure the group IDs range from 0 to N_groups-1

    Returns
    -------
    np.ndarray
        (C) Group assignments for each of the nodes in the graph
    Dict[int, np.ndarray]
        Dictionary which maps groups to indexes
    """
    labels = np.arange(count)
    groups = {i: np.array([i]) for i in labels}
    for e in edge_index:
        li, lj = labels[e[0]], labels[e[1]]
        if li != lj:
            labels[groups[lj]] = li
            groups[li] = np.concatenate((groups[li], groups[lj]))
            del groups[lj]

    if return_inverse:
        mask = np.zeros(count, dtype=np.bool_)
        mask[labels] = True
        mapping = np.empty(count, dtype=labels.dtype)
        mapping[mask] = np.arange(np.sum(mask))
        labels = mapping[labels]

    return labels, groups


@nb.njit(cache=True)
def dbscan(x: nb.float32[:, :],
           eps: nb.float32,
           metric: str = 'euclidean') -> nb.float32[:]:
    """Runs DBSCAN on 3D points and returns the group assignments.

    Notes
    -----
    The traditional 'min_samples' is always set to 1 here.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of point coordinates
    eps : float
        Distance below which two points are considered neighbors
    metric : str, default 'euclidean'
        Distance metric used to compute pdist

    Returns
    -------
    np.ndarray
        (N) Group assignments
    """
    # Produce a sparse adjacency matrix (edge index)
    edge_index = radius_graph(x, eps, metric)

    # Build groups
    return union_find(edge_index, len(x), return_inverse=True)[0]


@nb.njit(cache=True)
def principal_components(x: nb.float32[:,:]) -> nb.float32[:,:]:
    """Computes the principal components of a point cloud by computing the
    eigenvectors of the centered covariance matrix.

    Parameters
    ----------
    x : np.ndarray
        (N, d) Coordinates in d dimensions

    Returns
    -------
    np.ndarray
        (d, d) List of principal components (row-ordered)
    """
    # Get covariance matrix
    A = np.cov(x.T, ddof = len(x) - 1).astype(x.dtype) # Casting needed...

    # Get eigenvectors
    _, v = np.linalg.eigh(A)
    v = np.ascontiguousarray(np.fliplr(v).T)

    return v


@nb.njit(cache=True)
def farthest_pair(x: nb.float32[:,:],
                  algorithm: str = 'brute') -> (nb.int32, nb.int32, nb.float32):
    """Algorithm which finds the two points which are farthest from each other
    in a set.

    Two algorithms:
    - `brute`: compute pdist, use argmax
    - `recursive`: Start with the first point in one set, find the farthest
                   point in the other, move to that point, repeat. This
                   algorithm is *not* exact, but a good and very quick proxy.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of point coordinates
    algorithm : str
        Name of the algorithm to use: `brute` or `recursive`

    Returns
    -------
    int
        ID of the first point that makes up the pair
    int
        ID of the second point that makes up the pair
    float
        Distance between the two points
    """
    if algorithm == 'brute':
        dist_mat = pdist(x)
        index = np.argmax(dist_mat)
        idxs = [index//x.shape[0], index%x.shape[0]]
        dist = dist_mat[idxs[0], idxs[1]]

    elif algorithm == 'recursive':
        centroid = mean(x, 0)
        start_idx = np.argmax(cdist(centroid.reshape(1, -1), x))
        idxs, subidx, dist, tempdist = [start_idx, start_idx], 0, 0., -1.
        while dist > tempdist:
            tempdist = dist
            dists = cdist(np.ascontiguousarray(x[idxs[subidx]]).reshape(1,-1), x).flatten()
            idxs[~subidx] = np.argmax(dists)
            dist = dists[idxs[~subidx]]
            subidx = ~subidx

    else:
        raise ValueError("Algorithm not supported")

    return idxs[0], idxs[1], dist


@nb.njit(cache=True)
def closest_pair(x1: nb.float32[:,:],
                 x2: nb.float32[:,:],
                 algorithm: bool = 'brute',
                 seed: bool = True) -> (nb.int32, nb.int32, nb.float32):
    """Algorithm which finds the two points which are closest to each other
    from two separate sets.

    Two algorithms:
    - `brute`: compute cdist, use argmin
    - `recursive`: Start with one point in one set, find the closest
                   point in the other set, move to theat point, repeat. This
                   algorithm is *not* exact, but a good and very quick proxy.

    Parameters
    ----------
    x1 : np.ndarray
        (Nx3) array of point coordinates in the first set
    x1 : np.ndarray
        (Nx3) array of point coordinates in the second set
    algorithm : str
        Name of the algorithm to use: `brute` or `recursive`
    seed : bool
        Whether or not to use the two farthest points in one set to seed the recursion

    Returns
    -------
    int
        ID of the first point that makes up the pair
    int
        ID of the second point that makes up the pair
    float
        Distance between the two points
    """
    # Find the two points in two sets of points that are closest to each other
    if algorithm == 'brute':
        # Compute every pair-wise distances between the two sets
        dist_mat = cdist(x1, x2)

        # Select the closest pair of point
        index = np.argmin(dist_mat)
        idxs = [index//dist_mat.shape[1], index%dist_mat.shape[1]]
        dist = dist_mat[idxs[0], idxs[1]]

    elif algorithm == 'recursive':
        # Pick the point to start iterating from
        xarr = [x1, x2]
        idxs, set_id, dist, tempdist = [0, 0], 0, 1e9, 1e9+1.
        if seed:
            # Find the end points of the two sets
            for i, x in enumerate(xarr):
                seed_idxs    = np.array(farthest_pair(xarr[i], 'recursive')[:2])
                seed_dists   = cdist(xarr[i][seed_idxs], xarr[~i])
                seed_argmins = argmin(seed_dists, axis=1)
                seed_mins    = np.array([seed_dists[0][seed_argmins[0]],
                                         seed_dists[1][seed_argmins[1]]])
                if np.min(seed_mins) < dist:
                    set_id = ~i
                    seed_choice = np.argmin(seed_mins)
                    idxs[int(~set_id)] = seed_idxs[seed_choice]
                    idxs[int(set_id)] = seed_argmins[seed_choice]
                    dist = seed_mins[seed_choice]

        # Find the closest point in the other set, repeat until convergence
        while dist < tempdist:
            tempdist = dist
            dists = cdist(np.ascontiguousarray(xarr[set_id][idxs[set_id]]).reshape(1,-1), xarr[~set_id]).flatten()
            idxs[~set_id] = np.argmin(dists)
            dist = dists[idxs[~set_id]]
            subidx = ~set_id
    else:
        raise ValueError("Algorithm not supported")

    return idxs[0], idxs[1], dist
