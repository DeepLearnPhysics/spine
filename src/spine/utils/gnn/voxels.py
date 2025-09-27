"""Module with functions that operate on single voxels in the context of GNNs."""

import numba as nb
import numpy as np

import spine.math as sm
from spine.data import TensorBatch
from spine.utils.globals import COORD_COLS
from spine.utils.jit import numbafy


def get_voxel_features_batch(data, max_dist=5.0):
    """Returns an array of features for each voxel.

    The basic 16 geometric features are composed of:
    - Voxel coordinates
    - Covariance matrix of its neighborhood (3)
    - Principal axis of its neighborhood (3)
    - Voxel count in its neighborhood (1)

    The neighborhood of the voxel i defined as all voxels within some distance
    of the voxel to get features for.

    Parameters
    ----------
    data : TensorBatch
        Batch of cluster label data tensor
    max_dist : float, default 5.0
        Neighborhood radius

    Returns
    -------
    np.ndarray
        (C, N_c) Tensor of voxels features
    """
    feats = get_voxel_features(data.tensor, max_dist)

    return TensorBatch(feats, data.counts)


@numbafy(cast_args=["data"], keep_torch=True, ref_arg="data")
def get_voxel_features(data, max_dist=5.0):
    """Returns an array of features for each voxel.

    The basic 16 geometric features are composed of:
    - Voxel coordinates
    - Covariance matrix of its neighborhood (3)
    - Principal axis of its neighborhood (3)
    - Voxel count in its neighborhood (1)

    The neighborhood of the voxel i defined as all voxels within some distance
    of the voxel to get features for.

    Parameters
    ----------
    data : np.ndarray
        Cluster label data tensor
    max_dist : float, default 5.0
        Neighborhood radius

    Returns
    -------
    np.ndarray
        (C, N_c) Tensor of voxels features
    """
    return _get_voxel_features(voxels, max_dist)


@nb.njit(parallel=True, cache=True)
def _get_voxel_features(data: nb.float32[:, :], max_dist=5.0):

    # Compute intervoxel distance matrix
    voxels = data[:, COORD_COLS]
    dist_mat = sm.distance.cdist(voxels, voxels)

    # Get local geometrical features for each voxel
    feats = np.empty((len(voxels), 16), dtype=data.dtype)
    for k in nb.prange(len(voxels)):

        # Restrict the points to the neighborood of the voxel
        voxel = voxels[k]
        x = voxels[dist_mat[k] < max_dist]

        # Get orientation matrix
        A = np.cov(x.T, ddof=len(x) - 1).astype(x.dtype)

        # Center data around voxel
        x = x - voxel

        # Get eigenvectors, normalize orientation matrix and eigenvalues to
        # largest. If points are superimposed, i.e. if the largest eigenvalue
        # != 0, no need to keep going
        w, v = np.linalg.eigh(A)
        if w[2] == 0.0:
            feats[k] = np.concatenate((center, np.zeros(12), np.array([len(clust)])))
            continue
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]

        # Get the principal direction, identify the direction of the spread
        v0 = v[:, 2]

        # Projection all points, x, along the principal axis
        x0 = x.dot(v0)

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
        feats[k] = np.concatenate((voxel, B.flatten(), v0, np.array([len(x)])))

    return np.vstack(feats)
