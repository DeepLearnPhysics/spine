"""Geometric node features for voxel-level GrapPA graphs."""

from __future__ import annotations

import numba as nb
import numpy as np
import torch

import spine.math as sm
from spine.data import ClusterLabelBatch, EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.jit import numbafy

__all__ = [
    "VoxelGeoNodeEncoder",
    "get_voxel_edge_features",
    "get_voxel_edge_features_batch",
    "get_voxel_features",
    "get_voxel_features_batch",
]


def get_voxel_edge_features_batch(
    data: TensorBatch, edge_index: EdgeIndexBatch
) -> TensorBatch:
    """Batched version of :func:`get_voxel_edge_features`.

    Parameters
    ----------
    data : TensorBatch
        Batch containing one three-dimensional coordinate group.
    edge_index : EdgeIndexBatch
        Batched ``(2, E)`` sparse incidence matrix.

    Returns
    -------
    TensorBatch
        ``(E, 19)`` edge features between voxels.
    """
    directed = edge_index.directed
    index = edge_index.index_t if directed else edge_index.directed_index_t
    counts = edge_index.counts if directed else edge_index.directed_counts
    features = get_voxel_edge_features(data.coords.tensor, index)

    return TensorBatch(features, counts)


@numbafy(
    cast_args=["coordinates", "edge_index"],
    keep_torch=True,
    ref_arg="coordinates",
)
def get_voxel_edge_features(
    coordinates: np.ndarray, edge_index: np.ndarray
) -> np.ndarray:
    """Build geometric features for edges connecting individual voxels.

    The edge features (N_e = 19) include (in that order):
    - Coordinates of the source voxel (3)
    - Coordinates of the target voxel (3)
    - Displacement vector between the two aforementioned voxels (3)
    - Magnitude of the displacement vector (1)
    - Outer product of the displacement vector (9)

    Parameters
    ----------
    coordinates : Union[np.ndarray, torch.Tensor]
        ``(N, 3)`` spatial coordinates.
    edge_index : Union[np.ndarray, torch.Tensor]
        ``(E, 2)`` incidence map between voxels.

    Returns
    -------
    np.ndarray
        ``(E, 19)`` tensor of edge features.
    """
    return _get_voxel_edge_features(coordinates, edge_index)


@nb.njit(parallel=True, cache=True)
def _get_voxel_edge_features(
    coordinates: nb.float32[:, :], edge_index: nb.int64[:, :]
) -> nb.float32[:, :]:
    """Numba implementation of voxel-edge geometric features."""
    features = np.empty((len(edge_index), 19), dtype=coordinates.dtype)
    for k in nb.prange(len(edge_index)):
        # Get the voxel coordinates
        xi = coordinates[edge_index[k, 0]]
        xj = coordinates[edge_index[k, 1]]

        # Displacement
        disp = xj - xi

        # Distance
        lend = np.linalg.norm(disp)
        if lend > 0:
            disp = disp / lend

        # Outer product
        outer = np.outer(disp, disp).flatten()

        features[k] = np.concatenate(
            (xi, xj, disp, np.array([lend], dtype=coordinates.dtype), outer)
        )

    return features


def get_voxel_features_batch(data: TensorBatch, max_dist: float = 5.0) -> TensorBatch:
    """Compute local geometric features without mixing batch entries.

    Each 16-component row contains the voxel coordinate, normalized local
    covariance matrix, weighted principal axis, and neighborhood population.

    Parameters
    ----------
    data : TensorBatch
        Batched sparse point data with one three-dimensional coordinate group.
    max_dist : float, default 5
        Open radius defining the local voxel neighborhood.
    """
    if max_dist <= 0.0:
        raise ValueError("`max_dist` must be positive.")

    coordinates = data.coords
    features = [
        get_voxel_features(coordinates[batch_id], max_dist)
        for batch_id in range(data.batch_size)
    ]
    if data.is_numpy:
        values = (
            np.concatenate(features)
            if features
            else np.empty((0, 16), dtype=data.dtype)
        )
    else:
        values = (
            torch.cat(features)
            if features
            else torch.empty((0, 16), dtype=data.dtype, device=data.device)
        )
    return TensorBatch(values, data.counts)


@numbafy(cast_args=["voxels"], keep_torch=True, ref_arg="voxels")
def get_voxel_features(voxels: np.ndarray, max_dist: float = 5.0) -> np.ndarray:
    """Compute 16 local geometric features for each input voxel."""
    if max_dist <= 0.0:
        raise ValueError("`max_dist` must be positive.")
    return _get_voxel_features(voxels, max_dist)


@nb.njit(parallel=True, cache=True)
def _get_voxel_features(voxels: nb.float32[:, :], max_dist: float) -> np.ndarray:
    """Numba implementation of local voxel geometry."""
    if voxels.shape[1] != 3:
        raise ValueError("Voxel geometric features require three coordinates.")

    dist_mat = sm.distance.cdist(voxels, voxels)
    features = np.zeros((len(voxels), 16), dtype=voxels.dtype)
    for index in nb.prange(len(voxels)):
        voxel = voxels[index]
        neighborhood = voxels[dist_mat[index] < max_dist]
        count = len(neighborhood)
        features[index, :3] = voxel
        features[index, 15] = count
        if count < 2:
            continue

        # Estimate covariance explicitly: unlike np.cov, this remains stable
        # and well-defined for the small neighborhoods common at image edges.
        centered = neighborhood - sm.mean(neighborhood, 0)
        covariance = centered.T.dot(centered) / count
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
        eigenvalues = eigenvalues.astype(voxels.dtype)
        eigenvectors = eigenvectors.astype(voxels.dtype)
        scale = eigenvalues[2]
        if scale <= 0.0:
            continue

        features[index, 3:12] = (covariance / scale).flatten()
        direction = eigenvectors[:, 2]
        displacement = neighborhood - voxel
        projection = displacement.dot(direction)
        transverse = displacement - np.outer(projection, direction)
        transverse_norm = sm.linalg.norm(transverse, 1)
        if np.dot(projection, transverse_norm) < 0.0:
            direction = -direction
        features[index, 12:15] = (1.0 - eigenvalues[1] / scale) * direction

    return features


class VoxelGeoNodeEncoder(torch.nn.Module):
    """Encode singleton voxel nodes from their local image geometry."""

    name = "voxel_geometric"
    aliases = ("voxel_geo",)
    feature_size = 16

    def __init__(self, max_dist: float = 5.0) -> None:
        """Initialize the voxel neighborhood radius."""
        super().__init__()
        if max_dist <= 0.0:
            raise ValueError("`max_dist` must be positive.")
        self.max_dist = max_dist

    def forward(
        self,
        data: ClusterLabelBatch | TensorBatch,
        clusts: IndexBatch,
        **_kwargs: object,
    ) -> TensorBatch:
        """Return one feature row for every singleton cluster in ``clusts``."""
        if not np.all(clusts.to_numpy().single_counts == 1):
            raise ValueError("Voxel node encoding requires singleton clusters.")

        tensor_data = (
            data.to_tensor_batch() if isinstance(data, ClusterLabelBatch) else data
        )
        features = get_voxel_features_batch(tensor_data, self.max_dist)
        index = torch.as_tensor(
            clusts.full_index,
            dtype=torch.long,
            device=features.device,
        )
        return TensorBatch(features.torch_tensor()[index], clusts.counts)
