"""Delaunay graph constructor for GNNs."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

from spine.constants import COORD_COLS
from spine.data import IndexBatch, TensorBatch

from .base import GraphBase

__all__ = ["DelaunayGraph"]


class DelaunayGraph(GraphBase):
    """Connect clusters whose voxels share a Delaunay simplex.

    A triangulation is constructed independently for each batch entry. Voxel
    simplices are converted to unique cluster-pair edges; simplices containing
    voxels from only one cluster do not produce graph edges. Degenerate point
    clouds that Qhull cannot triangulate fall back to a complete graph over the
    clusters in that entry.
    """

    name = "delaunay"

    def generate(
        self,
        *,
        data: TensorBatch,
        clusts: IndexBatch,
        dist_mat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate batched Delaunay cluster graphs.

        Parameters
        ----------
        data : TensorBatch
            Batched voxel/value table.
        clusts : IndexBatch
            Batched cluster index list.
        dist_mat : np.ndarray, optional
            Pairwise distance matrix, unused by this graph.

        Returns
        -------
        tuple of np.ndarray
            Edge index with shape ``(2, E)`` and per-entry edge counts.
        """
        # Normalize the graph inputs and initialize per-event outputs
        data_array = data.numpy_tensor()
        batch_ids = np.asarray(clusts.batch_ids)
        edge_blocks: list[np.ndarray] = []
        edge_counts = np.zeros(clusts.batch_size, dtype=np.int64)

        # Triangulate each batch entry independently
        for batch_id in range(clusts.batch_size):
            cluster_ids = np.flatnonzero(batch_ids == batch_id)
            edges = self._generate_entry(
                data_array,
                clusts.index_list,
                cluster_ids,
            )
            edge_blocks.append(edges)
            edge_counts[batch_id] = edges.shape[1]

        # Concatenate nonempty event graphs into one batched edge index
        nonempty_blocks = [edges for edges in edge_blocks if edges.shape[1] > 0]
        if nonempty_blocks:
            edge_index = np.concatenate(nonempty_blocks, axis=1)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        return edge_index, edge_counts

    @staticmethod
    def _generate_entry(
        data: np.ndarray,
        clusters: Sequence[Any],
        cluster_ids: np.ndarray,
    ) -> np.ndarray:
        """Generate Delaunay edges for one batch entry."""
        # Graphs with fewer than two nodes contain no edges
        if len(cluster_ids) < 2:
            return np.empty((2, 0), dtype=np.int64)

        # Map each participating voxel back to its owning cluster
        voxel_indices = np.concatenate([clusters[index] for index in cluster_ids])
        voxel_clusters = np.concatenate(
            [
                np.full(len(clusters[index]), index, dtype=np.int64)
                for index in cluster_ids
            ]
        )
        points = data[voxel_indices][:, COORD_COLS]

        # Triangulate the voxel cloud, falling back for degenerate geometry
        try:
            simplices = Delaunay(points, qhull_options="QJ").simplices
        except QhullError:
            return DelaunayGraph._complete_entry(cluster_ids)

        # Convert voxel simplices to unique cluster-pair edges
        edge_pairs: set[tuple[int, int]] = set()
        for simplex in simplices:
            simplex_clusters = np.unique(voxel_clusters[simplex])
            edge_pairs.update(combinations(simplex_clusters.tolist(), 2))

        if not edge_pairs:
            return DelaunayGraph._complete_entry(cluster_ids)
        return np.asarray(sorted(edge_pairs), dtype=np.int64).T

    @staticmethod
    def _complete_entry(cluster_ids: np.ndarray) -> np.ndarray:
        """Return upper-triangular edges for one cluster collection."""
        return np.asarray(list(combinations(cluster_ids.tolist(), 2)), dtype=np.int64).T
