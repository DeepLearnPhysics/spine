"""Module with a class that leverages particle endpoints to do clustering."""

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from spine.math.cluster import DBSCAN

__all__ = ["PointBreakClusterer"]


class PointBreakClusterer:
    """Leverages particles start/end point positions to break up instances
    of particles which touch (vertex, secondary interaction point, etc.).

    Two methods are supported: `masked_dbscan` and `closest_path`.

    The masked DBSCAN method proceeds as follows:
    - Break up the input point cloud into DBSCAN clusters
    - For each cluster, mask out regions around the particle end points
    - Run DBSCAN on the masked point cloud
    - Each new instance is a particle instance
    - Unlabeled, masked points are merged to the closest instance

    The closest path method proceeds as follows:
    - Break up the input point cloud into DBSCAN clusters
    - For each cluster, build a radius graph on its constituents
    - Find the closest graph paths for each pair of particle end points
    - The paths that belong to the minimum spanning tree are particles
    - Points from the cloud are assigned to a particle using their
      proximity to a particle path

    The latter only works on track clusters, not on EM showers.
    """

    def __init__(
        self,
        method="masked_dbscan",
        eps=1.8,
        min_samples=1,
        metric="euclidean",
        mask_radius=5.0,
    ):
        """Initialize the particle point-enhanced clustering algorithm.

        Parameters
        ----------
        method : str, default 'masked_dbscan'
            Clustering method
        eps : float, default 1.8
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.
        min_samples : int, default 1
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point.
        metric : str, default 'euclidean'
            Metric used to compute the pair-wise distances between space points
        mask_radius : float, default 5.0
            Radius to mask around each particle point
        """
        # Store the attributes
        self.method = method
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.mask_radius = mask_radius

        # Initialize the DBSCAN algorithm
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def __call__(self, voxels, points, method=None):
        """Produce instance clusters using the point-enhanced method.

        Parameters
        ----------
        voxels : np.ndarray
            (N, 3) Set of voxel coordinates
        points : np.ndarray
            (P, 3) Set of particle end points
        method : str, optional
            Override the method defined in the initialzer

        Returns
        -------
        np.ndarray
            (N) Array of cluster assignments for each voxel in the input
        """
        method = method if method is not None else self.method
        if method == "masked_dbscan":
            return self.get_masked_dbscan_labels(voxels, points)

        elif method == "closest_path":
            return self.get_closest_path_labels(voxels, points)

        else:
            raise ValueError(
                "Point-enhanced clustering method not recognized: "
                f"{self.method}. Should be one of 'masked_dbscan' or "
                "'closest_path'."
            )

    def get_masked_dbscan_labels(self, voxels, points):
        """Produce instance clusters using the masked-DBSCAN method.

        Parameters
        ----------
        voxels : np.ndarray
            (N, 3) Set of voxel coordinates
        points : np.ndarray
            (P, 3) Set of particle end points

        Returns
        -------
        np.ndarray
            (N) Array of cluster assignments for each voxel in the input
        """
        # Find voxels above a threshold distance from any particle end point
        pair_mat = cdist(voxels, points, metric=self.metric)
        dist_mask = np.all((pair_mat > self.mask_radius), axis=1)

        # Form preliminary clusters on the entire set of voxels, loop over them
        labels = self.dbscan.fit_predict(voxels)
        for l in np.unique(labels):
            # Find the list of masked and unmasked voxels in this cluster
            group_mask = labels == l
            active_index = np.where(dist_mask & group_mask)[0]
            passive_index = np.where(~dist_mask & group_mask)[0]

            if len(active_index):
                # If there are active pixels, run DBSCAN on the masked voxels
                labels_c = self.dbscan.fit_predict(voxels[active_index])
                labels[active_index] = np.max(labels) + 1 + labels_c

                if len(passive_index):
                    # For masked pixel, matched to the closest cluster
                    dist_mat = cdist(
                        voxels[active_index], voxels[passive_index], metric=self.metric
                    )
                    argmins = np.argmin(dist_mat, axis=0)
                    labels[passive_index] = labels[active_index][argmins]

        return labels

    def get_closest_path_labels(self, voxels, points):
        """Produce instance clusters using the closest-path method

        Parameters
        ----------
        voxels : np.ndarray
            (N, 3) Set of voxel coordinates
        points : np.ndarray
            (P, 3) Set of particle end points

        Returns
        -------
        np.ndarray
            (N) Array of cluster assignments for each voxel in the input
        """
        # Precompute the pair-wise distances between voxels and points
        pair_mat = cdist(voxels, points, metric=self.metric)

        # Form preliminary clusters on the entire set of voxels, loop over them
        labels = self.dbscan.fit_predict(voxels)
        for l in np.unique(labels):
            # Restrict voxel set and point set to those in the group
            group_mask = labels == l
            point_mask = np.min(pair_mat[group_mask], axis=0) < self.eps
            point_ids = np.unique(
                np.argmin(pair_mat[np.ix_(group_mask, point_mask)], axis=0)
            )
            if len(point_ids) > 2:
                # Build a graph on the group voxels that respect the DBSCAN distance scale
                dist_mat = cdist(
                    voxels[group_mask], voxels[group_mask], metric=self.metric
                )
                graph = dist_mat * (dist_mat < self.eps)
                cs_graph = scipy.sparse.csr_matrix(graph)

                # Find the shortest between each of the breaking points, identify segments that minimize absolute excursion
                graph_mat, predecessors = scipy.sparse.csgraph.shortest_path(
                    csgraph=cs_graph, directed=False, return_predecessors=True
                )
                break_ix = np.ix_(point_ids, point_ids)
                chord_mat = dist_mat[break_ix]
                mst_mat = scipy.sparse.csgraph.minimum_spanning_tree(
                    graph_mat[break_ix] - chord_mat + 1e-6
                ).toarray()
                mst_edges = np.vstack(np.where(mst_mat > 0)).T

                # Construct graph paths along the tree
                paths = [[] for _ in range(len(mst_edges))]
                for i, e in enumerate(mst_edges):
                    k, l = point_ids[e]
                    paths[i].append(l)
                    while l != k:
                        l = predecessors[k, l]
                        paths[i].append(l)

                # Find the path closest to each of the voxels in the group. If a path does not improve reachability, remove
                mindists = np.vstack([np.min(graph_mat[:, p], axis=1) for p in paths])
                mst_tort = mst_mat[mst_mat > 0] / chord_mat[mst_mat > 0]  # tau - 1
                tort_ord = np.argsort(-mst_tort)
                least_r = np.max(
                    np.min(mindists, axis=0)
                )  # Least reachable point distance
                path_mask = np.ones(len(mst_tort), dtype=np.bool)
                for i in range(len(mst_tort)):
                    if np.sum(path_mask) == 1:
                        break
                    path_mask[tort_ord[i]] = False
                    reach = np.max(np.min(mindists[path_mask], axis=0))
                    if reach > least_r:
                        path_mask[tort_ord[i]] = True

                # Associate voxels with closest remaining path
                mindists = np.vstack(
                    [
                        np.min(
                            graph_mat[:, p[min(1, len(p) - 2) : max(len(p) - 1, 2)]],
                            axis=1,
                        )
                        for i, p in enumerate(paths)
                        if path_mask[i]
                    ]
                )
                sublabels = np.argmin(mindists, axis=0)
                labels[group_mask] = max(labels) + 1 + sublabels

        return np.unique(labels, return_inverse=True)[1]
