"""Simple wrapper for sklearn's DBSCAN to turn its label output into
a list of clusters in the form of point index lists."""

import numpy as np
from typing import List
from sklearn.cluster import DBSCAN


def dbscan_points(coordinates, eps=1.999, min_samples=1, metric='euclidean'):
    """Runs DBSCAN on an input point cloud.

    Returns the clusters as a list of indexes.

    Parameters
    ----------
    coordinates : np.ndarray
        Set of point coordinates
    eps : float, default 1.999
        Distance parameter of DBSCAN
    min_samples : int, default 1
        Minimum number of points in a cluster to be valid
    metric : str, default 'euclidean'
        Metric used to compute distances

    Returns
    -------
    List[np.ndarray]
        List of cluster indexes
    """
    # Initialize DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    # Build clusters
    labels = dbscan.fit(coordinates).labels_
    clusters = []
    for c in np.unique(labels):
        if c > -1:
            clusters.append(np.where(labels == c)[0])

    return clusters
