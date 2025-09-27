"""Defines class used to assign orphaned points to a sensible cluster."""

import numpy as np

from spine.math.cluster import DBSCAN
from spine.math.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

__all__ = ["OrphanAssigner"]


class OrphanAssigner:
    """Clustering orphan assignment.

    This class takes care of finding the best match cluster ID for points that
    have not found a suitable group in the upstream clustering.

    This is a wrapper class for two classes:
    - :class:`KNeighborsClassifier`
    - :class:`RadiusNeighborsClassifier`
    """

    def __init__(self, mode, assign_all=True, **kwargs):
        """Initialize the orphan assigner.

        Parameters
        ----------
        mode : str
            Orphan assignment mode, one of 'knn' or 'radius'
        assign_all : bool, default True
            If `True`, force assign all orphans to a cluster. In the 'knn' mode,
            this is guaranteed, provided there is at least one labeled point.
            In the 'radius' mode, this uses DBSCAN for outliers.
        **kwargs : dict
            Arguments to pass to the underlying classifier function
        """
        # Initialize the classifier
        self.mode = mode
        if mode == "knn":
            self.classifier = KNeighborsClassifier(**kwargs)
        elif mode == "radius":
            self.classifier = RadiusNeighborsClassifier(**kwargs)
        else:
            raise ValueError(
                "The orphan assignment mode must be one of 'knn' or "
                f"'radius'. Got '{mode}' instead."
            )

        # Store the extra parameter
        self.assign_all = assign_all

        # If needed, initialize DBSCAN
        if mode == "radius" and assign_all:
            radius = kwargs.get("radius")
            metric = kwargs.get("metric", "euclidean")
            self.dbscan = DBSCAN(
                eps=radius, min_samples=1, metric=metric, p=self.classifier.p
            )

    def __call__(self, X, y):
        """Place-holder for a function which assigns labels to orphan points.

        Parameters
        ----------
        X : np.ndarray
            (N, 3) Coordinates of the points in the image
        y : np.ndarray
            (N) Labels of the points (-1 if orphaned)

        Returns
        -------
        np.ndarray
            (M) Labels assigned to each of the orphans
        """
        # Create a mask to identify labeled and orphaned points
        orphan_mask = y == -1
        orphan_index = np.where(orphan_mask)[0]
        valid_index = np.where(~orphan_mask)[0]

        # Assign orphan points using the neighbor classifier
        labels, orphan_update = self.classifier.fit_predict(
            X[valid_index], y[valid_index], X[orphan_index]
        )

        y_updated = y.copy()
        y_updated[orphan_index] = labels
        orphan_index = orphan_index[orphan_update]

        # If required, assign stragglers using DBSCAN
        if len(orphan_index) and self.mode == "radius" and self.assign_all:
            # Get the assignment for each of the orphaned points
            update = self.dbscan.fit_predict(X[orphan_index])

            # Update the labels accordingly
            offset = np.max(y_updated) + 1
            y_updated[orphan_index] = offset + update

        return y_updated
