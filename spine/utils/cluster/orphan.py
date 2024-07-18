"""Defines class used to assign orphaned points to a sensible cluster."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.cluster import DBSCAN

__all__ = ['OrphanAssigner']


class OrphanAssigner:
    """Clustering orphan assignment.

    This class takes care of finding the best match cluster ID for points that
    have not found a suitable group in the upstream clustering.

    This is a wrapper class for two `scikit-learn` classes:
    - :class:`KNeighborsClassifier`
    - :class:`RadiusNeighborsClassifier`
    """

    def __init__(self, mode, iterate=True, assign_all=True, **kwargs):
        """Initialize the orphan assigner.

        Parameters
        ----------
        mode : str
            Orphan assignment mode, one of 'knn' or 'radius'
        iterate : bool, default True
            Iterate the process until no additional orphans can be assigned
        assign_all : bool, default True
            If `True`, force assign all orphans to a cluster. In the 'knn' mode,
            this is guaranteed, provided there is at least one labeled point.
            In the 'radius' mode, this uses DBSCAN for outliers.
        **kwargs : dict
            Arguments to pass to the underlying classifier function
        """
        # Initialize the classifier
        self.mode = mode
        if mode == 'knn':
            self.classifier = KNeighborsClassifier(**kwargs)
        elif mode == 'radius':
            self.classifier = RadiusNeighborsClassifier(
                    outlier_label=-1, **kwargs)
        else:
            raise ValueError(
                     "The orphan assignment mode must be one of 'knn' or "
                    f"'radius', got '{mode}' instead.")

        # Store the extra parameters
        self.iterate = iterate
        self.assign_all = assign_all

        # If needed, initialize DBSCAN
        if mode == 'radius' and assign_all:
            self.dbscan = DBSCAN(
                    eps=self.classifier.radius, min_samples=1,
                    metric=self.classifier.metric, p=self.classifier.p)

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
        # Create a mask for orphaned points, throw if there are only orphans
        orphan_index = np.where(y == -1)[0]
        num_orphans = len(orphan_index)
        if (self.mode == 'knn' or not self.assign_all) and len(y) == num_orphans:
            raise RuntimeError(
                    "Cannot assign orphans without any valid labels.")

        # Loop until all there is no more orphans to assign
        y_updated = y.copy()
        while num_orphans:
            # Fit the classifier with the labeled points
            valid_index = np.where(y_updated > -1)[0]
            if not len(valid_index):
                break

            self.classifier.fit(X[valid_index], y_updated[valid_index])

            # Get the assignment for each of the orphaned points
            update = self.classifier.predict(X[orphan_index])

            # Update the labels accordingly
            y_updated[orphan_index] = update

            # If iterating is not required, break (iterating on kNN does nothing)
            if not self.iterate or self.mode == 'knn':
                break

            # If the number of orphans has not changed, no point in proceeding
            orphan_index = orphan_index[update < 0]
            if len(orphan_index) == num_orphans:
                break

            num_orphans = len(orphan_index)

        # If required, assign stragglers using DBSCAN
        if num_orphans and self.mode == 'radius' and self.assign_all:
            # Get the assignment for each of the orphaned points
            update = self.dbscan.fit(X[orphan_index]).labels_

            # Update the labels accordingly
            offset = np.max(y_updated) + 1
            y_updated[orphan_index] = offset + update

        return y_updated
