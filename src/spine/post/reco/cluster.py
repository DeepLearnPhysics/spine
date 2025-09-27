"""Clustering reconstruction module.

This module implements very basic clustering routines to build particles
without requiring model weights. This is useful to produce basic data
assessment metrics without relying on high-level reconstruction tools.
"""

import numpy as np

from spine.data import ObjectList
from spine.data.out import RecoParticle, TruthParticle
from spine.math.cluster import DBSCAN
from spine.math.decomposition import PCA
from spine.post.base import PostBase
from spine.utils.geo import Geometry
from spine.utils.globals import TRACK_SHP

__all__ = ["TrackClusterer"]


class TrackClusterer(PostBase):
    """Class which performs basic track clustering.

    Uses DBSCAN to build basic reconstructed particle objects.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "track_cluster"

    # Set of data keys needed for this post-processor to operate
    _keys = (("points", True), ("depositions", True), ("sources", False))

    # List of recognized volumes one can split between
    _split_volumes = ("tpc", "module")

    # List of recognized output particle classes
    _particle_classes = (("reco", RecoParticle), ("truth", TruthParticle))

    def __init__(
        self,
        eps=5.0,
        min_samples=1,
        metric="euclidean",
        min_size=10,
        max_rel_spread=0.1,
        max_axis_dist=None,
        split_volume=None,
        particle_type="reco",
    ):
        """Initialize the track factory.

        Parameters
        ----------
        eps : float, default 5.
            DBSCAN distance parameter for intial clustering (cm)
        min_samples : int, default 1
            DBSCAN min samples parameter for initial clustering
        metric : str, default 'euclidean'
            DBSCAN metric for intial clustering
        min_size : int, default 10
            Minimum track size, in number of hits
        max_rel_spread : float, default 0.1
            Maximum relative spread allowed to be considered a track
        max_axis_dist : float, optional
            Maximum track axis distance to filter points to, if requested
        split_volume : str, optional
            If specified, the track clusters are restricted within either
            a detecor 'module' or 'tpc'
        particle_type : str, default 'reco'
            Type of particle output by this algorithm
        """
        # Initialize DBSCAN
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        # Initialize PCA
        self.pca = PCA(3)

        # Save parameters
        self.min_size = min_size
        self.max_rel_spread = max_rel_spread
        self.max_axis_dist = max_axis_dist

        # Initialize the splitting procedure
        assert split_volume is None or split_volume in self._split_volumes, (
            f"Track clustering split volume not recognized: {split_volume}. "
            f"Must be one of {self._volume_modes}."
        )
        self.split_volume = split_volume

        # Check on which type of object to output
        assert particle_type in self.particle_classes, (
            f"Output particle type not recognized: {particle_type}. "
            f"Must be one of {self.particle_classes.keys()}"
        )
        self.particle_type = particle_type
        self.particle_class = self.particle_classes[particle_type]

    @property
    def particle_classes(self):
        """Dictionary which maps particle type to particle classes.

        Returns
        -------
        dict
            Dictionary of particle classes
        """
        return dict(self._particle_classes)

    def process(self, data):
        """Produce track clusters for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        dict
            Dictionary which reconstructed track instances
        """
        # Dispatch
        points, depositions = data["points"], data["depositions"]
        if self.split_volume is None:
            # If no splitting is required, feed to points to the algorithm
            clusts = self.process_volume(points)

        else:
            # If splitting is required, assign each point to a volume
            assert "sources" in data, (
                "Must provide sources to split points by " f"{self.split_volume}."
            )
            sources = data["sources"]

            if self.split_volume == "module":
                volume_ids = sources[:, 0]
            else:
                volume_ids = sources[:, 1]

            # Loop over unique volume IDs, cluster independently
            clusts = []
            for volume_id in np.unique(volume_ids):
                # Produce an index for this volume
                volume_index = np.where(volume_ids == volume_id)[0]
                if len(volume_index) < self.min_size:
                    continue

                # Cluster
                volume_clusts = self.process_volume(points[volume_index])

                # Bring to global index, extend existing list
                volume_clusts = [volume_index[clust] for clust in volume_clusts]
                clusts.extend(volume_clusts)

        # Convert clusters to particle instances
        particles = []
        for i, index in enumerate(clusts):
            # Create particle
            particle = self.particle_class(
                id=i,
                index=index,
                shape=TRACK_SHP,
                points=points[index],
                depositions=depositions[index],
                sources=sources[index],
            )

            # Append
            particles.append(particle)

        # Return
        particles = ObjectList(particles, default=self.particle_class())

        return {f"{self.particle_type}_particles": particles}

    def process_volume(self, points):
        """Produce track clusters for one volume.

        Parameters
        ----------
        points : np.ndarray
            Point coordinates in the volume

        Returns
        -------
        List[np.ndarray]
            List of cluster indexes
        """
        # Fetch the list of points in the image
        clust_labels = self.dbscan.fit_predict(points)

        # Loop over the clusters and apply some quality cuts
        clusts = []
        for c in np.unique(clust_labels):
            # Skip unassigned points
            if c < 0:
                continue

            # Restrict points to the relevant cluster
            index_c = np.where(clust_labels == c)[0]
            points_c = points[index_c, :3]
            if len(index_c) < self.min_size:
                continue

            # If the relative spread of the points w.r.t. the main axis is too
            # large, skip
            axes, var = self.pca.fit(points_c)
            axis = axes[0]
            rel_spread = np.sqrt((var[1] + var[2]) / var[0])
            if rel_spread > self.max_rel_spread:
                continue

            # Project all points on the principal axis, filter out those too
            # far from the main axis
            if self.max_axis_dist is not None:
                cent = np.mean(points_c, axis=0)
                dists = np.linalg.norm(np.cross(points_c - cent, axis), axis=1)
                index_c = index_c[dists < self.max_axis_dist]
                if len(index_c) < self.min_size:
                    continue

            # Append cluster index
            clusts.append(index_c)

        return clusts
