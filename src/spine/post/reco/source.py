"""Source assignment modules.

This module implements routines to assign module/tpc sources when
they are not explicitely provided.
"""

import numpy as np

from spine.post.base import PostBase
from spine.utils.geo import Geometry

__all__ = ["SourceAssigner"]


class SourceAssigner(PostBase):
    """Class which assigns depositions in the detector to specific sources.

    Uses proximity to specific modules/TPCs to assign sources. Note that this is
    not exact for out-of-time activity for which the drift position is not
    representative of the real position of the deposition.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "source"

    def __init__(
        self,
        detector=None,
        file_path=None,
        run_mode="reco",
        truth_point_mode="points",
        **kwargs,
    ):
        """Initialize the source assigner

        Parameters
        ----------
        detector : str, optional
            Name of a recognized detector to the geometry from
        file_path : str, optional
            Path to a `.yaml` geometry configuration
        """
        # Initialize the parent class, store run mode
        super().__init__(run_mode=run_mode, truth_point_mode=truth_point_mode)

        # Initialize the geometry
        self.geo = Geometry(detector, file_path)

        # Make sure the necessary attributes are loaded
        self.run_mode = run_mode
        if run_mode != "truth":
            self.update_keys({"points": True})
        if run_mode != "reco":
            self.update_keys({self.truth_point_key: True})

    def process(self, data):
        """Assign sources to the relevant depositions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        dict
            Dictionary which sources
        """
        # Initialize output
        result = {}

        # Assign sources to reconstructed points, if needed
        if self.run_mode != "truth":
            result["sources"] = self.get_sources(data["points"])

        # Assign sources to truth points, if needed
        if self.run_mode != "reco":
            result[self.truth_source_key] = self.get_sources(data[self.truth_point_key])

        return result

    def get_sources(self, points):
        """Assign sources to one set of points.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Point coordinates

        Returns
        -------
        np.ndarray
            (N, 2) Module/TPC sources
        """
        # Assign sources
        module_ids = self.geo.get_closest_module(points)
        tpc_ids = self.geo.get_closest_tpc(points)
        sources = np.vstack((module_ids, tpc_ids)).T

        return sources
