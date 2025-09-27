"""Track end point assignment module."""

from spine.post.base import PostBase
from spine.utils.globals import TRACK_SHP
from spine.utils.ppn import check_track_orientation_ppn
from spine.utils.tracking import check_track_orientation

__all__ = ["TrackExtremaProcessor"]


class TrackExtremaProcessor(PostBase):
    """Assigns track start point and end point."""

    # Name of the post-processor (as specified in the configuration)
    name = "track_extrema"

    # Alternative allowed names of the post-processor
    aliases = ("assign_track_extrema",)

    # Set of data keys needed for this post-processor to operate
    _keys = (("ppn_candidates", False),)

    def __init__(self, method="local", obj_type="particle", **kwargs):
        """Initialize the track end point assignment parameters.

        Parameters
        ----------
        method : str, default 'local'
            Algorithm to correct track startpoint/endpoint misplacement. The
            following methods are available:
            - local: computes local energy deposition density only at
            the extrema and chooses the higher one as the endpoint.
            - gradient: computes local energy deposition density throughout
            the track, computes the overall slope (linear fit) of the energy
            density variation to estimate the direction.
            - ppn: uses ppn candidate predictions (classify_endpoints) to
            assign start and endpoints.
        kwargs : dict
            Extra arguments to pass to the `check_track_orientation` or the
            `check_track_orientation_ppn' functions
        """
        # Initialize the parent class
        super().__init__(obj_type, "reco")

        # Store the orientation method and its arguments
        self.method = method
        self.kwargs = kwargs

    def process(self, data):
        """Assign track end points in one entry

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        for part in data["reco_particles"]:
            # Only process track objects
            if part.shape == TRACK_SHP:
                # Check if the end points need to be flipped
                if self.method in ["local", "gradient"]:
                    flip = not check_track_orientation(
                        part.points,
                        part.depositions,
                        part.start_point,
                        part.end_point,
                        self.method,
                        **self.kwargs,
                    )

                elif self.method == "ppn":
                    assert "ppn_candidates" in data, (
                        "Must run the `ppn` post-processor "
                        "before using PPN predictions to assign extrema."
                    )
                    flip = not check_track_orientation_ppn(
                        part.start_point, part.end_point, data["ppn_candidates"]
                    )

                else:
                    raise ValueError(
                        "Point assignment method not recognized: " f"{self.method}"
                    )

                # If needed, flip the end points and the directions
                if flip:
                    part.start_point, part.end_point = (
                        part.end_point,
                        part.start_point,
                    )
                    part.start_dir, part.end_dir = (-part.end_dir, -part.start_dir)
