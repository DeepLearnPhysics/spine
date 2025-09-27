"""Particle direction reconstruction module."""

import numpy as np

from spine.post.base import PostBase
from spine.utils.globals import TRACK_SHP
from spine.utils.gnn.cluster import get_cluster_directions

__all__ = ["DirectionProcessor"]


class DirectionProcessor(PostBase):
    """Reconstructs the direction of fragments and/or particles w.r.t. to
    their start (and end for tracks) points.

    This modules assign the `start_dir` and `end_dir` attributes.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "direction"

    # Alternative allowed names of the post-processor
    aliases = ("reconstruct_directions",)

    def __init__(
        self,
        radius=-1,
        optimize=True,
        obj_type="particle",
        truth_point_mode="points",
        run_mode="both",
    ):
        """Store the particle direction reconstruction parameters.

        Parameters
        ----------
        radius : float, default -1
            Radius around the start voxel to include in the direction estimate
        optimize : bool, default True
            Optimize the number of points involved in the direction estimate
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode)

        # Store the direction reconstruction parameters
        self.radius = radius
        self.optimize = optimize

        # Make sure the voxel coordinates are provided as a tensor
        if run_mode != "truth":
            self.update_keys({"points": True})
        if run_mode != "reco":
            self.update_keys({self.truth_point_key: True})

    def process(self, data):
        """Reconstruct the directions of all particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for k in self.obj_keys:
            # In order to parallelize the process, fetch all start points and
            # clusters once before passing them to the a parallelized function
            points_key = "points" if not "truth" in k else self.truth_point_key
            attrs, part_ids = [], []
            clusts, ref_points = [], []
            for obj in data[k]:
                # Get index
                index = self.get_index(obj)
                if not len(index):
                    continue

                # Store index, start point and the relevant mapping
                for side in ("start", "end"):
                    if side == "end" and obj.shape != TRACK_SHP:
                        continue

                    attr = f"reco_{side}_dir" if obj.is_truth else f"{side}_dir"
                    attrs.append(attr)
                    part_ids.append(obj.id)

                    clusts.append(index)
                    ref_points.append(getattr(obj, f"{side}_point"))

            # Check that there is at least one direction to compute
            if len(clusts) < 1:
                continue

            # Compute the direction vectors
            dirs = get_cluster_directions(
                data[points_key],
                np.vstack(ref_points),
                clusts,
                max_dist=self.radius,
                optimize=self.optimize,
            )

            # Assign directions to the appropriate particles
            for i, part_id in enumerate(part_ids):
                if attrs[i].startswith("start"):
                    setattr(data[k][part_id], attrs[i], dirs[i])
                else:
                    setattr(data[k][part_id], attrs[i], -dirs[i])
