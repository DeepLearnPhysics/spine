"""PPN point construction module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from spine.constants import COORD_COLS, PPN_SHAPE_COL
from spine.post.base import PostBase
from spine.utils.ppn import PPNPredictor

__all__ = ["PPNProcessor"]


class PPNProcessor(PostBase):
    """Run the PPN post-processing function to produce PPN candidate points
    from the raw PPN output.

    If requested, for each particle, match ppn_points that have hausdorff
    distance less than a threshold and update the particle `ppn_candidates`
    attribute in place.

    If `restrict_shape` is `True`, points will be matched to particles with
    the same predicted semantic type only.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "ppn"

    # Alternative allowed names of the post-processor
    aliases = ("get_ppn_candidates",)

    # Set of data keys needed for this post-processor to operate
    _keys = (
        ("segmentation", True),
        ("ppn_points", True),
        ("ppn_coords", True),
        ("ppn_masks", True),
        ("ppn_classify_endpoints", False),
    )

    def __init__(
        self,
        assign_to_particles: bool = False,
        restrict_shape: bool = False,
        match_threshold: float = 2.0,
        **ppn_pred_cfg: Any,
    ) -> None:
        """Store the `get_ppn_predictions` keyword arguments.

        Parameters
        ----------
        assign_to_particles : bool, default False
            If `True`, will assign PPN candidates to particle objects
        restrict_shape : bool, default False
            If `True`, only associate PPN candidates with compatible shape
        match_threshold : float, default 2.
            Maximum distance required to assign ppn point to particle
        **ppn_pred_cfg : dict, optional
            Keyword arguments to pass to the `PPNPredictor` class

        """
        # Intialize the parent class
        obj_type = "particle" if assign_to_particles else None
        super().__init__(obj_type, "reco")

        # Store the relevant parameters
        self.assign_to_particles = assign_to_particles
        self.restrict_shape = restrict_shape
        self.match_threshold = match_threshold
        self.ppn_predictor = PPNPredictor(**ppn_pred_cfg)

    def process(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Produce PPN candidates for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Get the PPN candidates
        # TODO: remove requirement to nest output
        data_nest = {k: [v] for k, v in data.items()}
        ppn_pred = self.ppn_predictor(**data_nest)[0]
        result = {"ppn_pred": ppn_pred}

        # If requested, assign PPN candidates to particles
        if self.assign_to_particles:
            ppn_points = ppn_pred[:, COORD_COLS]
            for part in data["reco_particles"]:
                # Get the valid list of candidates
                valid_index = np.arange(len(ppn_pred))
                if not self.restrict_shape:
                    candidates = ppn_points
                else:
                    valid_index = np.where(ppn_pred[:, PPN_SHAPE_COL] == part.shape)[0]
                    candidates = ppn_points[valid_index]

                # Restrict to points that are sufficiently close
                dists = np.min(cdist(candidates, part.points), axis=1)
                dist_index = np.where(dists < self.match_threshold)[0]

                # Compute and store point matches to this particle
                matches = candidates[dist_index]
                part.ppn_points = matches
                if not self.restrict_shape:
                    part.ppn_ids = dist_index
                else:
                    part.ppn_ids = valid_index[dist_index]

        return result
