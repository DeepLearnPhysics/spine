"""Particle identification modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spine.constants import TRACK_SHP
from spine.post.base import PostBase
from spine.utils.pid import TemplateParticleIdentifier

__all__ = ["PIDTemplateProcessor"]


class PIDTemplateProcessor(PostBase):
    """Produces particle species classification estimates based on dE/dx vs
    residual range templates of tracks.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "pid_template"

    def __init__(
        self,
        fill_per_pid: bool = False,
        obj_type: str | Sequence[str] | None = "particle",
        run_mode: str = "reco",
        truth_point_mode: str = "points",
        truth_dep_mode: str = "depositions",
        **identifier: Any,
    ) -> None:
        """Store the necessary attributes to do template-based PID prediction.

        Parameters
        ----------
        fill_per_pid : bool, default False
            If `True`, stores the scores associated with each PID candidate
        **identifier : dict, optional
            Particle template identifier configuration parameters
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode, truth_dep_mode)

        # Store additional parameter
        self.fill_per_pid = fill_per_pid

        # Initialize the underlying template-fitter class
        self.identifier = TemplateParticleIdentifier(**identifier)

    def process(self, data: Mapping[str, Any]) -> None:
        """Reconstruct template PID estimates for each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for k in self.fragment_keys + self.particle_keys:
            for obj in data[k]:
                # Only run this algorithm on tracks that have a CSDA table
                if not obj.shape == TRACK_SHP:
                    continue

                # Make sure the object coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates and depositions
                points = self.get_points(obj)
                values = self.get_depositions(obj)
                if len(points) == 0:
                    continue

                # Run the particle identifier
                pid, chi2_values = self.identifier(
                    points, values, obj.end_point, obj.start_point
                )

                # Store for this PID
                obj.chi2_pid = pid
                if self.fill_per_pid:
                    chi2_per_pid = np.full(
                        len(obj.chi2_per_pid), -1.0, dtype=obj.chi2_per_pid.dtype
                    )
                    for i, pid in enumerate(self.identifier.include_pids):
                        chi2_per_pid[pid] = chi2_values[i]

                    obj.chi2_per_pid = chi2_per_pid
