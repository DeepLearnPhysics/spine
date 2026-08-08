"""Calorimetric energy reconstruction module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numexpr as ne
import numpy as np

from spine.calib import CalibrationManager
from spine.constants import TRACK_SHP
from spine.post.base import PostBase

__all__ = ["CalorimetricEnergyProcessor", "CalibrationProcessor"]


class CalorimetricEnergyProcessor(PostBase):
    """Compute calorimetric energy by summing the charge depositions and
    scaling by the ADC to MeV conversion factor, if needed.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "calo_ke"

    # Alternative allowed names of the post-processor
    aliases = ("reconstruct_calo_energy",)

    def __init__(
        self,
        scaling: float | str = 1.0,
        shower_fudge: float | str = 1.0,
        obj_type: str | Sequence[str] | None = "particle",
        run_mode: str = "reco",
        truth_dep_mode: str = "depositions",
    ) -> None:
        """Stores the ADC to MeV conversion factor.

        Parameters
        ----------
        scaling : float or str, default 1.
            Global scaling factor for the depositions (can be an expression)
        shower_fudge : float or str, default 1.
            Shower energy fudge factor (accounts for missing cluster energy)
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_dep_mode=truth_dep_mode)

        # Store the conversion factor
        self.scaling = (
            float(ne.evaluate(scaling)) if isinstance(scaling, str) else scaling
        )

        # Store the shower fudge factor
        self.shower_fudge = (
            float(ne.evaluate(shower_fudge))
            if isinstance(shower_fudge, str)
            else shower_fudge
        )

    def process(self, data: Mapping[str, Any]) -> None:
        """Reconstruct the calorimetric KE for each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle types
        for k in self.obj_keys:
            # Loop over particles
            for part in data[k]:
                # Set up the scaling
                scaling = self.scaling
                if part.shape != TRACK_SHP:
                    scaling *= self.shower_fudge

                # Save the calorimetric energy
                depositions = self.get_depositions(part)
                part.calo_ke = scaling * np.sum(depositions)


class CalibrationProcessor(PostBase):
    """Apply calibrations to the reconstructed objects."""

    # Supported reconstructed deposition inputs. Calibration output always
    # populates ``depositions`` regardless of which source is selected.
    _depositions_sources = ("depositions_q", "depositions")

    # Name of the post-processor (as specified in the configuration)
    name = "calibration"

    # Alternative allowed names of the post-processor
    aliases = ("apply_calibrations",)

    # Set of data keys needed for this post-processor to operate
    _keys = (("run_info", False),)

    def __init__(
        self,
        do_tracking: bool = False,
        depositions_source: str = "depositions_q",
        obj_type: str | Sequence[str] | None = ("particle", "interaction"),
        run_mode: str = "reco",
        truth_point_mode: str = "points",
        **cfg: Any,
    ) -> None:
        """Initialize the calibration manager.

        Parameters
        ----------
        do_tracking : bool, default False
            Segment track to get a proper local dQ/dx estimate
        depositions_source : str, default "depositions_q"
            Reconstructed deposition attribute and tensor to calibrate. Use
            ``depositions_q`` to calibrate from preserved input charge, or
            ``depositions`` to continue from previously calibrated values.
            Results are always written to ``depositions``.
        **cfg : dict
            Calibration manager configuration
        """
        if depositions_source not in self._depositions_sources:
            raise ValueError(
                "`depositions_source` must be one of "
                f"{self._depositions_sources}. Got `{depositions_source}`."
            )

        # Figure out which truth deposition attribute to use
        truth_dep_mode = truth_point_mode.replace("points", "depositions") + "_q"

        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode, truth_dep_mode)

        # Initialize the calibrator
        self.calibrator = CalibrationManager(**cfg)
        self.do_tracking = do_tracking
        self.depositions_source = depositions_source

        # Add necessary keys
        keys = {}
        if run_mode != "truth":
            keys.update(
                {
                    "points": True,
                    "depositions": True,
                    self.depositions_source: True,
                    "sources": False,
                }
            )

        if run_mode != "reco":
            keys.update(
                {
                    self.truth_point_key: True,
                    self.truth_dep_key: True,
                    self.truth_source_key: False,
                }
            )

        self.update_keys(keys)

    def process(self, data: Mapping[str, Any]) -> None:
        """Apply calibrations to each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the run info
        run_id = None
        if "run_info" in data:
            run_info = data["run_info"]
            run_id = run_info.run

        # Loop over particle objects
        for k in self.particle_keys:
            sources = None
            is_truth = "truth" in k
            points_key = self.truth_point_key if is_truth else "points"
            source_key = self.truth_source_key if is_truth else "sources"
            input_dep_key = self.truth_dep_key if is_truth else self.depositions_source
            output_dep_key = self.truth_dep_key if is_truth else "depositions"
            unass_mask = np.ones(len(data[input_dep_key]), dtype=bool)
            for part in data[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(part)

                # Get point coordinates, sources and depositions
                points = self.get_points(part)
                if len(points) == 0:
                    continue

                if is_truth:
                    deps = self.get_depositions(part)
                else:
                    if not hasattr(part, self.depositions_source):
                        raise ValueError(
                            "Reconstructed particle is missing requested deposition "
                            f"source `{self.depositions_source}`."
                        )
                    deps = getattr(part, self.depositions_source)
                    if len(deps) != len(points):
                        raise ValueError(
                            f"Requested deposition source `{self.depositions_source}` "
                            f"has {len(deps)} values for {len(points)} points."
                        )
                if source_key in data:
                    sources = self.get_sources(part)

                # Apply calibration
                if not self.do_tracking or part.shape != TRACK_SHP:
                    cal_points, depositions = self.calibrator(
                        points, deps, sources, run_id
                    )
                else:
                    cal_points, depositions = self.calibrator(
                        points, deps, sources, run_id, track=True
                    )

                # Update the particle *and* the reference tensor
                if part.is_truth:
                    setattr(part, self.truth_dep_mode, depositions)
                else:
                    part.depositions = depositions

                if self.calibrator.update_points:
                    if not part.is_truth:
                        part.points = cal_points
                        self._update_reco_positions(part)
                    else:
                        setattr(part, self.truth_point_mode, cal_points)
                    data[points_key][part.index] = cal_points
                data[output_dep_key][part.index] = depositions
                unass_mask[part.index] = False

            # Apply calibration corrections to unassociated depositions
            unass_index = np.where(unass_mask)[0]
            points = data[points_key][unass_index]
            depositions = data[input_dep_key][unass_index]
            sources = None
            if source_key in data:
                sources = data[source_key][unass_index]

            cal_points, depositions = self.calibrator(
                points, depositions, sources, run_id
            )
            if self.calibrator.update_points:
                data[points_key][unass_index] = cal_points
            data[output_dep_key][unass_index] = depositions

        # If requested, updated the depositions attribute of interactions
        for k in self.interaction_keys:
            is_truth = "truth" in k
            points_key = self.truth_point_key if is_truth else "points"
            output_dep_key = self.truth_dep_key if is_truth else "depositions"
            for inter in data[k]:
                # Update depositions for the interaction
                depositions = data[output_dep_key][inter.index]
                if inter.is_truth:
                    setattr(inter, self.truth_dep_mode, depositions)
                else:
                    inter.depositions = depositions

                if self.calibrator.update_points:
                    points = data[points_key][inter.index]
                    if not inter.is_truth:
                        inter.points = points
                        self._update_reco_positions(inter)
                    else:
                        setattr(inter, self.truth_point_mode, points)

    def _update_reco_positions(self, obj: Any) -> None:
        """Apply field corrections to all auxiliary reconstructed positions."""
        attrs, arrays = [], []
        for attr in getattr(obj, "_pos_attrs", ()):
            # The primary point cloud is corrected by the main calibration
            # pass. All other declared positions must follow the same map.
            if attr == "points" or not hasattr(obj, attr):
                continue

            values = np.asarray(getattr(obj, attr))
            values_2d = values.reshape(-1, 3)
            valid = np.all(np.isfinite(values_2d), axis=1)
            if not np.any(valid):
                continue

            attrs.append((attr, values.shape, valid))
            arrays.append(values_2d[valid])

        if not arrays:
            return

        positions = np.concatenate(arrays)
        corrected = self.calibrator.process_points(positions)
        offset = 0
        for attr, shape, valid in attrs:
            count = np.count_nonzero(valid)
            values = np.asarray(getattr(obj, attr)).reshape(-1, 3).copy()
            values[valid] = corrected[offset : offset + count]
            setattr(obj, attr, values.reshape(shape))
            offset += count
