"""Diagnostic analysis of reconstructed space-point completeness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from spine.ana.base import AnaBase

__all__ = ["PointCompletenessAna"]


class PointCompletenessAna(AnaBase):
    """Measure space-point coverage relative to Geant4 energy depositions.

    For each truth object, this diagnostic compares a selected detector-level
    point representation against the corresponding Geant4 deposition points.
    Purity is the fraction of selected points close to a Geant4 point, while
    efficiency is the fraction of Geant4 points close to a selected point.
    """

    # Name of the analysis script (as specified in the configuration)
    name = "point_completeness"

    # Preserve the name proposed in the original contribution
    aliases = ("point_metrics",)

    def __init__(
        self,
        obj_type: str | Sequence[str] = "particle",
        truth_point_mode: str = "points",
        time_window: Sequence[float] | None = None,
        match_distance: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the point-completeness diagnostic.

        Parameters
        ----------
        obj_type : str or Sequence[str], default 'particle'
            Truth object type or types to evaluate
        truth_point_mode : str, default 'points'
            Detector-level truth point representation to compare to Geant4
        time_window : Sequence[float], optional
            Inclusive truth-object time window in nanoseconds
        match_distance : float, optional
            Maximum nearest-neighbor distance in centimeters. If omitted, use
            the diagonal length of one image voxel.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # This comparison is defined on truth-associated detector points
        super().__init__(
            obj_type=obj_type,
            run_mode="truth",
            truth_point_mode=truth_point_mode,
            **kwargs,
        )
        assert self.obj_type is not None
        self.object_types = tuple(self.obj_type)

        # Validate and store the optional time window
        normalized_time_window: tuple[float, float] | None = None
        if time_window is not None:
            if not isinstance(time_window, Sequence) or len(time_window) != 2:
                raise ValueError(
                    "Time window must be specified as an array of two values."
                )
            if time_window[0] > time_window[1]:
                raise ValueError(
                    "Time window lower bound must not exceed its upper bound."
                )
            normalized_time_window = (time_window[0], time_window[1])
        self.time_window = normalized_time_window

        # Validate and store the distance threshold
        if match_distance is not None:
            if not np.isfinite(match_distance) or match_distance <= 0.0:
                raise ValueError("Match distance must be finite and positive.")
        self.match_distance = match_distance

        # Geant4 points provide the reference; metadata provides the default
        # distance scale and ensures that coordinate units are explicit.
        self.update_keys({"points_g4": True, "meta": True})

        # Initialize one output file per requested truth object type
        for obj in self.object_types:
            self.initialize_writer(obj)

    def process(self, data: Mapping[str, Any]) -> None:
        """Evaluate point completeness for all requested truth objects."""
        match_distance = self.match_distance
        if match_distance is None:
            match_distance = float(np.linalg.norm(data["meta"].size))
            if not np.isfinite(match_distance) or match_distance <= 0.0:
                raise ValueError("Image metadata must define positive voxel sizes.")

        for obj_type in self.object_types:
            for obj in data[f"truth_{obj_type}s"]:
                if self.time_window is not None:
                    lower, upper = self.time_window
                    if not lower <= obj.time <= upper:
                        continue

                points = self.get_points(obj)
                points_g4 = obj.points_g4
                row = {
                    "id": obj.id,
                    "shape": getattr(obj, "shape", -1),
                    "num_points": len(points),
                    "num_points_g4": len(points_g4),
                    "purity": self.match_fraction(
                        points,
                        points_g4,
                        match_distance,
                    ),
                    "efficiency": self.match_fraction(
                        points_g4,
                        points,
                        match_distance,
                    ),
                }
                self.append(obj_type, **row)

    @staticmethod
    def match_fraction(
        source: np.ndarray,
        target: np.ndarray,
        distance: float,
    ) -> float:
        """Return the fraction of source points close to any target point.

        The fraction is undefined when there are no source points and is zero
        when source points exist but no target points are available.
        """
        if len(source) == 0:
            return np.nan
        if len(target) == 0:
            return 0.0

        distances, _ = cKDTree(target).query(source, k=1)
        return float(np.mean(distances <= distance))
