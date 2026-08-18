"""Module to evaluate diagnostic metrics on showers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spine.ana.base import AnaBase
from spine.cluster.direction import cluster_dedx, cluster_dedx_dir
from spine.constants import SHOWR_SHP

__all__ = ["ShowerStartDEdxAna"]


class ShowerStartDEdxAna(AnaBase):
    """Measure local dE/dx around the start of shower-like objects.

    This is a useful diagnostic tool to evaluate the calorimetric separability
    of different EM shower types (electron vs photon), which are expected to
    have different dE/dx patterns near their start point.

    Two local dE/dx definitions are supported:

    - ``default`` selects points in a sphere centered on the shower start and
      divides their total deposition by the largest radial displacement.
    - ``direction`` additionally restricts the points to the forward shower
      hemisphere and divides their total deposition by their longitudinal
      extent along the shower start direction.

    Supplying multiple radii or modes evaluates their Cartesian product for
    every shower. Each result is written to the CSV file associated with its
    reconstructed or truth object collection.
    """

    # Name of the analysis script (as specified in the configuration)
    name = "shower_start_dedx"

    # Shared cluster kernels exposed by this diagnostic
    _modes = ("default", "direction")

    # Keep public configuration options explicit for help() and generated docs
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        radius: float | Sequence[float],
        *,
        mode: str | Sequence[str] = "default",
        anchor: bool = False,
        obj_type: str | Sequence[str] = "particle",
        run_mode: str = "both",
        truth_point_mode: str = "points",
        truth_dep_mode: str = "depositions",
        **kwargs: Any,
    ) -> None:
        """Initialize the analysis script.

        Parameters
        ----------
        radius : float or Sequence[float]
            Positive neighborhood radius or radii within which to evaluate
            dE/dx, in centimeters
        mode : str or Sequence[str], default 'default'
            Local dE/dx definition or definitions. Must be ``default`` or
            ``direction``
        anchor : bool, default False
            If `True`, move the nominal start to the closest object point
            before evaluating dE/dx
        obj_type : str or Sequence[str], default 'particle'
            Shower-bearing object type or types to analyze. Fragments and
            particles are supported.
        run_mode : str, default 'both'
            Whether to analyze reconstructed, truth, or both object collections
        truth_point_mode : str, default 'points'
            Point representation to use for truth objects
        truth_dep_mode : str, default 'depositions'
            Deposition representation to use for truth objects
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`

        Raises
        ------
        ValueError
            If interactions are requested, no radius or mode is provided, or
            any radius or mode is invalid
        """
        # Initialize the parent class
        super().__init__(
            obj_type=obj_type,
            run_mode=run_mode,
            truth_point_mode=truth_point_mode,
            truth_dep_mode=truth_dep_mode,
            **kwargs,
        )

        # Interactions do not have an intrinsic shower start or shape.
        assert self.obj_type is not None
        if "interaction" in self.obj_type:
            raise ValueError("Shower start dE/dx does not support interactions.")

        # Normalize scan parameters once so processing is a simple cross-product.
        self.radii = self._normalize_radii(radius)
        self.modes = self._normalize_modes(mode)
        self.anchor = anchor

        # Keep reconstructed and truth object families in separate CSV files.
        for key in self.obj_keys:
            self.initialize_writer(key)

    @staticmethod
    def _normalize_radii(radius: float | Sequence[float]) -> tuple[float, ...]:
        """Validate and normalize one or more neighborhood radii.

        Parameters
        ----------
        radius : float or Sequence[float]
            Neighborhood radius or radii to normalize

        Returns
        -------
        tuple[float, ...]
            Positive finite radii represented as floats

        Raises
        ------
        ValueError
            If no radius is provided or a radius is non-finite or non-positive
        """
        # Treat a scalar radius as a one-element scan
        values = radius if isinstance(radius, Sequence) else (radius,)
        radii = tuple(float(value) for value in values)

        # Reject empty and non-physical radius specifications
        if not radii:
            raise ValueError("At least one dE/dx radius must be provided.")
        if any(not np.isfinite(value) or value <= 0.0 for value in radii):
            raise ValueError("Each dE/dx radius must be finite and positive.")

        return radii

    @classmethod
    def _normalize_modes(cls, mode: str | Sequence[str]) -> tuple[str, ...]:
        """Validate and normalize one or more dE/dx computation modes.

        Parameters
        ----------
        mode : str or Sequence[str]
            Computation mode or modes to normalize

        Returns
        -------
        tuple[str, ...]
            Validated computation modes

        Raises
        ------
        ValueError
            If no mode is provided or a mode is not supported
        """
        # Strings represent one mode rather than a sequence of characters
        modes = (mode,) if isinstance(mode, str) else tuple(mode)
        if not modes:
            raise ValueError("At least one dE/dx mode must be provided.")

        # Report the first invalid entry with the complete set of choices
        invalid = [value for value in modes if value not in cls._modes]
        if invalid:
            raise ValueError(
                f"dE/dx computation mode not recognized: {invalid[0]}. "
                f"Must be one of {cls._modes}."
            )

        return modes

    def process(self, data: Mapping[str, Any]) -> None:
        """Evaluate shower start dE/dx for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products for one entry

        Raises
        ------
        ValueError
            If a shower has different numbers of points and depositions
        """
        # Loop over each requested reconstructed or truth object collection.
        for key in self.obj_keys:
            # Only shower-like objects have a meaningful shower-start dE/dx
            for obj in data[key]:
                if obj.shape != SHOWR_SHP:
                    continue

                # Fetch matching spatial and calorimetric representations
                points = np.asarray(self.get_points(obj))
                depositions = np.asarray(self.get_depositions(obj))
                if len(points) != len(depositions):
                    raise ValueError(
                        "Shower points and depositions must have matching lengths."
                    )

                # Record every requested kernel/radius combination in a stable
                # schema, including undefined values for unusable showers.
                for mode in self.modes:
                    for radius in self.radii:
                        dedx = self.local_dedx(obj, points, depositions, radius, mode)
                        self.append(
                            key,
                            object_id=obj.id,
                            shape=obj.shape,
                            pid=getattr(obj, "pid", -1),
                            is_primary=obj.is_primary,
                            mode=mode,
                            radius=radius,
                            anchor=self.anchor,
                            dedx=dedx,
                        )

    def local_dedx(
        self,
        obj: Any,
        points: np.ndarray,
        depositions: np.ndarray,
        radius: float,
        mode: str,
    ) -> float:
        """Evaluate one local dE/dx configuration for a shower object.

        Parameters
        ----------
        obj : FragmentBase or ParticleBase
            Shower object which provides the start point and direction
        points : np.ndarray
            (N, 3) Point coordinates associated with the shower
        depositions : np.ndarray
            (N) Depositions associated with the shower points
        radius : float
            Neighborhood radius within which to evaluate dE/dx, in centimeters
        mode : str
            Local dE/dx definition, one of ``default`` or ``direction``

        Returns
        -------
        float
            Local dE/dx value. Returns ``np.nan`` when the shower geometry is
            insufficient to define the requested measurement.

        Raises
        ------
        ValueError
            If the computation mode is not supported
        """
        # A local extent cannot be formed from fewer than two valid points
        start = np.asarray(obj.start_point)
        if len(points) < 2 or start.shape != (3,) or not np.all(np.isfinite(start)):
            return np.nan

        # Use the direction-independent spherical estimator when requested
        if mode == "default":
            return float(
                cluster_dedx(
                    points,
                    depositions,
                    start,
                    max_dist=radius,
                    anchor=self.anchor,
                )
            )
        if mode != "direction":
            raise ValueError(f"Unsupported dE/dx mode: {mode}.")

        # Normalize the stored direction before projecting the shower points
        direction = np.asarray(obj.start_dir)
        norm = np.linalg.norm(direction)
        if direction.shape != (3,) or not np.isfinite(norm) or norm <= 0.0:
            return np.nan

        # The shared kernel also reports energy, length, spread, and point count
        result = cluster_dedx_dir(
            points,
            depositions,
            start,
            direction / norm,
            max_dist=radius,
            anchor=self.anchor,
        )
        return float(result[0])
