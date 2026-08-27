"""Module that supports barycenter-based flash matching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BarycenterMatchResult:
    """Transient quality information for one interaction--flash match.

    The flash-matching processor persists :attr:`score` in the matched
    interaction. The remaining quantities are kept here to make the matching
    decision inspectable without expanding the interaction data structure.

    Attributes
    ----------
    score : float
        Match quality, defined as the reciprocal of :attr:`chi2` after applying
        the configured numerical floor. Larger values represent better matches
    chi2 : float
        Sum of the configured spatial and PCA-angle chi-squared terms
    distance : float
        Euclidean distance between the charge and flash barycenters in the
        configured matching dimensions, in cm
    angle : float
        Acute angle between the charge and optical principal YZ axes, in
        degrees. This is ``NaN`` when the angle term is disabled
    charge_center : np.ndarray
        (3) Deposition-weighted charge barycenter in detector coordinates, in cm
    charge_width : np.ndarray
        (3) Deposition-weighted RMS charge width along each detector axis, in cm
    """

    score: float
    chi2: float
    distance: float
    angle: float
    charge_center: np.ndarray
    charge_width: np.ndarray


class BarycenterFlashMatcher:
    """Match interactions and flashes using their spatial barycenters.

    The matcher first applies the configured interaction and flash selections.
    It then compares each surviving pair using a spatial chi-squared, with an
    optional term for the angle between their principal YZ axes. Pairs above
    ``max_chi2`` are rejected when that cut is configured. The reporting mode
    then retains all surviving pairs, the best interaction per flash or the
    best flash per interaction.
    """

    # List of valid candidate reporting policies
    _report_modes = ("all", "best_per_flash", "best_per_interaction")

    def __init__(
        self,
        report_mode: str = "all",
        dimensions: Sequence[int] = (1, 2),
        charge_weighted: bool = True,
        time_window: Sequence[float] | None = None,
        first_flash_only: bool = False,
        min_inter_size: int | None = None,
        min_flash_pe: float | None = None,
        candidate_distance: float | None = None,
        position_errors: float | Sequence[float] = 1.0,
        angle_error: float | None = None,
        max_chi2: float | None = None,
        chi2_floor: float = 1.0e-6,
        optical: Any | None = None,
    ) -> None:
        r"""Initialize the barycenter flash matcher.

        Parameters
        ----------
        report_mode : str, default 'all'
            Policy used to report pairs which pass the candidate selections:

            - ``'all'`` retains every surviving interaction--flash pair
            - ``'best_per_flash'`` retains the lowest-chi-squared interaction
              for each flash
            - ``'best_per_interaction'`` retains the lowest-chi-squared flash
              for each interaction
        dimensions : Sequence[int], default (1, 2)
            Detector-coordinate dimensions included in the spatial distance
            and chi-squared. By convention, 0, 1 and 2 correspond to X, Y and Z
        charge_weighted : bool, default True
            Use calibrated interaction depositions to weight the charge center,
            width and PCA. If ``False``, give all valid points equal weight
        time_window : Sequence[float], optional
            Minimum and maximum optical flash times to consider, in microseconds
        first_flash_only : bool, default False
            Only attempt to match the first flash surviving the flash selections
        min_inter_size : int, optional
            Minimum number of voxels required in an interaction
        min_flash_pe : float, optional
            Minimum total number of PE required in a flash
        candidate_distance : float, optional
            Maximum Euclidean barycenter distance in the configured dimensions,
            in cm. Required in ``all`` mode and optional in either best mode
        position_errors : float or Sequence[float], default 1.0
            Position uncertainties used in the chi-squared, in cm. A scalar is
            shared by all matched dimensions. A sequence may provide one value
            per matched dimension or one value for each of X, Y and Z
        angle_error : float, optional
            PCA-angle uncertainty used in the chi-squared, in degrees. If
            omitted, the angle term is disabled and optical geometry is unused
        max_chi2 : float, optional
            Maximum chi-squared for an interaction--flash pair to be accepted.
            If omitted, no chi-squared acceptance cut is applied
        chi2_floor : float, default 1.e-6
            Strictly positive numerical floor applied when converting
            chi-squared to score. This limits the maximum score to the
            reciprocal of this value without changing the reported chi-squared
        optical : OptDetector, optional
            Optical detector geometry used to map per-channel PE to detector
            positions. Required when ``angle_error`` is set

        Notes
        -----
        For a pair with coordinate residuals :math:`\Delta_i` and optional PCA
        angle :math:`\theta`, the quality is

        .. math::

            \chi^2 = \sum_i \left(\frac{\Delta_i}{\sigma_i}\right)^2
            + \left(\frac{\theta}{\sigma_\theta}\right)^2,
            \qquad S = \frac{1}{\max(\chi^2, \chi^2_{\mathrm{floor}})}.

        The final angle term is omitted when ``angle_error`` is ``None``.
        """
        # Check the reporting-mode requirements
        if report_mode not in self._report_modes:
            raise ValueError(
                "Barycenter flash reporting mode not recognized: "
                f"{report_mode}. Must be one of {self._report_modes}."
            )

        if report_mode == "all" and candidate_distance is None:
            raise ValueError(
                "When using the `all` reporting mode, must specify the "
                "`candidate_distance` parameter."
            )

        if candidate_distance is not None and (
            not np.isfinite(candidate_distance) or candidate_distance < 0.0
        ):
            raise ValueError("`candidate_distance` must be finite and non-negative.")

        # Validate and normalize the dimensions involved in spatial matching
        dims = np.asarray(dimensions, dtype=np.int64)
        if (
            dims.ndim != 1
            or len(dims) == 0
            or len(np.unique(dims)) != len(dims)
            or np.any((dims < 0) | (dims > 2))
        ):
            raise ValueError(
                "`dimensions` must contain unique spatial axes selected from 0, 1, 2."
            )

        # Normalize the position errors to one value per matched dimension
        errors = np.asarray(position_errors, dtype=np.float64)
        if errors.ndim == 0:
            errors = np.full(len(dims), errors.item(), dtype=np.float64)
        elif errors.shape == (3,):
            errors = errors[dims]
        elif errors.shape != (len(dims),):
            raise ValueError(
                "`position_errors` must be a scalar, have one value per "
                "matched dimension, or have three spatial values."
            )
        if np.any(~np.isfinite(errors)) or np.any(errors <= 0.0):
            raise ValueError("`position_errors` must be finite and strictly positive.")

        # The quality cut is optional, but must define a valid chi-squared range
        if max_chi2 is not None and (not np.isfinite(max_chi2) or max_chi2 < 0.0):
            raise ValueError("`max_chi2` must be finite and non-negative.")

        # Keep the reciprocal score finite for exact or numerically tiny matches
        if not np.isfinite(chi2_floor) or chi2_floor <= 0.0:
            raise ValueError("`chi2_floor` must be finite and strictly positive.")

        # The angular term requires both a valid uncertainty and optical geometry
        if angle_error is not None:
            if not np.isfinite(angle_error) or angle_error <= 0.0:
                raise ValueError("`angle_error` must be finite and strictly positive.")
            if optical is None:
                raise ValueError(
                    "Optical detector geometry is required when `angle_error` is set."
                )

        # Store the normalized flash-matching parameters
        self.report_mode = report_mode
        self.dims = dims
        self.charge_weighted = charge_weighted
        self.time_window = time_window
        self.first_flash_only = first_flash_only
        self.min_inter_size = min_inter_size
        self.min_flash_pe = min_flash_pe
        self.candidate_distance = candidate_distance
        self.position_errors = errors
        self.angle_error = angle_error
        self.max_chi2 = max_chi2
        self.chi2_floor = chi2_floor
        self.optical = optical

    def get_matches(
        self, interactions: Sequence[Any], flashes: Sequence[Any]
    ) -> list[tuple[Any, Any, BarycenterMatchResult]]:
        """Build interaction--flash pairs with compatible barycenters.

        Invalid or non-positive deposition weights are excluded from the charge
        observables. A candidate is also excluded when its distance or
        chi-squared is non-finite, or when an enabled PCA term cannot be
        constructed from either footprint. The geometric distance cut is
        applied before the optional full chi-squared cut.

        Parameters
        ----------
        interactions : Sequence[RecoInteraction | TruthInteraction]
            Interactions to consider for matching
        flashes : Sequence[Flash]
            Optical flashes to consider for matching

        Returns
        -------
        list[tuple[Interaction, Flash, BarycenterMatchResult]]
            Accepted interaction, flash and match-quality triplets
        """
        # Convert the input sequences to lists for indexing and length checks
        interactions = list(interactions)
        flashes = list(flashes)

        # Restrict the flashes to those that fit the selection criteria
        if self.time_window is not None:
            t1, t2 = self.time_window
            flashes = [f for f in flashes if (f.time > t1 and f.time < t2)]

        if self.min_flash_pe is not None:
            flashes = [f for f in flashes if f.total_pe > self.min_flash_pe]

        if len(flashes) == 0:
            return []

        # If requested, only match the first flash that survived selection
        if self.first_flash_only:
            flashes = [flashes[0]]

        # Restrict interactions to those that fit the size selection
        if self.min_inter_size is not None:
            interactions = [
                inter
                for inter in interactions
                if len(inter.points) > self.min_inter_size
            ]

        if len(interactions) == 0:
            return []

        # Build charge observables once per interaction. Interactions with no
        # usable points or charge cannot form a barycenter match
        valid_interactions = []
        charge_centers = []
        charge_widths = []
        charge_axes = []
        for inter in interactions:
            observables = self._charge_observables(inter)
            if observables is None:
                continue
            center, width, axis = observables
            valid_interactions.append(inter)
            charge_centers.append(center)
            charge_widths.append(width)
            charge_axes.append(axis)

        if len(valid_interactions) == 0:
            return []

        interactions = valid_interactions
        int_centroids = np.asarray(charge_centers)[:, self.dims]
        op_centroids = np.asarray([f.center[self.dims] for f in flashes])

        # Compute the flash-to-interaction distance matrix
        dist_mat = np.linalg.norm(
            op_centroids[:, None, :] - int_centroids[None, :, :], axis=2
        )

        # Compute all candidate qualities. The distance threshold remains a
        # geometric selection; the returned score is consistently 1 / chi2
        results: dict[tuple[int, int], BarycenterMatchResult] = {}
        for i, flash in enumerate(flashes):
            optical_axis = None
            if self.angle_error is not None:
                optical_axis = self._optical_axis(flash)

            for j, _ in enumerate(interactions):
                distance = float(dist_mat[i, j])
                if not np.isfinite(distance) or (
                    self.candidate_distance is not None
                    and distance > self.candidate_distance
                ):
                    continue

                # Form the spatial chi-squared in the requested dimensions
                delta = op_centroids[i] - int_centroids[j]
                chi2 = float(np.sum(np.square(delta / self.position_errors)))

                # Add the acute PCA-angle contribution when it is enabled
                angle = np.nan
                if self.angle_error is not None:
                    charge_axis = charge_axes[j]
                    if charge_axis is None or optical_axis is None:
                        continue
                    cosine = np.clip(abs(np.dot(charge_axis, optical_axis)), 0.0, 1.0)
                    angle = float(np.degrees(np.arccos(cosine)))
                    chi2 += (angle / self.angle_error) ** 2

                if not np.isfinite(chi2):
                    continue

                # Optionally reject candidates which pass the geometric cut but
                # are incompatible under the full spatial and angular metric
                if self.max_chi2 is not None and chi2 > self.max_chi2:
                    continue

                # Match OpT0Finder's chi-squared-mode convention while keeping
                # exact or numerically tiny matches finite
                score = 1.0 / max(chi2, self.chi2_floor)
                results[(i, j)] = BarycenterMatchResult(
                    score=score,
                    chi2=chi2,
                    distance=distance,
                    angle=angle,
                    charge_center=charge_centers[j],
                    charge_width=charge_widths[j],
                )

        # Dispatch the accepted candidates according to the reporting mode
        matches = []
        if self.report_mode == "best_per_flash":
            # For each flash, select the lowest-chi-squared interaction
            for i, f in enumerate(flashes):
                candidates = [
                    (j, results[(i, j)])
                    for j in range(len(interactions))
                    if (i, j) in results
                ]
                if len(candidates) == 0:
                    continue
                best_match, result = min(candidates, key=lambda item: item[1].chi2)
                matches.append((interactions[best_match], f, result))

        elif self.report_mode == "best_per_interaction":
            # For each interaction, independently select its best flash. This
            # allows multiple interactions to report the same flash
            for j, interaction in enumerate(interactions):
                candidates = [
                    (i, results[(i, j)])
                    for i in range(len(flashes))
                    if (i, j) in results
                ]
                if len(candidates) == 0:
                    continue
                best_match, result = min(candidates, key=lambda item: item[1].chi2)
                matches.append((interaction, flashes[best_match], result))

        elif self.report_mode == "all":
            matches = [
                (interactions[j], flashes[i], result)
                for (i, j), result in results.items()
            ]

        return matches

    def _charge_observables(
        self, interaction: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Compute the charge center, RMS width and principal YZ axis.

        Parameters
        ----------
        interaction : Interaction
            Interaction containing an ``(N, 3)`` point array and one calibrated
            deposition value per point

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray], optional
            Three-dimensional charge center, three-dimensional RMS width and
            two-dimensional principal YZ axis. The axis is ``None`` for a
            degenerate YZ footprint. The full result is ``None`` when no valid
            points remain after selection
        """
        points = np.asarray(interaction.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Interaction points must have shape (N, 3).")

        # Keep only finite points and, when requested, positive finite weights
        valid = np.all(np.isfinite(points), axis=1)
        if self.charge_weighted:
            weights = np.asarray(interaction.depositions, dtype=np.float64)
            if weights.shape != (len(points),):
                raise ValueError(
                    "Interaction depositions must have one value per point."
                )
            valid &= np.isfinite(weights) & (weights > 0.0)
            weights = weights[valid]
        else:
            weights = np.ones(np.count_nonzero(valid), dtype=np.float64)

        points = points[valid]
        if len(points) == 0:
            return None

        # Compute the weighted first and second spatial moments
        total = np.sum(weights)
        center = np.sum(weights[:, None] * points, axis=0) / total
        variance = np.sum(weights[:, None] * np.square(points - center), axis=0) / total
        width = np.sqrt(np.maximum(variance, 0.0))
        axis = self._principal_axis(points[:, (1, 2)], weights)
        return center, width, axis

    def _optical_axis(self, flash: Any) -> np.ndarray | None:
        """Compute the PE-squared weighted principal optical YZ axis.

        Parameters
        ----------
        flash : Flash
            Optical flash containing a volume ID and one PE value per optical
            readout channel

        Returns
        -------
        np.ndarray, optional
            (2) Unit vector along the principal YZ axis, or ``None`` when the
            optical footprint is degenerate
        """
        # Select either the global detector index or this flash's local volume
        assert self.optical is not None, "Optical geometry is required for PCA."
        volume_id = int(flash.volume_id)
        if self.optical.global_index:
            positions = np.asarray(self.optical.positions, dtype=np.float64)
            det_ids = self.optical.det_ids
        else:
            if volume_id < 0 or volume_id >= self.optical.num_volumes:
                raise ValueError(f"Invalid optical volume ID: {volume_id}.")
            volume = self.optical.volumes[volume_id]
            positions = np.asarray(volume.positions, dtype=np.float64)
            det_ids = volume.det_ids

        # Aggregate channels that share a physical optical detector
        pe_per_ch = np.asarray(flash.pe_per_ch, dtype=np.float64)
        if det_ids is None:
            if len(pe_per_ch) != len(positions):
                raise ValueError(
                    "Flash PE vector does not match the optical detector geometry."
                )
            pe_per_det = pe_per_ch
        else:
            det_ids = np.asarray(det_ids, dtype=np.int64)
            if len(pe_per_ch) != len(det_ids):
                raise ValueError(
                    "Flash PE vector does not match the optical channel mapping."
                )
            pe_per_det = np.bincount(
                det_ids, weights=pe_per_ch, minlength=len(positions)
            )[: len(positions)]

        # Match SBND's footprint construction: merge collocated detectors, then
        # square their aggregate PE before forming the optical PCA.
        yz, inverse = np.unique(positions[:, (1, 2)], axis=0, return_inverse=True)
        pe_per_pos = np.bincount(inverse, weights=pe_per_det, minlength=len(yz))
        valid = (
            np.all(np.isfinite(yz), axis=1)
            & np.isfinite(pe_per_pos)
            & (pe_per_pos > 0.0)
        )
        return self._principal_axis(yz[valid], np.square(pe_per_pos[valid]))

    @staticmethod
    def _principal_axis(points: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
        """Return the principal axis of a weighted two-dimensional footprint.

        Parameters
        ----------
        points : np.ndarray
            (N, 2) Two-dimensional point coordinates
        weights : np.ndarray
            (N) Non-negative point weights

        Returns
        -------
        np.ndarray, optional
            (2) Unit eigenvector associated with the largest covariance
            eigenvalue, or ``None`` when the footprint is degenerate
        """
        if len(points) < 2:
            return None

        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            return None

        # Build the weighted covariance matrix about the weighted center
        center = np.sum(weights[:, None] * points, axis=0) / total
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / total
        if not np.all(np.isfinite(covariance)) or np.allclose(covariance, 0.0):
            return None

        # The principal direction is the eigenvector of largest variance
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] <= 0.0 or np.isclose(values[-1], values[-2]):
            return None

        axis = vectors[:, -1]
        return axis / np.linalg.norm(axis)
