"""Module with functions/classes used to identify particle species."""

import numpy as np
from scipy.optimize import minimize

from .energy_loss import (
    bethe_bloch_lar,
    bethe_bloch_mpv_lar,
    csda_ke_lar,
    csda_table_spline,
)
from .globals import KAON_PID, MUON_PID, PID_MASSES, PION_PID, PROT_PID, TRACK_SHP
from .tracking import get_track_segment_dedxs


class TemplateParticleIdentifier:
    """Class which uses dE/dx templates to do particle identification.

    The basics of template-based PID are as follows:
    - Chunk a track into chunks
    - Identify the dE/dx of the tracks in each chunk
    - Try to match the dE/dx profile to a known particle template
    """

    def __init__(
        self,
        use_table=True,
        use_mpv=False,
        include_pids=(MUON_PID, PION_PID, PROT_PID, KAON_PID),
        optimize_rr=False,
        max_rr=100,
        optimize_orient=True,
        tracking_mode="step_next",
        **tracker,
    ):
        """Initialize the template-based particle identifier.

        Parameters
        ----------
        use_table : bool, default True
            If `True`, use tabulated values of dE/dx vs residual range. The
            dE/dx is evaluated using the theoretical value otherwise.
        use_mpv : bool, default False
            If `True`, use the most-probable dE/dx value instead of the mean
        include_pids : list, default [2, 3, 4, 5]
            Particle species to consider as a viable PID candidate
        optimize_rr : bool, default False
            If `True`, vary RR to minimize the chi2 agreement. If `False`, the
            track is assumed to range out (no hard scattering, no exiting)
        max_rr : float
            Maximum allowed residual range offset
        optimize_orient : bool, default True
            If `True`, compute the chi2 w.r.t. to the start and end points of
            the track, in case the travel direction was mireconstructed
        tracking_mode : str, default 'step_next'
            Method used to chunk the track and compute the dE/dx vs RR (one of
            'step', 'step_next' or 'bin_pca')
        **tracker : dict, optional
            Arguments to pass to the tracking algorithm (track chunking)
        """
        # Store parameters
        self.use_table = use_table
        self.use_mpv = use_mpv
        self.optimize_rr = optimize_rr
        self.max_rr = max_rr
        self.optimize_orient = optimize_orient

        # If needed, initialize the spline fits to the dE/dx templates once
        self.include_pids = include_pids
        self.tables = {
            pid: csda_table_spline(pid, value="dE/dx") for pid in include_pids
        }

        # Store the tracking parameters
        # TODO: make this is a class which owns the parameters
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = tracker

        # Basic logic check
        assert not use_mpv or not use_table, (
            "MPV dE/dx values are not tabulated. Must use the theoretical "
            "approach (`use_table=False`) when `use_mpv=True`."
        )

    def __call__(self, points, depositions, end_point, start_point=None):
        """Evaluate the agreement between each PID template and the particle
        track of interest.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Track point coordinates
        depositions : np.ndarray
            (N) Deposition values in MeV
        end_point : np.ndarray
            (3) Coordinates of the end point of the track
        start_point : np.ndarray, optional
            (3) Coordinates of the start point of the track

        Returns
        -------
        int
            Best-fit particle ID
        np.ndarray
            (P) One chi2 score per particle species candidate
        """
        # Check which end of the tracks are to be considered
        end_points = [end_point]
        if self.optimize_orient:
            assert (
                start_point is not None
            ), "Must pass start_point when test both track orientations."
            end_points.append(start_point)

        # Loop over the track orientations considered
        chi2_per_pid = np.full(len(self.include_pids), -1.0)
        for epoint in end_points:
            # Segment the track, measure dE/dx and residual range values
            seg_dedxs, _, seg_rrs, _, _, _ = get_track_segment_dedxs(
                points,
                depositions,
                epoint,
                method=self.tracking_mode,
                **self.tracking_kwargs,
            )

            # Only keep segments for which the dE/dx could be measured
            mask = np.where(seg_dedxs > 0.0)[0]
            if len(mask) < 1:
                continue
            seg_dedxs, seg_rrs = seg_dedxs[mask], seg_rrs[mask]

            # Loop over PID candidates
            for i, pid in enumerate(self.include_pids):
                # Dispatch depending on the minimization method
                if not self.optimize_rr:
                    chi2 = self.chi2(seg_dedxs, seg_rrs, pid)
                else:
                    _, chi2 = self.minimize_rr(seg_dedxs, seg_rrs, pid)

                # Update chi2 for this PID
                if chi2_per_pid[i] < 0.0 or chi2 < chi2_per_pid[i]:
                    chi2_per_pid[i] = chi2

        # Return
        if np.all(chi2_per_pid == -1.0):
            return -1, chi2_per_pid
        else:
            pid = self.include_pids[np.argmin(chi2_per_pid)]
            return pid, chi2_per_pid

    def minimize_rr(self, dedxs, rrs, pid):
        """Find the residual range which minimises the chi2 fit between the
        observed dE/dx values and the spline fit to the template.

        Parameters
        ----------
        dedxs : np.ndarray
            (S) Measured values of dedxs at a set of unknown residual ranges
        rrs : np.ndarray
            (S) Residual ranges assuming the particle came to a complete stop
        pid : int
            Particle species enumerator

        Returns
        -------
        float
            Value of the residual range at the end of the track
        float
            Value of the chi2 for the optimal residual range
        """
        chi2 = lambda x: self.chi2(dedxs, x + rrs, pid)
        res = minimize(chi2, 0.0, bounds=((0.0, self.max_rr),))
        return res.x[0], res.fun

    def chi2(self, dedxs, rrs, pid):
        """Computes a chi2 score given a set of observed/expected dE/dx values.

        Parameters
        ----------
        dedxs : np.ndarray
            (S) Measured values of dedxs at a set of known residual ranges
        rrs : np.ndarray
            (S) Residual range values (one per track segment)
        pid : int
            Particle species enumerator

        Returns
        -------
        float
            Chi2 agreement value
        """
        # TODO: should get a better measure of uncertainty
        exp_dedxs = self.expected_dedxs(rrs, pid)
        diff = dedxs - exp_dedxs
        return np.sum(diff**2 / dedxs) / len(dedxs)

    def expected_dedxs(self, rrs, pid):
        """Computes the expected dE/dx values given a set of residual ranges.

        Parameters
        ----------
        rrs : np.ndarray
            (S) Residual range values (one per track segment)
        pid : int
            Particle species enumerator

        Returns
        -------
        np.ndarray
            Expected dE/dxs values from a table or the theory
        """
        if self.use_table:
            exp_dedxs = self.tables[pid](rrs)

        else:
            mass = PID_MASSES[pid]
            exp_dedxs = np.empty(len(rrs), dtype=seg_dedxs.dtype)
            for i in range(len(rrs)):
                T = csda_ke_lar(rrs[i], mass)
                if not use_mpv:
                    dedx = -bethe_bloch_lar(T, mass)
                else:
                    # TODO: get material thickness from tracker parameters
                    dedx = -bethe_bloch_mpv_lar(T, mass, 1)
                exp_dedxs[i] = dedx

        return exp_dedxs
