"""Applies electron recombination corrections."""

import numpy as np

from spine.utils.globals import LAR_DENSITY, LAR_WION
from spine.utils.tracking import get_track_segment_dedxs

__all__ = ['RecombinationCalibrator']


class RecombinationCalibrator:
    """Applies a recombination correction factor to account for some of the
    ionization electrons recombining with the Argon ions, which is an effect
    that depends on the local rate of energy deposition and the angle of
    the deposition trail (track) w.r.t. to the drift field.

    Notes
    -----
    Must call the gain calibrator upstream, which converts the number of ADCs
    to a number of observed ionization electrons.
    """
    name = 'recombination'

    def __init__(self, efield, drift_dir, model='mbox', birks_a=0.800,
                 birks_k=0.0486, mbox_alpha=0.906, mbox_beta=0.203,
                 mbox_ell_r=1.25, mip_dedx=2.2, tracking_mode='bin_pca',
                 **kwargs):
        """Initialize the recombination model and its constants.

        Parameters
        ----------
        efield : float
            Electric field in kV/cm
        drift_dir : np.ndarray
            (3) three-vector indicating the direction of the drift field
        model : str, default 'mbox'
            Recombination model name (one of 'birks', 'mbox' or 'mbox_ell')
        birks_a : float, default 0.800 (ICARUS CNGS fit)
            Birks model A parameter
        birks_k : float, default 0.0486 (ICARUS CNGS fit)
            Birks model k parameter in (kV/cm)(gm/cm^2)/MeV
        mbox_alpha : float, default 0.906 (ICARUS fit)
            Modified box model alpha parameter
        mbox_beta : float, default 0.203 (ICARUS fit)
            Modified box model beta parameter in (kV/cm)(g/cm^2)/MeV
        mbox_ell_r : float, default 1.25 (ICARUS fit)
            Modified box model ellipsoid correction R parameter
        mip_dedx : float, default 2.2 (must be changed to 2.105168)
            Mean dE/dx value of a MIP in LAr. Used to apply a flat recombination
            correction if the local dE/dx is not evaluated through tracking.
        track_mode : float, default 'bin_pca'
            If tracking is done to produce local dQ/dx values along tracks,
            defines the track chunking method to be used.
        **kwargs : dict, optional
            Additional arguments to pass to the tracking algorithm
        """
        # Store the drift direction
        self.drift_dir = drift_dir

        # Initialize the model parameters
        self.use_angles = False
        if model == 'birks':
            self.model = 'birks'
            self.a = birks_a
            self.k = birks_k/efield/LAR_DENSITY # cm/MeV

        elif model in ['mbox', 'mbox_ell']:
            self.model = 'mbox'
            self.alpha = mbox_alpha
            self.beta = mbox_beta/efield/LAR_DENSITY # cm/MeV
            self.r = None
            if model == 'mbox_ell':
                self.use_angles = True
                self.r = mbox_ell_r

        else:
            raise ValueError(
                    f"Recombination model not recognized: {model}. "
                     "Must be one of 'birks', 'mbox' or 'mbox_ell'")

        # Evaluate the MIP recombination factor, store it
        self.mip_recomb = self.recombination_factor(mip_dedx)

        # Store the tracking parameters
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = kwargs

    def birks(self, dedx):
        """Birks equation to calculate electron quenching (higher local energy
        deposition are prone to more electron-ion recombination).

        Parameters
        ----------
        dedx : Union[float, np.ndarray]
            Value or array of values of dE/dx in MeV/cm

        Returns
        -------
        Union[float, np.ndarray]
           Quenching factors in electrons/MeV
        """
        return self.a / (1. + self.k * dedx)

    def inv_birks(self, dqdx):
        """Inverse Birks equation to undo electron quenching (higher local
        energy deposition are prone to more electron-ion recombination).

        Parameters
        ----------
        dqdx : Union[float, np.ndarray]
            Value or array of values of dQ/dx in electrons/cm

        Returns
        -------
        Union[float, np.ndarray]
            Inverse quenching factors in MeV/electrons
        """
        return 1. / (self.a/LAR_WION - self.k * dqdx)

    def mbox(self, dedx, cosphi=None):
        """Modified box model equation to calculate electron quenching (higher
        local energy deposition are prone to more electron-ion recombination).

        Parameters
        ----------
        dedx : Union[float, np.ndarray]
            Value or array of values of dE/dx in MeV/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Quenching factors in electrons/MeV
        """
        beta = self.beta
        if cosphi is not None:
            beta /= np.sqrt(1 - (1 - 1./self.r**2)*cosphi**2)

        return np.log(self.alpha + beta * dedx) / beta / dedx

    def inv_mbox(self, dqdx, cosphi=None):
        """Inverse modified box model equation to undo electron quenching
        (higher local energy deposition are prone to more electron-ion
        recombination).

        Parameters
        ----------
        dqdx : Union[float, np.ndarray]
            Value or array of values of dQ/dx in electrons/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Inverse quenching factors in MeV/electrons
        """
        beta = self.beta
        if cosphi is not None:
            beta /= np.sqrt(1 - (1 - 1./self.r**2)*cosphi**2)

        return (np.exp(beta * LAR_WION * dqdx) - self.alpha) / beta / dqdx

    def recombination_factor(self, dedx, cosphi=None):
        """Calls the predefined recombination models to evaluate the
        appropriate quenching factors.

        Parameters
        ----------
        dedx : Union[float, np.ndarray]
            Value or array of values of dEdx in MeV/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Quenching factors in electrons/MeV
        """
        if self.model == 'birks':
            return self.birks(dedx)
        else:
            return self.mbox(dedx, cosphi)

    def inv_recombination_factor(self, dqdx, cosphi=None):
        """Calls the predefined inverse recombination models to evaluate the
        appropriate correction factors.

        Parameters
        ----------
        dqdx : Union[float, np.ndarray]
            Value or array of values of dQ/dx in electrons/cm
        cosphi : Union[float, np.ndarray]
            Value or array of values of the cosine of the angle w.r.t. the
            drift direction (in [0,1]).

        Returns
        -------
        Union[float, np.ndarray]
            Inverse quenching factors in MeV/electrons
        """
        if self.model == 'birks':
            return self.inv_birks(dqdx)
        else:
            return self.inv_mbox(dqdx, cosphi)

    def process(self, values, points=None, track=False):
        """Corrects for electron recombination.

        Parameters
        ----------
        values : np.ndarray
            (N) array of depositions in number of electrons
        points : np.ndarray, optional
            (N, 3) array of point coordinates associated with one particle.
            Only needed if `track` is set to `True`.
        track : bool, defaut `False`
            Whether the object is a track or not. If it is, the track gets
            segmented to evaluate local dE/dx and track angle.

        Returns
        -------
        np.ndarray
            (N) array of depositions in MeV
        """
        # If no tracking is applied, use the MIP recombination factor
        if not track:
            return values * LAR_WION / self.mip_recomb

        # If the object is a track, segment the track use each segment to
        # compute a local dQ/dx (+ angle w.r.t. to the drift direction, if
        # requested) and assign a correction for all points in the segment.
        assert points is not None, (
                "Cannot track the object without point coordinates")
        seg_dqdxs, _, _, seg_clusts, seg_dirs, _ = get_track_segment_dedxs(
                points, values, method=self.tracking_mode,
                **self.tracking_kwargs)

        corr_values = np.empty(len(values), dtype=values.dtype)
        for i, c in enumerate(seg_clusts):
            if not self.use_angles:
                corr = self.inv_recombination_factor(seg_dqdxs[i])
            else:
                seg_cosphi = np.abs(np.dot(seg_dirs[i], self.drift_dir))
                corr = self.inv_recombination_factor(seg_dqdxs[i], seg_cosphi)

            corr_values[c] = corr * values[c]

        return corr_values
