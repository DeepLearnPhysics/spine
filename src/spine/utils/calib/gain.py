"""Applies conversion form ADC to number ionization electrons."""

import numpy as np

__all__ = ["GainCalibrator"]


class GainCalibrator:
    """Converts all charge depositions in ADC to a number of electrons. It can
    either use a flat converstion factor or one per TPC in the detector
    """

    name = "gain"

    def __init__(self, gain, num_tpcs):
        """Initialize the recombination model and its constants.

        Parameters
        ----------
        gain : Union[list, float]
            Conversion factor from ADC to electrons (unique or per tpc)
        num_tpcs : int
            Number of TPCs in the detector
        """
        # Initialize the gain values
        assert (
            np.isscalar(gain) or len(gain) == num_tpcs
        ), f"Gain must be a scalar or given per TPC ({num_tpcs})."

        if np.isscalar(gain):
            self.gain = np.full(num_tpcs, gain)
        else:
            self.gain = gain

    def process(self, values, tpc_id):
        """Converts deposition values from ADC to a number of electrons.

        Parameters
        ----------
        values : np.ndarray
            (N) array of depositions in ADC in a specific TPC
        tpc_id : int
            ID of the TPC to use

        Returns
        -------
        np.ndarray
            (N) array of depositions in number of electrons
        """
        # Apply the gain factor to all values in the current TPC
        return values * self.gain[tpc_id]
