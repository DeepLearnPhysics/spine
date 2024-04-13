"""Module with a data class object which represents optical information.

This copies the internal structure of :class:`larcv.Flash`.
"""

import numpy as np
from dataclasses import dataclass
from larcv import larcv

__all__ = ['Flash']


@dataclass
class Flash:
    """Optical flash information.

    Attributes
    ----------
    id : int
        Index of the flash in the list
    time : float
        Time with respect to the trigger in microseconds
    time_width : float
        Width of the flash in microseconds
    time_abs : float
        Time in units of PMT readout clock
    frame : int
        Frame number
    in_beam_frame : bool
        Whether the flash is in the beam frame
    on_beam_time : bool
        Whether the flash time is consistent with the beam window
    total_pe : float
        Total number of PE in the flash
    fast_to_total : float
        Fraction of the total PE contributed by the fast component
    pe_per_optdet : np.ndarray
        (N) Fixed-length array of the number of PE per optical detector
    center : np.ndarray
        Barycenter of the flash in detector coordinates
    width : np.ndarray
        Spatial width of the flash in detector coordinates
    units : str
        Units in which the position coordinates are expressed
    """
    id: int               = -1
    frame: int            = -1
    in_beam_frame: bool   = False
    on_beam_time: bool    = False
    time: float           = -1.0
    time_width: float     = -1.0
    time_abs: float       = -1.0
    total_pe: float       = -1.0
    fast_to_total: float  = -1.0
    pe_per_ch: np.ndarray = np.empty(0, dtype=np.float32)
    center: np.ndarray    = np.full(3, -np.inf, dtype=np.float32)
    width: np.ndarray     = np.full(3, -np.inf, dtype=np.float32)
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = ['center', 'width']

    # Attributes specifying coordinates
    _pos_attrs = ['center']

    # String attributes
    _str_attrs = ['units']

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Used to type cast strings when they are provided as binary. Could
        also be used to check other inputs.
        """
        # Make sure  the strings are not binary
        for attr in self._str_attrs:
            if isinstance(getattr(self, attr), bytes):
                setattr(self, attr, getattr(self, attr).decode())

    def to_cm(self, meta):
        """Converts the coordinates of the positional attributes to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != '', "Must specify units"
        assert self.units != 'cm', "Units already expressed in cm"
        self.units = 'cm'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_cm(getattr(self, attr)))

    def to_pixel(self, meta):
        """Converts the coordinates of the positional attributes to pixel.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != '', "Must specify units"
        assert self.units != 'pixel', "Units already expressed in pixels"
        self.units = 'cm'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_pixel(getattr(self, attr)))

    @property
    def fixed_length_attrs(self):
        """Fetches the list of fixes-length array attributes.

        Returns
        -------
        List[str]
            List of fixed length array attribute names
        """
        return self._fixed_length_attrs

    @classmethod
    def from_larcv(cls, flash):
        """Builds and returns a Flash object from a LArCV Flash object.

        Parameters
        ----------
        flash : larcv.Flash
            LArCV-format optical flash

        Returns
        -------
        Flash
            Flash object
        """
        # Get the physical center and width of the flash
        axes = ['x', 'y', 'z']
        center = np.array([getattr(flash, f'{a}Center')() for a in axes])
        width = np.array([getattr(flash, f'{a}Width')() for a in axes])

        # Get the number of PEs per optical channel
        pe_per_ch = np.array(list(flash.PEPerOpDet()), dtype=np.float32)

        return cls(id=flash.id(), frame=flash.frame(),
                   in_beam_frame=flash.inBeamFrame(),
                   on_beam_time=flash.onBeamTime(), time=flash.time(),
                   time_abs=flash.absTime(), time_width=flash.timeWidth(),
                   total_pe=flash.TotalPE(), pe_per_ch=pe_per_ch,
                   center=center, width=width)
