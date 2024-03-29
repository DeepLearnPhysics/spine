"""Module with a data class object which represents CRT information.

This copies the internal structure of :class:`larcv.CRTHit`.
"""

import numpy as np
from dataclasses import dataclass
from larcv import larcv

__all__ = ['CRTHit']


@dataclass
class CRTHit:
    """CRT hit information.

    Attributes
    ----------
    id : int
        Index of the CRT hit in the list
    plane : int
        Index of the CRT tagger that registered the hit
    tagger : str
        Name of the CRT tagger that registered the hit
    feb_id : np.ndarray
        Address of the FEB board stored as a list of bytes (uint8)
    ts0_s : int
        Absolute time from White Rabbit (seconds component)
    ts0_ns : float
        Absolute time from White Rabbit (nanoseconds component)
    ts0_s_corr : float
        Unclear in the documentation, placeholder at this point
    ts0_ns_corr : float
        Unclear in the documentation, placeholder at this point
    ts1_ns : float
        Time relative to the trigger (nanoseconds component)
    total_pe : float
        Total number of PE in the CRT hit
    pe_per_ch : np.ndarray
        Number of PEs per FEB channel
    center : np.ndarray
        Barycenter of the CRT hit in detector coordinates
    width : np.ndarray
        Uncertainty on the barycenter of the CRT hit in detector coordinates
    units : str
        Units in which the position coordinates are expressed
    """
    id: int               = -1
    plane: int            = -1
    tagger: str           = ''
    feb_id: np.ndarray    = np.empty(0, dtype=np.ubyte)
    ts0_s: int            = -1
    ts0_ns: float         = -1.0
    ts0_s_corr: float     = -1.0
    ts0_ns_corr: float    = -1.0
    ts1_ns: float         = -1.0
    total_pe: float       = -1.0
    #pe_per_ch: np.ndarray = np.empty(0, dtype=np.float32)
    center: np.ndarray    = np.full(3, -np.inf, dtype=np.float32)
    width: np.ndarray     = np.full(3, -np.inf, dtype=np.float32)
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = ['center', 'width']

    # Attributes specifying coordinates
    _pos_attrs = ['position']

    # String attributes
    _str_attrs = ['tagger', 'units']

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

    @classmethod
    def from_larcv(cls, crthit):
        """Builds and returns a CRTHit object from a LArCV CRTHit object.

        Parameters
        ----------
        crthit : larcv.CRTHit
            LArCV-format CRT hit

        Returns
        -------
        CRTHit
            CRT hit object
        """
        # Get the physical center and width of the CRT hit
        axes = ['x', 'y', 'z']
        center = np.array([getattr(crthit, f'{a}_pos')() for a in axes])
        width = np.array([getattr(crthit, f'{a}_err')() for a in axes])

        # Convert the FEB address to a list of bytes
        feb_id = np.array([ord(c) for c in crthit.feb_id()], dtype=np.ubyte)

        # Get the number of PEs per FEB channel
        # TODO: This is a dictionary of dictionaries, hard to store

        return cls(id=crthit.id(), plane=crthit.plane(),
                   tagger=crthit.tagger(), feb_id=feb_id, ts0_s=crthit.ts0_s(),
                   ts0_ns=crthit.ts0_ns(), ts0_s_corr=crthit.ts0_s_corr(),
                   ts0_ns_corr=crthit.ts0_ns_corr(), ts1_ns=crthit.ts1_ns(),
                   total_pe=crthit.peshit(), center=center, width=width)
