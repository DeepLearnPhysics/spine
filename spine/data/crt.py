"""Module with a data class object which represents CRT information.

This copies the internal structure of :class:`larcv.CRTHit`.
"""

from dataclasses import dataclass

import numpy as np

from .base import PosDataBase

__all__ = ['CRTHit']


@dataclass(eq=False)
class CRTHit(PosDataBase):
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
        Units in which the position attributes are expressed
    """
    id: int = -1
    plane: int = -1
    tagger: str = ''
    feb_id: np.ndarray = None
    ts0_s: int = -1
    ts0_ns: float = -1.0
    ts0_s_corr: float = -1.0
    ts0_ns_corr: float = -1.0
    ts1_ns: float = -1.0
    total_pe: float = -1.0
    # pe_per_ch: np.ndarray = None
    center: np.ndarray = None
    width: np.ndarray = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = (('center', 3), ('width', 3))

    # Variable-length attributes
    _var_length_attrs = (('feb_id', np.ubyte),)

    # Attributes specifying coordinates
    _pos_attrs = ('position',)

    # Attributes specifying vector components
    _vec_attrs = ('width',)

    # String attributes
    _str_attrs = ('tagger', 'units')

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
        axes = ('x', 'y', 'z')
        center = np.array([getattr(crthit, f'{a}_pos')() for a in axes])
        width = np.array([getattr(crthit, f'{a}_err')() for a in axes])

        # Convert the FEB address to a list of bytes
        feb_id = np.array([ord(c) for c in crthit.feb_id()], dtype=np.ubyte)

        # Get the number of PEs per FEB channel
        # TODO: This is a dictionary of dictionaries, need to figure out
        # how to unpack in a sensible manner

        return cls(id=crthit.id(), plane=crthit.plane(),
                   tagger=crthit.tagger(), feb_id=feb_id, ts0_s=crthit.ts0_s(),
                   ts0_ns=crthit.ts0_ns(), ts0_s_corr=crthit.ts0_s_corr(),
                   ts0_ns_corr=crthit.ts0_ns_corr(), ts1_ns=crthit.ts1_ns(),
                   total_pe=crthit.peshit(), center=center, width=width)
