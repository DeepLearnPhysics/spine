"""Module with a data class object which represents CRT information.

This copies the internal structure of :class:`larcv.CRTHit`.
"""

from dataclasses import dataclass, field

import numpy as np

from .base import PosDataBase
from .derived import derived_property

__all__ = ["CRTHit"]


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
    time : float
        Alias for ts1_ns, but scaled to microseconds
    total_pe : float
        Total number of PE in the CRT hit
    center : np.ndarray
        Barycenter of the CRT hit in detector coordinates
    width : np.ndarray
        Uncertainty on the barycenter of the CRT hit in detector coordinates
    units : str
        Units in which the position attributes are expressed
    """

    # Index attributes
    id: int = field(default=-1, metadata={"index": True})

    # Scalar attributes
    plane: int = -1

    ts0_s: int = field(default=-1, metadata={"units": "s"})  # Integer to match LArCV
    ts0_ns: float = field(default=np.nan, metadata={"units": "ns"})
    ts0_s_corr: float = field(default=np.nan, metadata={"units": "s"})
    ts0_ns_corr: float = field(default=np.nan, metadata={"units": "ns"})
    ts1_ns: float = field(default=np.nan, metadata={"units": "ns"})
    total_pe: float = np.nan

    tagger: str = ""
    units: str = "cm"

    # Vector attributes
    feb_id: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.ubyte),
        metadata={"dtype": np.ubyte},
    )
    center: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata={
            "length": 3,
            "dtype": np.float32,
            "type": "position",
            "units": "instance",
        },
    )
    width: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata={
            "length": 3,
            "dtype": np.float32,
            "type": "vector",
            "units": "instance",
        },
    )

    @derived_property(units="us")
    def time(self) -> float:
        """Time w.r.t. to the trigger in microseconds.

        Returns
        -------
        float
            Time of the CRT hit w.r.t. to the trigger in microseconds.
        """
        return self.ts1_ns / 1000.0

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
        center = np.array(
            [getattr(crthit, f"{a}_pos")() for a in cls._axes], dtype=np.float32
        )
        width = np.array(
            [getattr(crthit, f"{a}_err")() for a in cls._axes], dtype=np.float32
        )

        # Convert the FEB address to a list of bytes
        feb_id = np.array([ord(c) for c in crthit.feb_id()], dtype=np.ubyte)

        return cls(
            id=crthit.id(),
            plane=crthit.plane(),
            tagger=crthit.tagger(),
            feb_id=feb_id,
            ts0_s=crthit.ts0_s(),
            ts0_ns=crthit.ts0_ns(),
            ts0_s_corr=crthit.ts0_s_corr(),
            ts0_ns_corr=crthit.ts0_ns_corr(),
            ts1_ns=crthit.ts1_ns(),
            total_pe=crthit.peshit(),
            center=center,
            width=width,
        )
