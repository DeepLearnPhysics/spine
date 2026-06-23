"""Module with a data class object which represents optical information.

This copies the internal structure of :class:`larcv.Flash`.
"""

from dataclasses import dataclass, field

import numpy as np

from spine.data.base import PosDataBase
from spine.data.field import FieldMetadata

__all__ = ["Flash"]


@dataclass(eq=False, repr=False)
class Flash(PosDataBase):
    """Optical flash information.

    Attributes
    ----------
    id : int
        Index of the flash in the list
    volume_id : int
        Index of the optical volume in which the flash was recorded
    time : float
        Time with respect to the trigger in microseconds
    time_width : float
        Width of the flash in microseconds
    time_abs : float
        Time in units of PMT readout clock
    frame : int
        Frame number
    on_beam_time : int
        Whether the flash time is consistent with the beam window
        (inheriting typing from LArCV, so int instead of bool)
    in_beam_frame : bool
        Whether the flash is in the beam frame
    total_pe : float
        Total number of PE in the flash
    fast_to_total : float
        Fraction of the total PE contributed by the fast component
    pe_per_ch : np.ndarray
        (N) Fixed-length array of the number of PE per optical detector
    center : np.ndarray
        Barycenter of the flash in detector coordinates
    width : np.ndarray
        Spatial width of the flash in detector coordinates
    units : str
        Units in which the position coordinates are expressed
    """

    # Index attributes
    id: int = field(default=-1, metadata=FieldMetadata(index=True))

    # Scalar attributes
    volume_id: int = -1
    frame: int = -1
    on_beam_time: int = -1

    in_beam_frame: bool = False

    time: float = field(default=np.nan, metadata=FieldMetadata(units="us"))
    time_width: float = field(default=np.nan, metadata=FieldMetadata(units="us"))
    time_abs: float = np.nan
    total_pe: float = np.nan
    fast_to_total: float = np.nan

    units: str = "cm"

    # Vector attributes
    pe_per_ch: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32),
    )
    center: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="instance",
        ),
    )
    width: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            vector=True,
            units="instance",
        ),
    )

    @classmethod
    def from_larcv(cls, flash) -> "Flash":
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
        axes = ("x", "y", "z")
        center = np.array(
            [getattr(flash, f"{a}Center")() for a in axes], dtype=np.float32
        )
        width = np.array(
            [getattr(flash, f"{a}Width")() for a in axes], dtype=np.float32
        )

        # Get the number of PEs per optical channel
        pe_per_ch = np.array(list(flash.PEPerOpDet()), dtype=np.float32)

        # Backward compatibility: older LArCV versions do not include volume_id or tpc
        # information, so we check for both and set
        volume_id = -1
        for attr in ("tpc", "volume_id"):
            if hasattr(flash, attr):
                volume_id = getattr(flash, attr)()

        return cls(
            id=flash.id(),
            volume_id=volume_id,
            frame=flash.frame(),
            in_beam_frame=flash.inBeamFrame(),
            on_beam_time=flash.onBeamTime(),
            time=flash.time(),
            time_abs=flash.absTime(),
            time_width=flash.timeWidth(),
            total_pe=flash.TotalPE(),
            pe_per_ch=pe_per_ch,
            center=center,
            width=width,
        )

    def merge(self, other: "Flash") -> None:
        """Merge another flash into this one.

        The merge works as follows:

        - The earlier flash takes precedence for timing-related information.
        - The combined centroid is computed as the weighted average of the two
          existing centroids.
        - The PE values from each light collection system are added together.

        Parameters
        ----------
        other : Flash
            Flash to merge into this one
        """
        # Check that the position units are the same
        if self.units != other.units:
            raise ValueError("The units of the flash to be merged do not match.")

        # Determine the flash window end points (to merge time widths later)
        end_i, end_j = self.time + self.time_width, other.time + other.time_width

        # If the other flash happened first, update the timing information
        if self.time > other.time:
            self.time = other.time
            self.time_abs = other.time_abs
            self.on_beam_time = other.on_beam_time
            self.frame = other.frame
            self.in_beam_frame = other.in_beam_frame

        # Take the union of the two time widths as the new combined width
        self.time_width = max(end_i, end_j) - self.time

        # Take the weighted average of the centroids to compute the new one
        valid_mask = (self.width > 0.0) & (other.width > 0.0)

        w_i, w_j = 1.0 / self.width**2, 1.0 / other.width**2
        self.center = (w_i * self.center + w_j * other.center) / (w_i + w_j)

        self.width = 1.0 / np.sqrt(w_i + w_j)
        self.width[~valid_mask] = -1.0

        # Compute the new total PE and fast light component to total ratio
        t_i, t_j = self.total_pe, other.total_pe
        self.total_pe = t_i + t_j

        r_i, r_j = self.fast_to_total, other.fast_to_total
        self.fast_to_total = (r_i * t_i + r_j * t_j) / (t_i + t_j)

        # Merge the PE count in each PMT
        pe_per_ch = np.zeros(
            max(len(self.pe_per_ch), len(other.pe_per_ch)), dtype=self.pe_per_ch.dtype
        )
        pe_per_ch[: len(self.pe_per_ch)] += self.pe_per_ch
        pe_per_ch[: len(other.pe_per_ch)] += other.pe_per_ch

        self.pe_per_ch = pe_per_ch

        # The new volume ID is invalid if the two original volumes differ
        if self.volume_id != other.volume_id:
            self.volume_id = -1
