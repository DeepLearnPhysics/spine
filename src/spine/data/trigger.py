"""Module with a data class object which represents trigger information.

This copies the internal structure of :class:`larcv.Trigger`.
"""

from dataclasses import dataclass, field
from typing import Self

from .base import DataBase
from .field import FieldMetadata

__all__ = ["Trigger"]


@dataclass(eq=False)
class Trigger(DataBase):
    """Trigger information.

    Attributes
    ----------
    id : int
        Trigger ID
    type : int
        DAQ-specific trigger type
    time_s : int
        Integer seconds component of the UNIX trigger time
    time_ns : int
        Integer nanoseconds component of the UNIX trigger time
    beam_time_s : int
        Integer seconds component of the UNIX beam pulse time
    beam_time_ns : int
        Integer seconds component of the UNIX beam pulse time
    """

    # Scalar attributes
    id: int = -1
    type: int = -1
    time_s: int = field(default=-1, metadata=FieldMetadata(units="s"))
    time_ns: int = field(default=-1, metadata=FieldMetadata(units="ns"))
    beam_time_s: int = field(default=-1, metadata=FieldMetadata(units="s"))
    beam_time_ns: int = field(default=-1, metadata=FieldMetadata(units="ns"))

    @classmethod
    def from_larcv(cls, trigger) -> Self:
        """Builds and returns a Trigger object from a LArCV Trigger object.

        Parameters
        ----------
        trigger : larcv.Trigger
            LArCV-format trigger information

        Returns
        -------
        Trigger
            Trigger object
        """
        return cls(
            id=trigger.id(),
            time_s=trigger.time_s(),
            time_ns=trigger.time_ns(),
            type=trigger.type(),
        )
