"""Module with a data class object which represents trigger information.

This copies the internal structure of :class:`larcv.Trigger`.
"""

from dataclasses import dataclass

from .base import DataBase

__all__ = ["Trigger"]


@dataclass(eq=False)
class Trigger(DataBase):
    """Trigger information.

    Attributes
    ----------
    id : int
        Trigger ID
    time_s : int
        Integer seconds component of the UNIX trigger time
    time_ns : int
        Integer nanoseconds component of the UNIX trigger time
    beam_time_s : int
        Integer seconds component of the UNIX beam pulse time
    beam_time_ns : int
        Integer seconds component of the UNIX beam pulse time
    type : int
        DAQ-specific trigger type
    """

    id: int = -1
    time_s: int = -1
    time_ns: int = -1
    beam_time_s: int = -1
    beam_time_ns: int = -1
    type: int = -1

    @classmethod
    def from_larcv(cls, trigger):
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
