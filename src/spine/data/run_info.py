"""Module with a data class object which represents the run information.

It can extract run attributes from any event-level LArCV object.
"""

from dataclasses import dataclass

from .base import DataBase

__all__ = ["RunInfo"]


@dataclass(eq=False)
class RunInfo(DataBase):
    """Run information related to a specific event.

    Attributes
    ----------
    run : int
        Run ID
    subrun : int
        Sub-run ID
    event : int
        Event ID
    """

    run: int = -1
    subrun: int = -1
    event: int = -1

    @classmethod
    def from_larcv(cls, larcv_event):
        """
        Builds and returns a Meta object from a LArCV 2D metadata object

        Parameters
        ----------
        larcv_event : larcv.EventBase
             LArCV event object which contains the run information as attributes

        Returns
        -------
        Meta
            Metadata object
        """
        return cls(
            run=larcv_event.run(),
            subrun=larcv_event.subrun(),
            event=larcv_event.event(),
        )
