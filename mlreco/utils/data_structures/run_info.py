"""Module with a data class object which represents the run information.

It can extract run attributes from any event-level LArCV object.
"""

from dataclasses import dataclass

__all__ = ['RunInfo']


@dataclass
class RunInfo:
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
    run: int    = -1
    subrun: int = -1
    event: int  = -1

    @classmethod
    def from_larcv(cls, tensor):
        """
        Builds and returns a Meta object from a LArCV 2D metadata object

        Parameters
        ----------
        larcv_class : object
             LArCV tensor which contains the run information as attributes

        Returns
        -------
        Meta
            Metadata object
        """
        return cls(run=tensor.run(),
                   subrun=tensor.subrun(),
                   event=tensor.event())
