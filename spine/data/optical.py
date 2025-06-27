"""Module with a data class object which represents optical information.

This copies the internal structure of :class:`larcv.Flash`.
"""

from dataclasses import dataclass

import numpy as np

from .base import PosDataBase

__all__ = ['Flash']


@dataclass(eq=False)
class Flash(PosDataBase):
    """Optical flash information.

    Attributes
    ----------
    id : int
        Index of the flash in the list
    interaction_id : int
        Index of the interaction in the list
    volume_id : int
        Index of the optical volume in which the flahs was recorded
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
    id: int = -1
    interaction_id: int = -1
    volume_id: int = -1
    frame: int = -1
    in_beam_frame: bool = False
    on_beam_time: bool = False
    time: float = -1.0
    time_width: float = -1.0
    time_abs: float = -1.0
    total_pe: float = -1.0
    fast_to_total: float = -1.0
    pe_per_ch: np.ndarray = None
    center: np.ndarray = None
    width: np.ndarray = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = (('center', 3), ('width', 3))

    # Variable-length attributes
    _var_length_attrs = (('pe_per_ch', np.float32),)

    # Attributes specifying coordinates
    _pos_attrs = ('center',)

    # Attributes specifying vector components
    _vec_attrs = ('width',)

    # Index attributes
    _index_attrs = ('id',)

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
        axes = ('x', 'y', 'z')
        center = np.array([getattr(flash, f'{a}Center')() for a in axes])
        width = np.array([getattr(flash, f'{a}Width')() for a in axes])

        # Get the number of PEs per optical channel
        pe_per_ch = np.array(list(flash.PEPerOpDet()), dtype=np.float32)

        # Get the volume ID, if it is filled (TODO: simplify with update)
        volume_id = -1
        for attr in ('tpc', 'volume_id'):
            if hasattr(flash, attr):
                volume_id = getattr(flash, attr)()

        return cls(id=flash.id(), volume_id=volume_id, frame=flash.frame(),
                   in_beam_frame=flash.inBeamFrame(),
                   on_beam_time=flash.onBeamTime(), time=flash.time(),
                   time_abs=flash.absTime(), time_width=flash.timeWidth(),
                   total_pe=flash.TotalPE(), pe_per_ch=pe_per_ch,
                   center=center, width=width)

    @classmethod
    def from_hypothesis(cls, flash, interaction_id, id, volume_id=None,negative_id=True):
        """Builds and returns a Flash object from a flashmatch::Flash_t object.
        From the hypothesis flash.

        Parameters
        ----------
        flash : flashmatch::Flash_t
            Flash object
        interaction_id : int
            Interaction ID to make the flash
        id : int
            ID of the flash
        volume_id : int, optional
            Volume ID to use for the flash, set manually if provided
        negative_id : bool, default True
            If `True`, use the negative of the id as the flash ID. This is used to identify which hypothesis is matched. If it's positive, it's matched.

        Returns
        -------
        Flash
            Flash object
        """
        # Get the number of PEs per optical channel
        pe_per_ch = np.array(flash.pe_v, dtype=np.float32)

        # Get the center and width of the flash
        center = np.array([flash.x, flash.y, flash.z])
        width = np.array([flash.x_err, flash.y_err, flash.z_err])
        
        #Get the volume ID
        if volume_id is None:
            volume_id = -1
            for attr in ('tpc', 'volume_id'):
                if hasattr(flash, attr):
                    volume_id = getattr(flash, attr)()

        # Create the Flash object
        if negative_id:
            id = -(id+1) #+1 to avoid negative zero
        return cls(id=id, interaction_id=interaction_id, volume_id=volume_id,
                   time=flash.time,
                   time_width=flash.time_width,
                   total_pe=flash.TotalPE(), pe_per_ch=pe_per_ch,
                   center=center, width=width)
    def __str__(self):
        return f"Flash(id={self.id}, interaction_id={self.interaction_id}, volume_id={self.volume_id}, time={self.time}, time_width={self.time_width}, total_pe={self.total_pe}, center={self.center}, width={self.width})"
