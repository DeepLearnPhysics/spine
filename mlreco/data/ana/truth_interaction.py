"""Module with a data class object which represents a reconstructed interaction."""

from dataclasses import dataclass, asdict

import numpy as np

from mlreco.utils.globals import TRACK_SHP, SHAPE_LABELS, PID_LABELS, PID_MASSES
from mlreco.utils.numba_local import cdist
from mlreco.utils.decorators import inherit_docstring

from mlreco.data import Neutrino

from .interaction_base import InteractionBase

__all__ = ['TruthInteraction']


@dataclass
@inherit_docstring(InteractionBase)
class TruthInteraction(Neutrino, InteractionBase):
    """Truth interaction information.

    Attributes
    ----------
    orig_id : int
        Unaltered index of the interaction in the MC truth
    nu_id : int
        Index of the neutrino attached to this interaction
    index_adapt: np.ndarray
        (N) Interaction voxel indexes in the adapted cluster label tensor
    points_adapt : np.ndarray
        (N, 3) Set of voxel coordinates using adapted cluster labels
    sources_adapt : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs, adapted
    depositions_adapt : np.ndarray
        (N) Array of values for each voxel in the adapted cluster label tensor
    index_g4: np.ndarray
        (N) Interaction voxel indexes in the true Geant4 energy deposition tensor
    points_g4 : np.ndarray
        (N, 3) Set of voxel coordinates of true Geant4 energy depositions
    depositions_g4 : np.ndarray
        (N) Array of true Geant4 energy depositions per voxel
    """
    orig_id: int = -1
    nu_id: int = -1
    index_adapt: np.ndarray = None
    points_adapt: np.ndarray = None
    sources_adapt: np.ndarray = None
    depositions_adapt: np.ndarray = None
    index_g4: np.ndarray = None
    points_g4: np.ndarray = None
    depositions_g4: np.ndarray = None

    # Variable-length attributes
    _var_length_attrs = {
            'index_adapt': np.int64, 'index_g4': np.int64,
            'depositions_adapt': np.float32, 'depositions_g4': np.float32,
            'points_adapt': (3, np.float32), 'points_g4': (3, np.float32),
            'sources_adapt': (2, np.int64),
            **InteractionBase._var_length_attrs}

    # Attributes that should not be stored
    _skip_attrs = [
            'points_adapt', 'sources_adapt', 'depositions_adapt',
            'points_g4', 'depositions_g4', *InteractionBase._skip_attrs]

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return 'Truth' + super().__str__()

    def attach_neutrino(self, neutrino):
        """Attach neutrino generator information to this interaction.

        Parameters
        ----------
        neutrino : Neutrino
            Neutrino to fetch the attributes from
        """
        for attr, val in asdict(neutrino):
            if attr != 'id':
                setattr(self, attr, val)
            else:
                self.nu_id = val
