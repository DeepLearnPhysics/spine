"""Module with a data class object which represents true particle information.

This copies the internal structure of :class:`larcv.Particle`.
"""

import numpy as np
from dataclasses import dataclass
from larcv import larcv
from warnings import warn

from mlreco.utils.globals import UNKWN_SHP, SHAPE_LABELS, PID_LABELS

from .meta import Meta


@dataclass
class Particle:
    """Particle truth information.

    Attributes
    ----------
    id : int
        Index of the particle in the list
    mct_index : int
        Index in the original MCTruth array from whence it came
    mcst_index : int
        Index in the original MCTrack/MCShower array from whence it came
    gen_id : int
        Index of the particle at the generator level
    group_id : int
        Index of the group the particle belongs to
    interaction_id : int
        Index of the interaction the partile belongs to
    nu_id : int
        Index of the neutrino this particle belongs to
    interaction_primary : int
        Whether the particle is primary in its interaction or not
    group_primary : int
        Whether this particle is primary in its group or not
    parent_id : int
        Index of the parent particle
    children_id : np.ndarray
        List of indexes of the children particles
    track_id : int
        Geant4 track ID
    parent_track_id : int
        Geant4 track ID of the parent particle
    ancestor_track_id : int
        Geant4 track ID of the ancestor particle
    shape : int
        Enumerated semantic type of the particle
    num_voxels : int
        Number of voxels matched to this particle instance
    energy_init : float
        True initial energy in GeV
    energy_deposit : float
        Amount of energy matched to this particle instance in GeV
    distance_travel : float
        True amount of distance traveled by the particle in the active volume
    creation_process : str
        Creation process
    parent_creation_process : str
        Creation process of the parent particle
    ancestor_creation_process : str
        Creation process of the ancestor particle
    pid : int
        Enumerated particle species type of the particle
    pdg_code : int
        Particle PDG code
    parent_pdg_code : int
        Particle PDG code of the parent particle
    ancestor_pdg_code : int
        Particle PDG code of the ancestor particle
    t : float
        Particle creation time
    parent_t : float
        Particle creation time of the parent particle
    ancestor_t : float
        Particle creation time of the ancestor particle
    position : np.ndarray
        Location of the creation point of the particle
    end_position : np.ndarray
        Location where the particle stopped or exited the detector
    parent_position : np.ndarry
        Location of the creation point of the parent particle
    ancestor_position : np.ndarray
        Location of the creation point of the ancestor particle
    first_step : np.ndarray
        Location of the first energy deposition of the particle
    last_step : np.ndarray
        Location of the last energy deposition of the particle
    momentum : np.ndarray
        3-momentum of the particle at the production point
    end_momentum : np.ndarray
        3-momentum of the particle at where it stops or exits the detector
    units : str
        Units in which the position coordinates are expressed
    """
    # Attributes
    id: int = -1
    mct_index: int = -1
    mcst_index: int = -1
    gen_id: int = -1
    group_id: int = -1
    interaction_id: int = -1
    nu_id: int = -1
    interaction_primary: int = -1
    group_primary: int = -1
    parent_id: int = -1
    children_id: int = np.empty(0, dtype=np.int64)
    track_id: int = -1
    parent_track_id: int = -1
    ancestor_track_id: int = -1
    pid: int = -1
    pdg_code: int = -1
    parent_pdg_code: int = -1
    ancestor_pdg_code: int = -1
    num_voxels: int = -1
    shape: int = UNKWN_SHP
    energy_init: float = -1.
    energy_deposit: float = -1.
    distance_travel: float = -1.
    creation_process: str = ''
    parent_creation_process: str = ''
    ancestor_creation_process: str = ''
    t: float = -np.inf
    parent_t: float = -np.inf
    ancestor_t: float = -np.inf
    position: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    end_position: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    parent_position: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    ancestor_position: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    first_step: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    last_step: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    momentum: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    end_momentum: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = ['position', 'end_position', 'parent_position',
                           'ancestor_position', 'first_step', 'last_step',
                           'momentum', 'end_momentum']

    # Attributes specifying coordinates
    _pos_attrs = ['position', 'end_position', 'parent_position',
                  'ancestor_position', 'first_step', 'last_step']

    # Enumerated attributes
    _enum_attrs = {
            'shape': {v : k for k, v in SHAPE_LABELS.items()},
            'pid': {v : k for k, v in SHAPE_LABELS.items()}
    }

    def to_cm(self, meta):
        """Converts the coordinates of the positional attributes to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
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
        assert self.units != 'pixel', "Units already expressed in pixels"
        self.units = 'pixel'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_pixel(getattr(self, attr)))

    @property
    def p(self):
        """Computes the magnitude of the initial momentum.

        Returns
        -------
        float
            Norm of the initial momentum vector
        """
        return np.linalg.norm(self.momentum)

    @property
    def end_p(self):
        """Computes the magnitude of the final momentum.

        Returns
        -------
        float
            Norm of the final momentum vector
        """
        return np.linalg.norm(self.end_momentum)

    @classmethod
    def from_larcv(cls, particle):
        """Builds and returns a Particle object from a LArCV Particle object.

        Parameters
        ----------
        particle : larcv.Particle
            LArCV-format particle object

        Returns
        -------
        Particle
            Particle object
        """
        # Initialize the dictionary to initialize the object with
        obj_dict = {}

        # Load the scalar attributes
        for prefix in ['', 'parent_', 'ancestor_']:
            for key in ['track_id', 'pdg_code', 'creation_process', 't']:
                obj_dict[prefix+key] = getattr(particle, prefix+key)()
        for key in ['id', 'gen_id', 'group_id', 'interaction_id', 'parent_id',
                    'mct_index', 'mcst_index', 'shape', 'energy_init',
                    'energy_deposit', 'distance_travel']:
            if not hasattr(particle, key):
                warn(f"The LArCV Particle object is missing the {key} "
                      "attribute. It will miss from the Particle object.")
                continue
            obj_dict[key] = getattr(particle, key)()

        # Load the positional attribute
        pos_attrs = ['x', 'y', 'z']
        for key in cls._pos_attrs:
            vector = getattr(particle, key)()
            obj_dict[key] = np.asarray(
                    [getattr(vector, a)() for a in pos_attrs], dtype=np.float32)

        # Load the other array attributes (special care needed)
        obj_dict['children_id'] = np.asarray(particle.children_id(), dtype=int)

        mom_attrs = ['px', 'py', 'pz']
        for prefix in ['', 'end_']:
            key = prefix + 'momentum'
            if not hasattr(particle, key):
                warn(f"The LArCV Particle object is missing the {key} "
                      "attribute. It will miss from the Particle object.")
                continue
            obj_dict[key] = np.asarray(
                    [getattr(particle, prefix + a)() for a in mom_attrs],
                    dtype=np.float32)

        return cls(**obj_dict)
