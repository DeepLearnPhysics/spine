"""Module with a data class object which represents true particle information.

This copies the internal structure of :class:`larcv.Particle`.
"""

from warnings import warn
from dataclasses import dataclass, field

import numpy as np

from spine.utils.globals import UNKWN_SHP, SHAPE_LABELS, PID_LABELS

from .base import PosDataBase

__all__ = ['Particle']


@dataclass(eq=False)
class Particle(PosDataBase):
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
        True initial energy in MeV
    energy_deposit : float
        Amount of energy matched to this particle instance in MeV
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
    end_t : float
        Particle death time
    parent_t : float
        Particle creation time of the parent particle
    ancestor_t : float
        Particle creation time of the ancestor particle
    position : np.ndarray
        Location of the creation point of the particle
    end_position : np.ndarray
        Location where the particle stopped
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
    p : float
        Momentum magnitude of the particle at the production point
    end_p : float
        Momentum magnitude of the particle where it stops or exits the detector
    mass : float
        Rest mass of the particle in MeV/c^2
    units : str
        Units in which the position attributes are expressed
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
    children_id: int = None
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
    end_t: float = -np.inf
    parent_t: float = -np.inf
    ancestor_t: float = -np.inf
    position: np.ndarray = None
    end_position: np.ndarray = None
    parent_position: np.ndarray = None
    ancestor_position: np.ndarray = None
    first_step: np.ndarray = None
    last_step: np.ndarray = None
    momentum: np.ndarray = None
    end_momentum: np.ndarray = None
    mass: float = None
    p: float = None
    end_p: float = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = (
            ('position', 3), ('end_position', 3), ('parent_position', 3),
            ('ancestor_position', 3), ('first_step', 3), ('last_step', 3),
            ('momentum', 3), ('end_momentum', 3)
    )

    # Variable-length attributes
    _var_length_attrs = (('children_id', np.int64),)

    # Attributes specifying coordinates
    _pos_attrs = (
            'position', 'end_position', 'parent_position', 'ancestor_position',
            'first_step', 'last_step'
    )

    # Attributes specifying vector components
    _vec_attrs = ('momentum', 'end_momentum')

    # Enumerated attributes
    _enum_attrs = (
            ('shape', tuple((v, k) for k, v in SHAPE_LABELS.items())),
            ('pid', tuple((v, k) for k, v in PID_LABELS.items()))
    )

    # String attributes
    _str_attrs = ('creation_process', 'parent_creation_process',
                  'ancestor_creation_process')

    @property
    def p(self):
        """Computes the magnitude of the initial momentum.

        Returns
        -------
        float
            Norm of the initial momentum vector
        """
        return np.linalg.norm(self.momentum)

    @p.setter
    def p(self, p):
        pass

    @property
    def end_p(self):
        """Computes the magnitude of the final momentum.

        Returns
        -------
        float
            Norm of the final momentum vector
        """
        return np.linalg.norm(self.end_momentum)

    @end_p.setter
    def end_p(self, end_p):
        pass

    @property
    def mass(self):
        """Computes the rest mass of the particle from its energy/momentum.

        Returns
        -------
        float
            Rest mass of the particle in MeV/c^2
        """
        if self.energy_init < 0.:
            return -1.

        return np.sqrt(max(0., self.energy_init**2 - np.sum(self.momentum**2)))

    @mass.setter
    def mass(self, mass):
        pass

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
        for prefix in ('', 'parent_', 'ancestor_'):
            for key in ('track_id', 'pdg_code', 'creation_process', 't'):
                obj_dict[prefix+key] = getattr(particle, prefix+key)()
        for key in ('id', 'gen_id', 'group_id', 'interaction_id', 'parent_id',
                    'mct_index', 'mcst_index', 'num_voxels', 'shape',
                    'energy_init', 'energy_deposit', 'distance_travel'):
            if not hasattr(particle, key):
                warn(f"The LArCV Particle object is missing the {key} "
                      "attribute. It will miss from the Particle object.")
                continue
            obj_dict[key] = getattr(particle, key)()

        obj_dict['end_t'] = particle.end_position().t()

        # Load the positional attribute
        pos_attrs = ['x', 'y', 'z']
        for key in cls._pos_attrs:
            vector = getattr(particle, key)()
            obj_dict[key] = np.asarray(
                    [getattr(vector, a)() for a in pos_attrs], dtype=np.float32)

        # Load the other array attributes (special care needed)
        obj_dict['children_id'] = np.asarray(particle.children_id(), dtype=int)

        mom_attrs = ('px', 'py', 'pz')
        for prefix in ('', 'end_'):
            key = prefix + 'momentum'
            if not hasattr(particle, key):
                warn(f"The LArCV Particle object is missing the {key} "
                      "attribute. It will miss from the Particle object.")
                continue
            obj_dict[key] = np.asarray(
                    [getattr(particle, prefix + a)() for a in mom_attrs],
                    dtype=np.float32)

        return cls(**obj_dict)
