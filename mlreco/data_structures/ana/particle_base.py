"""Module with a base class for both reconstruction and true particles."""

from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import TRACK_SHP, SHAPE_LABELS, PID_LABELS, PID_MASSES
from mlreco.utils.numba_local import cdist

from mlreco.data_structures.base import PosDataStructBase


@dataclass
class ParticleBase(PosDataStructBase):
    """Base particle information.

    Attributes
    ----------
    id : int
        Unique index of the particle within the particle list
    fragment_ids : np.ndarray, 
        List of Fragment IDs that make up this particle
    interaction_id : int
        Unique index of the interaction this particle belongs to
    index : np.ndarray
        (N) Voxel indexes corresponding to this particle in the input tensor
    points : np.ndarray
        (N, 3) Set of voxel coordinates that make up this particle
    sources : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs
    depositions : np.ndarray
        (N) Array of charge deposition values for each voxel
    shape : int
        Semantic type (shower (0), track (1), Michel (2), delta (3),
        low energy scatter (4)) of this particle
    pid : int
        Particle spcies (Photon (0), Electron (1), Muon (2), Charged Pion (3),
        Proton (4)) of this particle
    pdg_code : int
        PDG code corresponding to the PID number
    start_point : np.ndarray
        (3) Particle start point
    end_point : np.ndarray
        (3) Particle end point (only assigned to track objects)
    start_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the start point
    end_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the end point (only assigned
        to track objects)
    length : float
        Length of the particle (only assigned to track objects)
    is_primary : bool
        Whether this particle is a primary within its interaction
    is_contained : bool
        Whether this particle is contained within the detector
    is_valid : bool
        Whether this particle counts towards an interaction topology. This
        may be False if a particle is below some defined energy threshold.
    is_cathode_crosser : bool
        True if the particle crossed a cathode, i.e. if it is made up
        of space points coming from > 1 TPC in one module
    cathode_offset : float
        If the particle is a cathode crosser, this corresponds to the offset
        to apply to the particle to match its components at the cathode
    is_matched: bool
        True if a true particle match was found
    match : np.ndarray
        List of Particle IDs for which this particle is matched to
    match_overlap : np.ndarray
        List of match overlaps (IoU) between the particle and its matches
    units : str
        Units in which coordinates are expressed
    """
    id: int = -1
    fragment_ids: np.ndarray = None
    interaction_id: int = -1
    index: np.ndarray = None
    points: np.ndarray = None
    sources: np.ndarray = None
    depositions: np.ndarray = None
    shape: int = -1
    pid: int = -1
    pdg_code: int = -1
    is_primary: bool = False
    length: float = -1.
    start_point: np.ndarray = None
    end_point: np.ndarray = None
    start_dir: np.ndarray = None
    end_dir: np.ndarray = None
    ke: float = -1.
    momentum: np.ndarray = None
    is_contained: bool = False
    is_valid: bool = False
    is_cathode_crosser: bool = False
    cathode_offset: float = -1.
    is_matched: bool = False
    match: np.ndarray = None
    match_overlap: np.ndarray = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = {'start_point': 3, 'end_point': 3, 'start_dir': 3,
                           'end_dir': 3, 'momentum': 3}

    # Variable-length attribtues
    _var_length_attrs = {'fragment_ids': np.int64, 'index': np.int64,
                         'depositions': np.float32, 'match': np.int64,
                         'match_overlap': np.float32,
                         'points': (3, np.float32), 'sources': (2, np.int64)}

    # Attributes specifying coordinates
    _pos_attrs = ['start_point', 'end_point', 'start_dir', 'end_dir']

    # Enumerated attributes
    _enum_attrs = {
            'shape': {v : k for k, v in SHAPE_LABELS.items()},
            'pid': {v : k for k, v in PID_LABELS.items()}
    }

    # Attributes that should not be stored
    _skip_attrs = ['points', 'sources', 'depositions']

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        shape_label = SHAPE_LABELS[self.shape]
        pid_label = PID_LABELS[self.pid]
        match = self.match[0] if len(self.match) > 0 else -1
        return (f"Particle(ID={self.id:<3} | Semantic_type: {shape_label:<11} "
                f"| PID: {pid_label:<8} | Primary: {self.is_primary:<2} "
                f"| Interaction ID: {self.interaction_id:<2} "
                f"| Size: {self.size:<5} | Match: {match:<3})")

    @property
    def num_fragments(self):
        """Number of fragments that make up this particle.

        Returns
        -------
        int
            Number of fragments that make up the particle instance
        """
        return len(self.fragment_ids)

    @property
    def depositions_sum(self):
        """Total deposition value for the entire particle.

        Returns
        -------
        float
            Sum of all depositions that make up the particle
        """
        return np.sum(self.depositions)

    @property
    def size(self):
        """Total number of voxels that make up the particle.

        Returns
        -------
        int
            Total number of voxels in the particle
        """
        return len(self.index)
