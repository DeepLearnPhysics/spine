"""Module with a base class for both reconstruction and true fragments."""

from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import SHAPE_LABELS
from mlreco.utils.numba_local import cdist

from mlreco.data.base import PosDataStructBase


@dataclass
class FragmentBase(PosDataStructBase):
    """Base fragment information.

    Attributes
    ----------
    id : int
        Unique index of the fragment within the fragment list
    particle_id : int
        Index of the particle this fragment belongs to
    interaction_id : int
        Index of the interaction this fragment belongs to
    index : np.ndarray
        (N) Voxel indexes corresponding to this fragment in the input tensor
    points : np.ndarray
        (N, 3) Set of voxel coordinates that make up this fragment
    sources : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs
    depositions : np.ndarray
        (N) Array of charge deposition values for each voxel
    shape : int
        Semantic type (shower (0), track (1), Michel (2), delta (3),
        low energy scatter (4)) of this particle
    is_primary : bool
        Whether this fragment was the first in the particle group
    length : float
        Length of the particle (only assigned to track objects)
    start_point : np.ndarray
        (3) Particle start point
    end_point : np.ndarray
        (3) Particle end point (only assigned to track objects)
    start_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the start point
    end_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the end point (only assigned
        to track objects)
    is_matched: bool
        True if a true particle match was found
    match : np.ndarray
        List of Particle IDs for which this particle is matched to
    match_overlap : np.ndarray
        List of match overlaps (IoU) between the particle and its matches
    is_truth: bool
        Whether this fragment contains truth information or not
    units : str
        Units in which coordinates are expressed
    """
    id: int = -1
    particle_id: int = -1
    interaction_id: int = -1
    index: np.ndarray = None
    points: np.ndarray = None
    sources: np.ndarray = None
    depositions: np.ndarray = None
    shape: int = -1
    is_primary: bool = False
    length: float = -1.
    start_point: np.ndarray = None
    end_point: np.ndarray = None
    start_dir: np.ndarray = None
    end_dir: np.ndarray = None
    is_matched: bool = False
    match: np.ndarray = None
    match_overlap: np.ndarray = None
    is_truth: bool = False
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = {'start_point': 3, 'end_point': 3, 'start_dir': 3,
                           'end_dir': 3}

    # Variable-length attribtues
    _var_length_attrs = {'index': np.int64, 'depositions': np.float32,
                         'match': np.int64, 'match_overlap': np.float32,
                         'points': (3, np.float32), 'sources': (2, np.int64)}

    # Attributes specifying coordinates
    _pos_attrs = ['start_point', 'end_point']

    # Attributes specifying vector components
    _vec_attrs = ['start_dir', 'end_dir']

    # Enumerated attributes
    _enum_attrs = {
            'shape': {v : k for k, v in SHAPE_LABELS.items()}
    }

    # Attributes that should not be stored
    _skip_attrs = ['points', 'sources', 'depositions']

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        shape_label = SHAPE_LABELS[self.shape]
        match = self.match[0] if len(self.match) > 0 else -1
        return (f"Fragment(ID: {self.id:<3} | Shape: {shape_label:<11} "
                f"| Primary: {self.is_primary:<2} "
                f"| Particle ID: {self.particle_id} "
                f"| Interaction ID: {self.interaction_id:<2} "
                f"| Size: {self.size:<5} | Match: {match:<3})")

    @property
    def size(self):
        """Total number of voxels that make up the fragment.

        Returns
        -------
        int
            Total number of voxels in the fragment
        """
        return len(self.index)

    @property
    def depositions_sum(self):
        """Total deposition value for the entire fragment.

        Returns
        -------
        float
            Sum of all depositions that make up the fragment
        """
        return np.sum(self.depositions)
