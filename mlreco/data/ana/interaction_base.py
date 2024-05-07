"""Module with a base class for both reconstruction and true interactions."""

from typing import List
from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import TRACK_SHP, SHAPE_LABELS, PID_LABELS, PID_MASSES
from mlreco.utils.numba_local import cdist

from mlreco.data.base import PosDataStructBase


@dataclass
class InteractionBase(PosDataStructBase):
    """Base interaction information.

    Attributes
    ----------
    id : int
        Unique index of the interaction within the interaction list
    nu_id : int
        Index of the neutrino matched to this interaction
    particle_ids : np.ndarray, 
        List of Particle IDs that make up this interaction
    index : np.ndarray
        (N) Voxel indexes corresponding to this interaction in the input tensor
    points : np.ndarray
        (N, 3) Set of voxel coordinates that make up this interaction
    sources : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs
    depositions : np.ndarray
        (N) Array of charge deposition values for each voxel
    vertex : np.ndarray
        (3) Coordinates of the ineraction vertex
    is_contained : bool
        Whether this interaction is contained within the detector
    is_fiducial : bool
        Whether this interaction vertex is inside the fiducial volume
    is_cathode_crosser : bool
        True if the interaction crossed a cathode, i.e. if it is made up
        of space points coming from > 1 TPC in one module
    if_flash_matched : bool
        True if the interaction was matched to an optical flash
    flash_id : int
        Index of the optical flash the interaction was matched to
    flash_time : float
        Time at which the flash occurred in nanoseconds
    flash_total_pe : float
        Total number of photoelectrons associated with the flash
    flash_hypo_pe : float
        Total number of photoelectrons expected to be produced by the interaction
    is_matched: bool
        True if a true interaction match was found
    match : np.ndarray
        List of Interaction IDs for which this interaction is matched to
    match_overlap : np.ndarray
        List of match overlaps (IoU) between the interaction and its matches
    is_truth: bool
        Whether this interaction contains truth information or not
    units : str
        Units in which coordinates are expressed
    """
    id: int = -1
    particles: List[object] = None
    particle_ids: np.ndarray = None
    index: np.ndarray = None
    points: np.ndarray = None
    sources: np.ndarray = None
    depositions: np.ndarray = None
    vertex: np.ndarray = None
    is_contained: bool = False
    is_fiducial: bool = False
    is_cathode_crosser: bool = False
    is_flash_matched: bool = False
    flash_id: int = -1
    flash_time: float = -1.
    flash_total_pe: float = -1.
    flash_hypo_pe: float = -1.
    is_matched: bool = False
    match: np.ndarray = None
    match_overlap: np.ndarray = None
    is_truth: bool = False
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = {'vertex': 3}

    # Variable-length attribtues
    _var_length_attrs = {'particle_ids': np.int64, 'index': np.int64,
                         'depositions': np.float32, 'match': np.int64,
                         'match_overlap': np.float32,
                         'points': (3, np.float32), 'sources': (2, np.int64)}

    # Attributes specifying coordinates
    _pos_attrs = ['vertex']

    # Attributes that should not be stored
    _skip_attrs = ['index', 'points', 'sources', 'depositions']

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        shape_label = SHAPE_LABELS[self.shape]
        pid_label = PID_LABELS[self.pid]
        match = self.match[0] if len(self.match) > 0 else -1
        return (f"Interaction(ID={self.id:<3} | Vertex={self.vertex} "
                f"| Size={self.size:<4} | Topology: {self.topology:<10})")

    @property
    def num_particles(self):
        """Number of particles that make up this interaction.

        Returns
        -------
        int
            Number of particles that make up the interaction instance
        """
        return len(self.particle_ids)

    @property
    def size(self):
        """Total number of voxels that make up the interaction.

        Returns
        -------
        int
            Total number of voxels in the interaction
        """
        return len(self.index)

    @property
    def depositions_sum(self):
        """Total deposition value for the entire interaction.

        Returns
        -------
        float
            Sum of all depositions that make up the interaction
        """
        return np.sum(self.depositions)

    @classmethod
    def from_particles(cls, particles):
        """Builds an Interaction instance from its constituent Particle objects.

        Parameters
        ----------
        particles : List[ParticleBase]
            List of Particle objects that make up the Interaction

        Returns
        -------
        InteractionBase
            Interaction built from the particle list
        """
        # Construct interaction object
        interaction = cls()

        # Fill unique attributes which must be shared between particles
        unique_attrs = ['is_truth', 'units']
        for attr in unique_attrs:
            assert len(np.unique([getattr(p, attr) for p in particles])) < 2, (
                    f"{attr} must be unique in the list of particles.")

        # Attach particle list
        interaction.particles = particles
        interaction.particle_ids = [p.id for p in particles]

        # Build long-form attributes
        for attr in self._skip_attrs:
            val_list = [getattr(p, attr) for p in particles]
            setattr(interaction, attr, np.concatenate(val_list))

        return interaction
