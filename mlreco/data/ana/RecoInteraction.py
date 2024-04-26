"""Module with a data class object which represents a reconstructed
interaction.
"""

from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import TRACK_SHP, SHAPE_LABELS, PID_LABELS, PID_MASSES
from mlreco.utils.numba_local import cdist

__all__ = ['RecoInteraction']


@dataclass
class RecoInteraction:
    """Reconstructed interaction information.

    Attributes
    ----------
    id : int
        Unique index of the interaction within the interaction list
    particle_ids : np.ndarray
        List of Particle IDs that make up this particle
    index : np.ndarray
        (N) Voxel indexes corresponding to this interaction in the input tensor
    points : np.dnarray
        (N, 3) Set of voxel coordinates that make up this interaction
    sources : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs
    depositions : np.ndarray
        (N) Array of charge deposition values for each voxel
    vertex : np.ndarray
        (3) Interaction vertex
    is_contained : bool
        Indicator whether this interaction is contained or not
    is_fiducial : bool
        Indicator whether this interaction vertex is fiducial or not
    is_cathode_crosser : bool
        True if the interaction crossed a cathode, i.e. if it is made up
        of space points coming from > 1 TPC in one module
    cathode_offset : float
        If the particle is a cathode crosser, this corresponds to the offset
        to apply to the particle to match its components at the cathode
    is_matched: bool
        True if a true particle match was found
    match : np.ndarray
        List of TruthParticle IDs for which this particle is matched to
    match_overlap : np.ndarray
        List of match overlaps (IoU) between the particle and its matches
    units : str
        Units in which coordinates are expressed
    """
    id: int = -1
    particle_ids: None
    index: np.ndarray = None
    points: np.ndarray = None
    sources: np.ndarray = None
    depositions: np.ndarray = None
    vertex: np.ndarray = None
    is_contained: bool = False
    is_fiducial: bool = False
    is_cathode_crosser: bool = False
    cathode_offset: float = -1.
    is_matched: bool = False
    match: np.ndarray = None
    match_overlap: np.ndarray = None
    particles: List[Particle] = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = ['vertex']

    # Attributes specifying coordinates
    _pos_attrs = ['vertex']

    # Enumerated attributes
    _enum_attrs = {}

    # String attributes
    _str_attrs = ['units']

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Provides two functions:
        - Gives default values to array-like attributes. If a default value was
          provided in the attribute definition, all instances of this class
          would point to the same memory location.
        - Casts strings when they are provided as binary objects, which is the
          format one gets when loading string from HDF5 files.
        """
        # Provide default values to the array-like attributes
        if self.points is None:
            self.points = np.empty((0, 3), dtype=np.float32)

        if self.sources is None:
            self.sources = np.empty((0, 2), dtype=np.int64)

        for attr in ['depositions', 'match_overlap']:
            if getattr(self, attr) is None:
                setattr(self, attr, np.empty(0, dtype=np.float32))

        for attr in ['particle_ids', 'index', 'match']:
            if getattr(self, attr) is None:
                setattr(self, attr, np.empty(0, dtype=np.int64))

        for attr in self._pos_attrs:
            if getattr(self, attr) is None:
                setattr(self, attr, np.full(3, -np.inf, dtype=np.float32))

        # Make sure  the strings are not binary
        for attr in self._str_attrs:
            if isinstance(getattr(self, attr), bytes):
                setattr(self, attr, getattr(self, attr).decode())

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        shape_label = SHAPE_LABELS[self.shape]
        pid_label = PID_LABELS[self.pid]
        return (f"Particle( Particle ID={self.id:<3} | Semantic_type: "
                f"{shape_label:<11} | PID: {pid_label:<8} | Primary: "
                f"{self.is_primary:<2} | Interaction ID: "
                f"{self.interaction_id:<2} | Size: {self.size:<5} | "
                f"Match: {self.match:<3})")

    @property
    def num_particles(self):
        """Number of particles that make up this interaction."""
        return len(self.fragment_ids)

    @property
    def depositions_sum(self):
        """Total deposition value for the entire partile."""
        return np.sum(self.depositions)

    @property
    def size(self):
        """Total number of voxels that make up the particle."""
        return len(self.index)

    @property
    def ke(self):
        """Best-guess kinetic energy in MeV.

        Uses calorimetry for EM activity and this order for track:
        - CSDA-based estimate if it is available
        - MCS-based estimate if it is available
        - Calorimetry if all else fails

        Returns
        -------
        float
            Best-guess kinetic energy
        """
        if self.shape != TRACK_SHP:
            # If a particle is not a track, can only use calorimetry
            return self.calo_ke

        else:
            # If a particle is a track, pick CSDA for contained tracks and
            # pick MCS for uncontained tracks, unless specified otherwise
            if self.is_contained and self.csda_ke > 0.:
                return self.csda_ke
            elif not self.is_contained and self.mcs_ke > 0.:
                return self.mcs_ke
            else:
                return self.calo_ke

    @property
    def momentum(self):
        """Best-guess momentum in MeV/c.

        Returns
        -------
        np.ndarray
            (3) Momentum vector
        """
        ke = self.ke
        if ke > 0. and self.start_dir[0] != -np.inf:
            mass = PID_MASSES[self.pid]
            mom = np.sqrt(ke**2 + 2*ke*mass)
            return mom * self.start_dir
        else:
            return np.full(3, -np.inf, dtype=np.float32)

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
    def fixed_length_attrs(self):
        """Fetches the list of fixes-length array attributes.

        Returns
        -------
        List[str]
            List of fixed length array attribute names
        """
        return self._fixed_length_attrs
