"""Module with a data class objects which represent output particles."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial.distance import cdist

from spine.data.particle import Particle
from spine.utils.docstring import inherit_docstring
from spine.utils.globals import (
    PID_LABELS,
    PID_MASSES,
    PID_TO_PDG,
    SHAPE_LABELS,
    SHOWR_SHP,
    TRACK_SHP,
)

from .base import RecoBase, TruthBase

__all__ = ["RecoParticle", "TruthParticle"]


@dataclass(eq=False)
class ParticleBase:
    """Base particle-specific information.

    Attributes
    ----------
    fragments : List[object]
        List of fragments that make up this particle
    fragment_ids : np.ndarray
        List of Fragment IDs that make up this particle
    num_fragments : int
        Number of fragments that make up this particle
    interaction_id : int
        Index of the interaction this particle belongs to
    shape : int
        Semantic type (shower (0), track (1), Michel (2), delta (3),
        low energy scatter (4)) of this particle
    pid : int
        Particle species (Photon (0), Electron (1), Muon (2), Charged Pion (3),
        Proton (4), Kaon (5)) of this particle
    chi2_pid : int
        Particle species as predicted by the chi2 template method (Muon (2),
        Charged Pion (3), Proton (4), Kaon (5)) of this particle
    chi2_per_pid : np.ndarray
        (P) Array of chi2 values associated with each particle class
    pdg_code : int
        PDG code corresponding to the PID number
    is_primary : bool
        Whether this particle was the first in the particle group
    length : float
        Length of the particle (only assigned to track objects)
    start_point : np.ndarray
        (3) Particle start point
    end_point : np.ndarray
        (3) Particle end point (only assigned to track objects)
    start_dir : np.ndarray
        (3) Particle direction w.r.t. the start point
    end_dir : np.ndarray
        (3) Particle direction w.r.t. the end point (only assigned
        to track objects)
    mass : float
        Rest mass of the particle in MeV/c^2
    ke : float
        Kinetic energy of the particle in MeV
    calo_ke : float
        Kinetic energy reconstructed from the energy depositions alone in MeV
    csda_ke : float
        Kinetic energy reconstructed from the particle range in MeV
    csda_ke_per_pid : np.ndarray
        (P) Same as `csda_ke` but for every available track PID hypothesis
    mcs_ke : float
        Kinetic energy reconstructed using the MCS method in MeV
    mcs_ke_per_pid : np.ndarray
        (P) Same as `mcs_ke` but for every available track PID hypothesis
    momentum : np.ndarray
        3-momentum of the particle at the production point in MeV/c
    p : float
        Momentum magnitude of the particle at the production point in MeV/c
    is_crt_matched : bool
        True if the particle was matched to a CRT hit
    crt_ids : np.ndarray
        (C) Indices of the CRT hits the particle is matched to
    crt_times : np.ndarray
        (C) Times at which the CRT hits occurred in microseconds
    crt_scores : np.ndarray
        (C) Quality metric associated with the CRT matches
    is_valid : bool
        Whether this particle counts towards an interaction topology. This
        may be False if a particle is below some defined energy threshold.
    """

    fragments: List[object] = None
    fragment_ids: np.ndarray = None
    num_fragments: int = None
    interaction_id: int = -1
    shape: int = -1
    pid: int = -1
    chi2_pid: int = -1
    chi2_per_pid: np.ndarray = None
    pdg_code: int = -1
    is_primary: bool = False
    length: float = -1.0
    start_point: np.ndarray = None
    end_point: np.ndarray = None
    start_dir: np.ndarray = None
    end_dir: np.ndarray = None
    mass: float = -1.0
    ke: float = -1.0
    calo_ke: float = -1.0
    csda_ke: float = -1.0
    csda_ke_per_pid: np.ndarray = None
    mcs_ke: float = -1.0
    mcs_ke_per_pid: np.ndarray = None
    momentum: np.ndarray = None
    p: float = None
    is_crt_matched: bool = False
    crt_ids: np.ndarray = None
    crt_times: np.ndarray = None
    crt_scores: np.ndarray = None
    is_valid: bool = True

    # Fixed-length attributes
    _fixed_length_attrs = (
        ("start_point", 3),
        ("end_point", 3),
        ("start_dir", 3),
        ("end_dir", 3),
        ("momentum", 3),
        ("chi2_per_pid", len(PID_LABELS) - 1),
        ("csda_ke_per_pid", len(PID_LABELS) - 1),
        ("mcs_ke_per_pid", len(PID_LABELS) - 1),
    )

    # Variable-length attributes as (key, dtype) pairs
    _var_length_attrs = (
        ("fragments", object),
        ("fragment_ids", np.int32),
        ("crt_ids", np.int32),
        ("crt_times", np.float32),
        ("crt_scores", np.float32),
    )

    # Attributes specifying coordinates
    _pos_attrs = ("start_point", "end_point")

    # Attributes specifying vector components
    _vec_attrs = ("start_dir", "end_dir", "momentum")

    # Boolean attributes
    _bool_attrs = ("is_primary", "is_crt_matched", "is_valid")

    # Enumerated attributes
    _enum_attrs = (
        ("shape", tuple((v, k) for k, v in SHAPE_LABELS.items())),
        ("pid", tuple((v, k) for k, v in PID_LABELS.items())),
    )

    # Attributes that must never be stored to file
    _skip_attrs = ("fragments", "ppn_points")

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        pid_label = PID_LABELS[self.pid]
        match = self.match_ids[0] if len(self.match_ids) > 0 else -1
        return (
            f"Particle(ID: {self.id:<3} | PID: {pid_label:<8} "
            f"| Primary: {self.is_primary:<2} "
            f"| Size: {self.size:<5} | Match: {match:<3})"
        )

    def reset_crt_match(self, typed=True):
        """Reset all the CRT hit matching attributes.

        Parameters
        ----------
        typed : bool, default True
            If `True`, the underlying arrays are reset to typed empty arrays
        """
        self.is_crt_matched = False
        if typed:
            self.crt_ids = np.empty(0, dtype=np.int32)
            self.crt_times = np.empty(0, dtype=np.float32)
            self.crt_scores = np.empty(0, dtype=np.float32)
        else:
            self.crt_ids = []
            self.crt_times = []
            self.crt_scores = []

    @property
    def num_fragments(self):
        """Number of fragments that make up this particle.

        Returns
        -------
        int
            Number of fragments that make up the particle instance
        """
        return len(self.fragment_ids)

    @num_fragments.setter
    def num_fragments(self, num_fragments):
        pass

    @property
    def pdg_code(self):
        """Translates the enumerated particle type to a sign-less PDG code.

        Returns
        -------
        int
            Reconstructed sign-less PDG code
        """
        return PID_TO_PDG[self.pid]

    @pdg_code.setter
    def pdg_code(self, pdg_code):
        pass

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


@dataclass(eq=False)
@inherit_docstring(RecoBase, ParticleBase)
class RecoParticle(ParticleBase, RecoBase):
    """Reconstructed particle information.

    Attributes
    ----------
    pid_scores : np.ndarray
        (P) Array of softmax scores associated with each particle class
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    ppn_ids : np.ndarray
        (M) List of indexes of PPN points associated with this particle
    ppn_points : np.ndarray
        (M, 3) List of PPN points tagged to this particle
    vertex_distance : float
        Set-to-point distance between all particle points and the parent
        interaction vertex position in cm
    start_dedx : float
        dE/dx around a user-defined neighborhood of the start point in MeV/cm
    end_dedx : float
        dE/dx around a user-defined neighborhood of the end point in MeV/cm
    start_straightness : float
        Explained variance ratio of the beginning of the particle
    directional_spread : float
        Estimate of the angular spread of the particle (cosine spread)
    axial_spread : float
        Pearson correlation coefficient of the axial profile of the particle
        w.r.t. to the distance from its start point
    """

    pid_scores: np.ndarray = None
    primary_scores: np.ndarray = None
    ppn_ids: np.ndarray = None
    ppn_points: np.ndarray = None
    vertex_distance: float = -1.0
    start_dedx: float = -1.0
    end_dedx: float = -1.0
    start_straightness: float = -1.0
    directional_spread: float = -1.0
    axial_spread: float = -np.inf

    # Fixed-length attributes
    _fixed_length_attrs = (
        ("pid_scores", len(PID_LABELS) - 1),
        ("primary_scores", 2),
        *ParticleBase._fixed_length_attrs,
    )

    # Variable-length attributes
    _var_length_attrs = (
        ("ppn_ids", np.int32),
        ("ppn_points", (3, np.float32)),
        *RecoBase._var_length_attrs,
        *ParticleBase._var_length_attrs,
    )

    # Boolean attributes
    _bool_attrs = (*RecoBase._bool_attrs, *ParticleBase._bool_attrs)

    # Attributes that must never be stored to file
    _skip_attrs = ("ppn_points", *RecoBase._skip_attrs, *ParticleBase._skip_attrs)

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return "Reco" + super().__str__()

    def merge(self, other):
        """Merge another particle instance into this one.

        The merging strategy differs depending on the the particle shapes
        merged together. There are two categories:
        - Track + track
          - The start/end points are produced by finding the combination of points
            which are farthest away from each other (one from each constituent)
          - The primary scores/primary status match that of the constituent
            particle with the highest primary score
          - The PID scores/PID value match that of the constituent particle with
            the highest primary score
        - Shower + Track
          - The track is always merged into the shower, not the other way around
          - The start point of the shower is updated to be the track end point
          further away from the current shower start point
          - The primary scores/primary status match that of the constituent
            particle with the highest primary score
          - The PID scores/PID value is kept unchanged (that of the shower)

        Parameters
        ----------
        other : RecoParticle
            Other reconstructed particle to merge into this one
        """
        # Check that the particles being merged fit one of two categories
        assert (
            self.shape in (SHOWR_SHP, TRACK_SHP) and other.shape == TRACK_SHP
        ), "Can only merge two track particles or a track into a shower."

        # Check that neither particle has yet been matched
        assert (
            not self.is_matched and not other.is_matched
        ), "Cannot merge particles that already have matches."

        # Concatenate the two particle long-form attributes together
        for attr in self._cat_attrs:
            val = np.concatenate([getattr(self, attr), getattr(other, attr)])
            setattr(self, attr, val)

        # Select end points and end directions appropriately
        if self.shape == TRACK_SHP:
            # If two tracks, pick points furthest apart
            points_i = np.vstack([self.start_point, self.end_point])
            points_j = np.vstack([other.start_point, other.end_point])
            dirs_i = np.vstack([self.start_dir, self.end_dir])
            dirs_j = np.vstack([other.start_dir, other.end_dir])

            dists = cdist(points_i, points_j)
            max_index = np.argmax(dists)
            max_i, max_j = max_index // 2, max_index % 2

            self.start_point = points_i[max_i]
            self.end_point = points_j[max_j]
            self.start_dir = dirs_i[max_i]
            self.end_dir = dirs_j[max_j]

        else:
            # If a shower and a track, pick track point furthest from shower
            points_i = self.start_point.reshape(-1, 3)
            points_j = np.vstack([other.start_point, other.end_point])
            dirs_j = np.vstack([other.start_dir, other.end_dir])

            dists = cdist(points_i, points_j)
            max_j = np.argmax(dists)

            self.start_point = points_j[max_j]
            self.start_dir = dirs_j[max_j]

        # Match primary/PID to the most primary particle
        if other.primary_scores[-1] > self.primary_scores[-1]:
            self.primary_scores = other.primary_scores
            self.is_primary = other.is_primary
            if self.shape == TRACK_SHP:
                self.pid_scores = other.pid_scores
                self.pid = other.pid

        # If the calorimetric KEs have been computed, can safely sum
        if other.calo_ke > 0.0:
            self.calo_ke += other.calo_ke

    @property
    def mass(self):
        """Rest mass of the particle in MeV/c^2.

        The mass is inferred from the predicted mass.

        Returns
        -------
        float
            Rest mass of the particle
        """
        if self.pid in PID_MASSES:
            return PID_MASSES[self.pid]

        return -1.0

    @mass.setter
    def mass(self, mass):
        pass

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
            if self.is_contained and self.csda_ke > 0.0:
                return self.csda_ke
            elif not self.is_contained and self.mcs_ke > 0.0:
                return self.mcs_ke
            else:
                return self.calo_ke

    @ke.setter
    def ke(self, ke):
        pass

    @property
    def momentum(self):
        """Best-guess momentum in MeV/c.

        Returns
        -------
        np.ndarray
            (3) Momentum vector
        """
        ke = self.ke
        if ke >= 0.0 and self.start_dir[0] != -np.inf and self.pid in PID_MASSES:
            mass = PID_MASSES[self.pid]
            mom = np.sqrt(ke**2 + 2 * ke * mass)
            return mom * self.start_dir

        else:
            return np.full(3, -np.inf, dtype=np.float32)

    @momentum.setter
    def momentum(self, momentum):
        pass

    @property
    def reco_ke(self):
        """Alias for `ke`, to match nomenclature in truth."""
        return self.ke

    @property
    def reco_momentum(self):
        """Alias for `momentum`, to match nomenclature in truth."""
        return self.momentum

    @property
    def reco_length(self):
        """Alias for `length`, to match nomenclature in truth."""
        return self.length

    @property
    def reco_start_dir(self):
        """Alias for `start_dir`, to match nomenclature in truth."""
        return self.start_dir

    @property
    def reco_end_dir(self):
        """Alias for `end_dir`, to match nomenclature in truth."""
        return self.end_dir


@dataclass(eq=False)
@inherit_docstring(TruthBase, ParticleBase)
class TruthParticle(Particle, ParticleBase, TruthBase):
    """Truth particle information.

    This inherits all of the attributes of :class:`Particle`, which contains
    the G4 truth information for the particle.

    Attributes
    ----------
    orig_interaction_id : int
        Unaltered index of the interaction in the original MC particle list
    orig_parent_id : int
        Unaltered index of the particle parent in the original MC particle list
    orig_group_id : int
        Unaltered index of the particle group in the original MC particle list
    orig_children_id : np.ndarray
        Unaltered list of the particle children in the original MC particle list
    children_counts : np.ndarray
        (P) Number of truth child particle of each shape
    reco_length : float
        Reconstructed length of the particle (only assigned to track objects)
    reco_start_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the start point
    reco_end_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the end point (only assigned
        to track objects)
    reco_ke : float
        Best-guess reconstructed KE of the particle
    reco_momentum : np.ndarray
        Best-guess reconstructed momentum of the particle
    """

    orig_interaction_id: int = -1
    orig_parent_id: int = -1
    orig_group_id: int = -1
    orig_children_id: np.ndarray = None
    children_counts: np.ndarray = None
    reco_length: float = -1.0
    reco_start_dir: np.ndarray = None
    reco_end_dir: np.ndarray = None
    reco_ke: float = -1.0
    reco_momentum: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = (
        ("reco_start_dir", 3),
        ("reco_end_dir", 3),
        ("reco_momentum", 3),
        *ParticleBase._fixed_length_attrs,
        *Particle._fixed_length_attrs,
    )

    # Variable-length attributes
    _var_length_attrs = (
        ("orig_children_id", np.int64),
        ("children_counts", np.int32),
        *TruthBase._var_length_attrs,
        *ParticleBase._var_length_attrs,
        *Particle._var_length_attrs,
    )

    # Attributes specifying coordinates
    _pos_attrs = (*ParticleBase._pos_attrs, *Particle._pos_attrs)

    # Attributes specifying vector components
    _vec_attrs = (
        "reco_start_dir",
        "reco_end_dir",
        "reco_momentum",
        *ParticleBase._vec_attrs,
        *Particle._vec_attrs,
    )

    # Boolean attributes
    _bool_attrs = (*TruthBase._bool_attrs, *ParticleBase._bool_attrs)

    # Attributes that must never be stored to file
    _skip_attrs = (*TruthBase._skip_attrs, *ParticleBase._skip_attrs)

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return "Truth" + super().__str__()

    @property
    def start_dir(self):
        """Converts the initial momentum to a direction vector.

        Returns
        -------
        np.ndarray
            (3) Start direction vector
        """
        if self.momentum is not None:
            norm = np.linalg.norm(self.momentum)
            if norm > 0.0 and norm != np.inf:
                return self.momentum / norm

        return np.full(3, -np.inf, dtype=np.float32)

    @start_dir.setter
    def start_dir(self, start_dir):
        pass

    @property
    def end_dir(self):
        """Converts the final momentum to a direction vector.

        Note that if a particle stops, this is unreliable as an estimate of the
        direction of the particle before it stops.

        Returns
        -------
        np.ndarray
            (3) End direction vector
        """
        if self.end_momentum is not None:
            norm = np.linalg.norm(self.end_momentum)
            if self.shape == TRACK_SHP and norm > 0.0 and norm != np.inf:
                return self.end_momentum / norm

        return np.full(3, -np.inf, dtype=np.float32)

    @end_dir.setter
    def end_dir(self, end_dir):
        pass

    @property
    def ke(self):
        """Converts the particle initial energy to a kinetic energy.

        This only works for particles with a known mass (as defined in
        `spine.utils.globals`).

        Returns
        -------
        float
            Initial kinetic energy of the particle
        """
        if self.mass > -1.0:
            return self.energy_init - self.mass

        return -1.0

    @ke.setter
    def ke(self, ke):
        pass

    @property
    def reco_ke(self):
        """Best-guess reconstructed kinetic energy in MeV.

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
            if self.is_contained and self.csda_ke > 0.0:
                return self.csda_ke
            elif not self.is_contained and self.mcs_ke > 0.0:
                return self.mcs_ke
            else:
                return self.calo_ke

    @reco_ke.setter
    def reco_ke(self, reco_ke):
        pass

    @property
    def reco_momentum(self):
        """Best-guess reconstructed momentum in MeV/c.

        Returns
        -------
        np.ndarray
            (3) Momentum vector
        """
        ke = self.reco_ke
        if ke >= 0.0 and self.reco_start_dir[0] != -np.inf and self.pid in PID_MASSES:
            mass = PID_MASSES[self.pid]
            mom = np.sqrt(ke**2 + 2 * ke * mass)
            return mom * self.reco_start_dir

        else:
            return np.full(3, -np.inf, dtype=np.float32)

    @reco_momentum.setter
    def reco_momentum(self, reco_momentum):
        pass
