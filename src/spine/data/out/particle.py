"""Module with a data class objects which represent output particles."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from spine.data.decorator import stored_alias, stored_property
from spine.data.field import FieldMetadata
from spine.data.larcv.particle import Particle
from spine.utils.globals import (
    PID_LABELS,
    PID_MASSES,
    PID_TO_PDG,
    SHAPE_LABELS,
    SHOWR_SHP,
    TRACK_SHP,
)

from .base import OutBase, RecoBase, TruthBase
from .fragment import RecoFragment, TruthFragment

__all__ = ["RecoParticle", "TruthParticle"]


@dataclass(eq=False, repr=False)
class ParticleBase(OutBase):
    """Base particle-specific information.

    Attributes
    ----------
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
    is_primary : bool
        Whether this particle was the first in the particle group
    length : float
        Length of the particle (only assigned to track objects)
    start_point : np.ndarray
        (3) Particle start point
    end_point : np.ndarray
        (3) Particle end point (only assigned to track objects)
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

    # Scalar attributes
    interaction_id: int = -1
    chi2_pid: int = -1

    is_primary: bool = False
    is_crt_matched: bool = False
    is_valid: bool = True

    length: float = field(default=np.nan, metadata=FieldMetadata(units="instance"))
    calo_ke: float = field(default=np.nan, metadata=FieldMetadata(units="MeV"))
    csda_ke: float = field(default=np.nan, metadata=FieldMetadata(units="MeV"))
    mcs_ke: float = field(default=np.nan, metadata=FieldMetadata(units="MeV"))

    # Enumerated attributes
    shape: int = field(default=-1, metadata=FieldMetadata(enum=SHAPE_LABELS))
    pid: int = field(default=-1, metadata=FieldMetadata(enum=PID_LABELS))

    # Vector attributes
    fragment_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32, cat=True, units="instance"),
    )

    start_point: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="instance",
        ),
    )
    end_point: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="instance",
        ),
    )

    chi2_per_pid: np.ndarray = field(
        default_factory=lambda: np.full(len(PID_LABELS) - 1, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=len(PID_LABELS) - 1, dtype=np.float32),
    )
    csda_ke_per_pid: np.ndarray = field(
        default_factory=lambda: np.full(len(PID_LABELS) - 1, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=len(PID_LABELS) - 1, dtype=np.float32, units="MeV"
        ),
    )
    mcs_ke_per_pid: np.ndarray = field(
        default_factory=lambda: np.full(len(PID_LABELS) - 1, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=len(PID_LABELS) - 1, dtype=np.float32, units="MeV"
        ),
    )

    crt_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32),
    )
    crt_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, units="us"),
    )
    crt_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32),
    )

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

    def reset_crt_match(self) -> None:
        """Reset all the CRT hit matching attributes."""
        self.is_crt_matched = False
        self.crt_ids = np.empty(0, dtype=np.int32)
        self.crt_times = np.empty(0, dtype=np.float32)
        self.crt_scores = np.empty(0, dtype=np.float32)

    @property
    @stored_property
    def num_fragments(self) -> int:
        """Number of fragments that make up this particle.

        Returns
        -------
        int
            Number of fragments that make up the particle instance
        """
        return len(self.fragment_ids)


@dataclass(eq=False, repr=False)
class RecoParticle(ParticleBase, RecoBase):
    """Reconstructed particle information.

    Attributes
    ----------
    fragments : List[RecoFragment]
        List of fragments that make up this particle
    pdg_code : int
        PDG code corresponding to the PID number
    mass : float
        Rest mass of the particle in MeV/c^2
    ke : float
        Best guess kinetic energy of the particle in MeV
    momentum : np.ndarray
        3-momentum of the particle at the production point in MeV/c
    start_dir : np.ndarray
        (3) Particle direction w.r.t. the start point
    end_dir : np.ndarray
        (3) Particle direction w.r.t. the end point (only assigned
        to track objects)
    pid_scores : np.ndarray
        (P) Array of softmax scores associated with each particle class
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    ppn_ids : np.ndarray
        (M) List of indexes of PPN points associated with this particle
    ppn_points : np.ndarray
        (M, 3) List of PPN points tagged to this particle
    start_dedx : float
        dE/dx around a user-defined neighborhood of the start point in MeV/cm
    end_dedx : float
        dE/dx around a user-defined neighborhood of the end point in MeV/cm
    vertex_distance : float
        Set-to-point distance between all particle points and the parent
        interaction vertex position in cm
    start_straightness : float
        Explained variance ratio of the beginning of the particle
    directional_spread : float
        Estimate of the angular spread of the particle (cosine spread)
    axial_spread : float
        Pearson correlation coefficient of the axial profile of the particle
        w.r.t. to the distance from its start point
    """

    # Scalar attributes
    start_dedx: float = field(default=np.nan, metadata=FieldMetadata(units="MeV/cm"))
    end_dedx: float = field(default=np.nan, metadata=FieldMetadata(units="MeV/cm"))
    vertex_distance: float = field(
        default=np.nan, metadata=FieldMetadata(units="instance")
    )
    start_straightness: float = np.nan
    directional_spread: float = np.nan
    axial_spread: float = np.nan

    # Object list attributes
    fragments: list[RecoFragment] = field(
        default_factory=lambda: [],
        metadata=FieldMetadata(skip=True, cat=True),
    )

    # Vector attributes
    start_dir: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True),
    )
    end_dir: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True),
    )

    pid_scores: np.ndarray = field(
        default_factory=lambda: np.full(len(PID_LABELS) - 1, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=len(PID_LABELS) - 1, dtype=np.float32),
    )
    primary_scores: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=2, dtype=np.float32),
    )

    ppn_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32),
    )
    ppn_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        metadata=FieldMetadata(
            dtype=np.float32,
            position=True,
            skip=True,
            units="instance",
        ),
    )

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
        if self.shape not in (SHOWR_SHP, TRACK_SHP) or other.shape != TRACK_SHP:
            raise ValueError(
                "Can only merge two track particles or a track into a shower."
            )

        # Check that neither particle has yet been matched
        if self.is_matched or other.is_matched:
            raise ValueError("Cannot merge particles that already have matches.")

        # Concatenate the two particle long-form attributes together
        for attr in self._cat_attrs:
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)
            # Handle lists separately from numpy arrays
            if isinstance(self_val, list):
                val = self_val + other_val
            else:
                val = np.concatenate([self_val, other_val])
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
        if not np.isnan(self.calo_ke) and not np.isnan(other.calo_ke):
            self.calo_ke += other.calo_ke

    @property
    @stored_property
    def pdg_code(self) -> int:
        """Translates the enumerated particle type to a sign-less PDG code.

        Returns
        -------
        int
            Reconstructed sign-less PDG code
        """
        return PID_TO_PDG[self.pid]

    @property
    @stored_property(units="MeV/c^2")
    def mass(self) -> float:
        """Rest mass of the particle in MeV/c^2.

        The mass is inferred from the predicted mass.

        Returns
        -------
        float
            Rest mass of the particle
        """
        if self.pid in PID_MASSES:
            return PID_MASSES[self.pid]

        return np.nan

    @property
    @stored_property(units="MeV")
    def ke(self) -> float:
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

    @property
    @stored_property(length=3, dtype=np.float32, units="MeV/c")
    def momentum(self) -> np.ndarray:
        """Best-guess momentum in MeV/c.

        Returns
        -------
        np.ndarray
            (3) Momentum vector
        """
        ke = self.ke
        if (
            not np.isnan(ke)
            and not np.any(np.isnan(self.start_dir))
            and self.pid in PID_MASSES
        ):
            mass = PID_MASSES[self.pid]
            mom = np.sqrt(ke**2 + 2 * ke * mass)
            return mom * self.start_dir

        else:
            return np.full(3, np.nan, dtype=np.float32)

    @property
    @stored_property(units="MeV/c")
    def p(self) -> float:
        """Computes the magnitude of the initial momentum.

        Returns
        -------
        float
            Norm of the initial momentum vector
        """
        return float(np.linalg.norm(self.momentum))

    @property
    @stored_alias("ke")
    def reco_ke(self) -> float:
        """Alias for `ke`, to match nomenclature in truth."""
        return self.ke

    @property
    @stored_alias("momentum")
    def reco_momentum(self) -> np.ndarray:
        """Alias for `momentum`, to match nomenclature in truth."""
        return self.momentum

    @property
    @stored_alias("length")
    def reco_length(self) -> float:
        """Alias for `length`, to match nomenclature in truth."""
        return self.length

    @property
    @stored_alias("start_dir")
    def reco_start_dir(self) -> np.ndarray:
        """Alias for `start_dir`, to match nomenclature in truth."""
        return self.start_dir

    @property
    @stored_alias("end_dir")
    def reco_end_dir(self) -> np.ndarray:
        """Alias for `end_dir`, to match nomenclature in truth."""
        return self.end_dir


@dataclass(eq=False, repr=False)
class TruthParticle(Particle, ParticleBase, TruthBase):
    """Truth particle information.

    This inherits all of the attributes of :class:`Particle`, which contains
    the G4 truth information for the particle.

    Attributes
    ----------
    fragments : List[TruthFragment]
        List of fragments that make up this particle
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
    pdg_code : int
        PDG code corresponding to the PID number
    mass : float
        Rest mass of the particle in MeV/c^2
    ke : float
        Kinetic energy of the particle in MeV
    momentum : np.ndarray
        3-momentum of the particle at the production point in MeV/c
    start_dir : np.ndarray
        (3) Particle direction w.r.t. the start point
    end_dir : np.ndarray
        (3) Particle direction w.r.t. the end point (only assigned
        to track objects)
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

    # Scalar attributes
    orig_interaction_id: int = -1
    orig_parent_id: int = -1
    orig_group_id: int = -1

    reco_length: float = np.nan

    # Object list attributes
    fragments: list[TruthFragment] = field(
        default_factory=lambda: [],
        metadata=FieldMetadata(skip=True),
    )

    # Vector attributes
    orig_children_id: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32),
    )
    children_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32),
    )

    reco_start_dir: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True),
    )
    reco_end_dir: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True),
    )

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return "Truth" + super().__str__()

    @property
    @stored_property(length=3, dtype=np.float32)
    def start_dir(self) -> np.ndarray:
        """Converts the initial momentum to a direction vector.

        Returns
        -------
        np.ndarray
            (3) Start direction vector
        """
        norm = np.linalg.norm(self.momentum)
        if norm > 0.0:
            return self.momentum / norm

        return np.full(3, np.nan, dtype=np.float32)

    @property
    @stored_property(length=3, dtype=np.float32)
    def end_dir(self) -> np.ndarray:
        """Converts the final momentum to a direction vector.

        Note that if a particle stops, this is unreliable as an estimate of the
        direction of the particle before it stops.

        Returns
        -------
        np.ndarray
            (3) End direction vector
        """
        if self.shape == TRACK_SHP:
            norm = np.linalg.norm(self.end_momentum)
            if norm > 0.0:
                return self.end_momentum / norm

        return np.full(3, np.nan, dtype=np.float32)

    @property
    @stored_property(units="MeV/c^2")
    def ke(self) -> float:
        """Converts the particle initial energy to a kinetic energy.

        This only works for particles with a known mass (as defined in
        `spine.utils.globals`).

        Returns
        -------
        float
            Initial kinetic energy of the particle
        """
        if not np.isnan(self.mass):
            return self.energy_init - self.mass

        return np.nan

    @property
    @stored_property(units="MeV")
    def reco_ke(self) -> float:
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
            if self.is_contained and not np.isnan(self.csda_ke):
                return self.csda_ke
            elif not self.is_contained and not np.isnan(self.mcs_ke):
                return self.mcs_ke
            else:
                return self.calo_ke

    @property
    @stored_property(length=3, dtype=np.float32, units="MeV/c")
    def reco_momentum(self) -> np.ndarray:
        """Best-guess reconstructed momentum in MeV/c.

        Returns
        -------
        np.ndarray
            (3) Momentum vector
        """
        ke = self.reco_ke
        if not np.isnan(ke) and self.pid in PID_MASSES:
            mass = PID_MASSES[self.pid]
            mom = np.sqrt(ke**2 + 2 * ke * mass)
            return mom * self.reco_start_dir

        else:
            return np.full(3, np.nan, dtype=np.float32)
