"""Module with a data class objects which represent output fragments."""

from dataclasses import dataclass, field

import numpy as np

from spine.constants import SHAPE_LABELS, TRACK_SHP, ParticleShape
from spine.data.decorator import stored_property
from spine.data.field import FieldMetadata
from spine.data.larcv.particle import Particle

from .base import OutBase, RecoBase, TruthBase

__all__ = ["RecoFragment", "TruthFragment"]


@dataclass(eq=False, repr=False)
class FragmentBase(OutBase):
    """Base fragment-specific information.

    Attributes
    ----------
    particle_id : int
        Index of the particle this fragment belongs to
    interaction_id : int
        Index of the interaction this fragment belongs to
    shape : int
        Semantic type (shower (0), track (1), Michel (2), delta (3),
        low energy scatter (4)) of this particle
    is_primary : bool
        Whether this fragment was the first in the particle group
    length : float
        Length of the particle (only assigned to track objects)
    start_point : np.ndarray
        (3) Fragment start point
    end_point : np.ndarray
        (3) Fragment end point (only assigned to track objects)
    """

    # Scalar attributes
    particle_id: int = -1
    interaction_id: int = -1

    is_primary: bool = False

    length: float = field(default=np.nan, metadata=FieldMetadata(units="instance"))

    # Enumerated attributes
    shape: int = field(default=-1, metadata=FieldMetadata(enum=ParticleShape))

    # Vector attributes
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

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        shape_label = SHAPE_LABELS[self.shape]
        match = self.match_ids[0] if len(self.match_ids) > 0 else -1
        return (
            f"Fragment(ID: {self.id:<3} | ParticleShape: {shape_label:<11} "
            f"| Primary: {self.is_primary:<2} "
            f"| Size: {self.size:<5} | Match: {match:<3})"
        )


@dataclass(eq=False, repr=False)
class RecoFragment(FragmentBase, RecoBase):
    """Reconstructed fragment information.

    Attributes
    ----------
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    start_dir : np.ndarray
        (3) Fragment direction w.r.t. the start point
    end_dir : np.ndarray
        (3) Fragment direction w.r.t. the end point (only assigned
        to track objects)
    """

    # Vector attributes
    start_dir: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True),
    )
    end_dir: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True),
    )

    primary_scores: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=2, dtype=np.float32),
    )

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return "Reco" + super().__str__()


@dataclass(eq=False, repr=False)
class TruthFragment(Particle, FragmentBase, TruthBase):
    """Truth fragment information.

    This inherits all of the attributes of :class:`Particle`, which contains
    the G4 truth information for the fragment.

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
        (P) Number of truth child fragment of each shape
    reco_length : float
        Reconstructed length of the fragment (only assigned to track objects)
    reco_start_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the start point
    reco_end_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the end point (only assigned
        to track objects)
    """

    # Scalar attributes
    orig_interaction_id: int = -1
    orig_parent_id: int = -1
    orig_group_id: int = -1

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
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return "Truth" + super().__str__()

    @property
    @stored_property
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
    @stored_property
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
