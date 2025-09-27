"""Module with a data class objects which represent output fragments."""

from dataclasses import dataclass

import numpy as np

from spine.data.particle import Particle
from spine.utils.docstring import inherit_docstring
from spine.utils.globals import SHAPE_LABELS, TRACK_SHP

from .base import RecoBase, TruthBase

__all__ = ["RecoFragment", "TruthFragment"]


@dataclass(eq=False)
class FragmentBase:
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
    start_dir : np.ndarray
        (3) Fragment direction w.r.t. the start point
    end_dir : np.ndarray
        (3) Fragment direction w.r.t. the end point (only assigned
        to track objects)
    """

    particle_id: int = -1
    interaction_id: int = -1
    shape: int = -1
    is_primary: bool = False
    length: float = -1.0
    start_point: np.ndarray = None
    end_point: np.ndarray = None
    start_dir: np.ndarray = None
    end_dir: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = (
        ("start_point", 3),
        ("end_point", 3),
        ("start_dir", 3),
        ("end_dir", 3),
    )

    # Attributes specifying coordinates
    _pos_attrs = ("start_point", "end_point")

    # Attributes specifying vector components
    _vec_attrs = ("start_dir", "end_dir")

    # Boolean attributes
    _bool_attrs = ("is_primary",)

    # Enumerated attributes
    _enum_attrs = (("shape", tuple((v, k) for k, v in SHAPE_LABELS.items())),)

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
            f"Fragment(ID: {self.id:<3} | Shape: {shape_label:<11} "
            f"| Primary: {self.is_primary:<2} "
            f"| Size: {self.size:<5} | Match: {match:<3})"
        )


@dataclass(eq=False)
@inherit_docstring(RecoBase, FragmentBase)
class RecoFragment(FragmentBase, RecoBase):
    """Reconstructed fragment information.

    Attributes
    ----------
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    """

    primary_scores: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = (("primary_scores", 2), *FragmentBase._fixed_length_attrs)

    # Boolean attributes
    _bool_attrs = (*RecoBase._bool_attrs, *FragmentBase._bool_attrs)

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return "Reco" + super().__str__()


@dataclass(eq=False)
@inherit_docstring(TruthBase, FragmentBase)
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

    orig_interaction_id: int = -1
    orig_parent_id: int = -1
    orig_group_id: int = -1
    orig_children_id: np.ndarray = None
    children_counts: np.ndarray = None
    reco_length: float = -1.0
    reco_start_dir: np.ndarray = None
    reco_end_dir: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = (
        ("reco_start_dir", 3),
        ("reco_end_dir", 3),
        *FragmentBase._fixed_length_attrs,
        *Particle._fixed_length_attrs,
    )

    # Variable-length attributes
    _var_length_attrs = (
        ("orig_children_id", np.int64),
        ("children_counts", np.int32),
        *TruthBase._var_length_attrs,
        *Particle._var_length_attrs,
    )

    # Attributes specifying coordinates
    _pos_attrs = (*FragmentBase._pos_attrs, *Particle._pos_attrs)

    # Attributes specifying vector components
    _vec_attrs = (
        "reco_start_dir",
        "reco_end_dir",
        *FragmentBase._vec_attrs,
        *Particle._vec_attrs,
    )

    # Boolean attributes
    _bool_attrs = (*TruthBase._bool_attrs, *FragmentBase._bool_attrs)

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
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
