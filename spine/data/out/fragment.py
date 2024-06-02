"""Module with a data class objects which represent output fragments."""

from dataclasses import dataclass

import numpy as np

from spine.utils.globals import SHAPE_LABELS
from spine.utils.decorators import inherit_docstring

from spine.data.particle import Particle

from .base import RecoBase, TruthBase

__all__ = ['RecoFragment', 'TruthFragment']


@dataclass
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
        (3) Fragment direction estimate w.r.t. the start point
    end_dir : np.ndarray
        (3) Fragment direction estimate w.r.t. the end point (only assigned
        to track objects)
    """
    particle_id: int = -1
    interaction_id: int = -1
    shape: int = -1
    is_primary: bool = False
    length: float = -1.
    start_point: np.ndarray = None
    end_point: np.ndarray = None
    start_dir: np.ndarray = None
    end_dir: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = {'start_point': 3, 'end_point': 3, 'start_dir': 3,
                           'end_dir': 3}

    # Attributes specifying coordinates
    _pos_attrs = ['start_point', 'end_point']

    # Attributes specifying vector components
    _vec_attrs = ['start_dir', 'end_dir']

    # Enumerated attributes
    _enum_attrs = {
            'shape': {v : k for k, v in SHAPE_LABELS.items()}
    }

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

@dataclass
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
    _fixed_length_attrs = {
            'primary_scores': 2, 
            **FragmentBase._fixed_length_attrs}

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return 'Reco' + super().__str__()


@dataclass
@inherit_docstring(TruthBase, FragmentBase)
class TruthFragment(Particle, FragmentBase, TruthBase):
    """Truth fragment information.

    This inherits all of the attributes of :class:`Particle`, which contains
    the G4 truth information for the fragment.
    """

    # Fixed-length attributes
    _fixed_length_attrs = {
            **FragmentBase._fixed_length_attrs, **Particle._fixed_length_attrs
    }

    # Variable-length attributes
    _var_length_attrs = {
            **TruthBase._var_length_attrs, **Particle._var_length_attrs
    }

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return 'Truth' + super().__str__()
