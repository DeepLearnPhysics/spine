"""Module with a data class objects which represent output particles."""

from typing import List
from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import SHAPE_LABELS, PID_LABELS
from mlreco.utils.decorators import inherit_docstring

from mlreco.data.particle import Particle

from .base import RecoBase, TruthBase

__all__ = ['RecoParticle', 'TruthParticle']


@dataclass
class ParticleBase:
    """Base particle-specific information.

    Attributes
    ----------
    fragments : List[object]
        List of fragments that make up the interaction
    fragment_ids : np.ndarray, 
        List of Fragment IDs that make up this particle
    interaction_id : int
        Index of the interaction this particle belongs to
    shape : int
        Semantic type (shower (0), track (1), Michel (2), delta (3),
        low energy scatter (4)) of this particle
    pid : int
        Particle spcies (Photon (0), Electron (1), Muon (2), Charged Pion (3),
        Proton (4)) of this particle
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
        (3) Particle direction estimate w.r.t. the start point
    end_dir : np.ndarray
        (3) Particle direction estimate w.r.t. the end point (only assigned
        to track objects)
    is_valid : bool
        Whether this particle counts towards an interaction topology. This
        may be False if a particle is below some defined energy threshold.
    """
    fragments: List[object] = None
    fragment_ids: np.ndarray = None
    interaction_id: int = -1
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
    is_valid: bool = False

    # Fixed-length attributes
    _fixed_length_attrs = {
            'start_point': 3, 'end_point': 3, 'start_dir': 3, 'end_dir': 3,
            'momentum': 3
    }

    # Attributes specifying coordinates
    _pos_attrs = ['start_point', 'end_point']

    # Attributes specifying vector components
    _vec_attrs = ['start_dir', 'end_dir', 'momentum']

    # Enumerated attributes
    _enum_attrs = {
            'shape': {v : k for k, v in SHAPE_LABELS.items()},
            'pid': {v : k for k, v in PID_LABELS.items()}
    }

    # Attributes that should not be stored
    _skip_attrs = ['fragments']

    @property
    def num_fragments(self):
        """Number of fragments that make up this particle.

        Returns
        -------
        int
            Number of fragments that make up the particle instance
        """
        return len(self.fragment_ids)

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        shape_label = SHAPE_LABELS[self.shape]
        match = self.match[0] if len(self.match) > 0 else -1
        return (f"Particle(ID: {self.id:<3} | Shape: {shape_label:<11} "
                f"| Primary: {self.is_primary:<2} "
                f"| Particle ID: {self.particle_id} "
                f"| Interaction ID: {self.interaction_id:<2} "
                f"| Size: {self.size:<5} | Match: {match:<3})")

@dataclass
@inherit_docstring(RecoBase, ParticleBase)
class RecoParticle(ParticleBase, RecoBase):
    """Reconstructed particle information.

    Attributes
    ----------
    pid_scores : np.ndarray
        (P) Array of softmax scores associated with each of particle class
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    calo_ke : float
        Kinetic energy reconstructed from the energy depositions alone
    csda_ke : float
        Kinetic energy reconstructed from the particle range
    mcs_ke : float
        Kinetic energy reconstructed using the MCS method
    """
    pid_scores: np.ndarray = None
    primary_scores: np.ndarray = None
    calo_ke: float = -1.
    csda_ke: float = -1.
    mcs_ke: float = -1.

    # Fixed-length attributes
    _fixed_length_attrs = {
            'pid_scores': len(PID_LABELS) - 1,
            'primary_scores': 2, 
            **ParticleBase._fixed_length_attrs}

    # Attributes that should not be stored
    _skip_attrs = [*RecoBase._skip_attrs, *ParticleBase._skip_attrs]

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return 'Reco' + super().__str__()

    def merge(self, other):
        """Merge another particle instance into this one.

        This method can only merge two track objects with well defined start
        and end points.

        Parameters
        ----------
        other : RecoParticle
            Other reconstructed particle to merge into this one
        """
        # Check that both particles being merged are tracks
        assert self.shape == TRACK_SHP and other.shape == TRACK_SHP, (
                "Can only merge two track particles.")

        # Check that neither particle has yet been matches
        assert not self.is_matched and not other.is_matched, (
                "Cannot merge particles that already have matches.")

        # Concatenate the two particle long-form attributes together
        for attr in self._cat_attrs:
            val = np.concatenate([getattr(self, attr), getattr(other, attr)])
            setattr(self, attr, val)

        # Select end points and end directions appropriately
        points_i = np.vstack([self.start_point, self.end_point])
        points_j = np.vstack([other.start_point, other.end_point])
        dirs_i = np.vstack([self.start_dir, self.end_dir])
        dirs_j = np.vstack([other.start_dir, other.end_dir])

        dists = cdist(points_i, points_j)
        max_index = np.argmax(dists)
        max_i, max_j = max_index//2, max_index%2

        self.start_point = points_i[max_i]
        self.end_point = points_j[max_j]
        self.start_dir = dirs_i[max_i]
        self.end_dir = dirs_j[max_j]

        # If one of the two particles is a primary, the new one is
        if other.primary_scores[-1] > self.primary_scores[-1]:
            self.primary_scores = other.primary_scores

        # For PID, pick the most confident prediction (could be better...)
        if np.max(other.pid_scores) > np.max(self.pid_scores):
            self.pid_scores = other.pid_scores

    '''
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
    '''


@dataclass
@inherit_docstring(TruthBase, ParticleBase)
class TruthParticle(Particle, ParticleBase, TruthBase):
    """Truth particle information.

    This inherits all of the attributes of :class:`Particle`, which contains
    the G4 truth information for the particle.

    Attributes
    ----------
    orig_interaction_id : int
        Unaltered index of the interaction in the original MC paricle list
    """
    orig_interaction_id: int = -1

    # Fixed-length attributes
    _fixed_length_attrs = {
            **ParticleBase._fixed_length_attrs, **Particle._fixed_length_attrs
    }

    # Variable-length attributes
    _var_length_attrs = {
            **TruthBase._var_length_attrs, **Particle._var_length_attrs
    }

    # Attributes that should not be stored
    _skip_attrs = [*TruthBase._skip_attrs, *ParticleBase._skip_attrs]

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return 'Truth' + super().__str__()
