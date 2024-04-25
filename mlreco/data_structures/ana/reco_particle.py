"""Module with a data class object which represents a reconstructed particle."""

from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import TRACK_SHP, SHAPE_LABELS, PID_LABELS, PID_MASSES
from mlreco.utils.numba_local import cdist
from mlreco.utils.decorators import inherit_docstring

from .particle_base import ParticleBase

__all__ = ['RecoParticle']


@dataclass
@inherit_docstring(ParticleBase)
class RecoParticle(ParticleBase):
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
                "Cannot merge particles that already have match.")

        # Concatenate the two particle array attributes together
        for attr in ['index', 'depositions', 'points',
                     'sources', 'fragment_ids']:
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

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return 'Reco' + super().__str__()

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
