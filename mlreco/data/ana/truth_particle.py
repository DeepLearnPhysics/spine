"""Module with a data class object which represents a reconstructed particle."""

from dataclasses import dataclass

import numpy as np

from mlreco.utils.decorators import inherit_docstring

from mlreco.data import Particle

from .particle_base import ParticleBase

__all__ = ['TruthParticle']


@dataclass
@inherit_docstring(ParticleBase)
class TruthParticle(Particle, ParticleBase):
    """Truth particle information.

    This inherits all of the attributes of :class:`Particle`, which contains
    the G4 truth information for the particle.

    Attributes
    ----------
    orig_id : int
        Unaltered index of the particle in the orginal MC particle list
    orig_interaction_id : int
        Unaltered index of the interaction in the original MC paricle list
    index_adapt: np.ndarray
        (N) Particle voxel indexes in the adapted cluster label tensor
    points_adapt : np.ndarray
        (N, 3) Set of voxel coordinates using adapted cluster labels
    sources_adapt : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs, adapted
    depositions_adapt : np.ndarray
        (N) Array of values for each voxel in the adapted cluster label tensor
    index_g4: np.ndarray
        (N) Particle voxel indexes in the true Geant4 energy deposition tensor
    points_g4 : np.ndarray
        (N, 3) Set of voxel coordinates of true Geant4 energy depositions
    depositions_g4 : np.ndarray
        (N) Array of true Geant4 energy depositions per voxel
    """
    orig_id: int = -1
    orig_interaction_id: int = -1
    index_adapt: np.ndarray = None
    points_adapt: np.ndarray = None
    sources_adapt: np.ndarray = None
    depositions_adapt: np.ndarray = None
    index_g4: np.ndarray = None
    points_g4: np.ndarray = None
    depositions_g4: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = {
            **Particle._fixed_length_attrs,
            **ParticleBase._fixed_length_attrs}

    # Variable-length attributes
    _var_length_attrs = {
            'index_adapt': np.int64, 'index_g4': np.int64,
            'depositions_adapt': np.float32, 'depositions_g4': np.float32,
            'points_adapt': (3, np.float32), 'points_g4': (3, np.float32),
            'sources_adapt': (2, np.int64),
            **Particle._var_length_attrs,
            **ParticleBase._var_length_attrs}

    # Attributes that should not be stored
    _skip_attrs = [
            'points_adapt', 'sources_adapt', 'depositions_adapt',
            'points_g4', 'depositions_g4', *ParticleBase._skip_attrs]

    def __str__(self):
        """Human-readable string representation of the particle object.

        Results
        -------
        str
            Basic information about the particle properties
        """
        return 'Truth' + super().__str__()
