"""Module with a data class objects which represent output interactions."""

from typing import List
from dataclasses import dataclass, asdict

import numpy as np

from mlreco.utils.decorators import inherit_docstring

from mlreco.data.neutrino import Neutrino

from .base import RecoBase, TruthBase

__all__ = ['RecoInteraction', 'TruthInteraction']


@dataclass
class InteractionBase:
    """Base interaction-specific information.

    Attributes
    ----------
    particles : List[object]
        List of particles that make up the interaction
    particle_ids : np.ndarray, 
        List of Particle IDs that make up this interaction
    vertex : np.ndarray
        (3) Coordinates of the interaction vertex
    is_fiducial : bool
        Whether this interaction vertex is inside the fiducial volume
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
    """
    particles: List[object] = None
    particle_ids: np.ndarray = None
    vertex: np.ndarray = None
    is_fiducial: bool = False
    is_flash_matched: bool = False
    flash_id: int = -1
    flash_time: float = -1.
    flash_total_pe: float = -1.
    flash_hypo_pe: float = -1.

    # Fixed-length attributes
    _fixed_length_attrs = {'vertex': 3}

    # Attributes specifying coordinates
    _pos_attrs = ['vertex']

    # Attributes specifying vector components
    _vec_attrs = ['vertex']

    # Attributes that should not be stored
    _skip_attrs = ['particles']

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        match = self.match[0] if len(self.match) > 0 else -1
        return (f"Interaction(ID: {self.id:<3} "
                f"| Size: {self.size:<5} | Match: {match:<3})")

    @property
    def num_particles(self):
        """Number of particles that make up this interaction.

        Returns
        -------
        int
            Number of particles that make up the interaction instance
        """
        return len(self.particle_ids)

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
        for attr in cls._cat_attrs:
            val_list = [getattr(p, attr) for p in particles]
            setattr(interaction, attr, np.concatenate(val_list))

        return interaction


@dataclass
@inherit_docstring(RecoBase, InteractionBase)
class RecoInteraction(InteractionBase, RecoBase):
    """Reconstructed interaction information."""

    # Attributes that should not be stored
    _skip_attrs = [*RecoBase._skip_attrs, *InteractionBase._skip_attrs]

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return 'Reco' + super().__str__()


@dataclass
@inherit_docstring(TruthBase, InteractionBase)
class TruthInteraction(Neutrino, InteractionBase, TruthBase):
    """Truth interaction information.

    This inherits all of the attributes of :class:`Interaction`, which contains
    the G4 truth information for the interaction.

    Attributes
    ----------
    nu_id : int
        Index of the neutrino matched to this interaction
    """
    nu_id: int = -1

    # Attributes that should not be stored
    _skip_attrs = [*TruthBase._skip_attrs, *InteractionBase._skip_attrs]

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return 'Truth' + super().__str__()

    def attach_neutrino(self, neutrino):
        """Attach neutrino generator information to this interaction.

        Parameters
        ----------
        neutrino : Neutrino
            Neutrino to fetch the attributes from
        """
        for attr, val in asdict(neutrino).items():
            if attr != 'id':
                setattr(self, attr, val)
            else:
                self.nu_id = val
