"""Module with a data class objects which represent output interactions."""

from typing import List
from dataclasses import dataclass, field

import numpy as np

from spine.utils.globals import PID_LABELS, PID_TAGS
from spine.utils.decorators import inherit_docstring

from spine.data.neutrino import Neutrino

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
    is_flash_matched : bool
        True if the interaction was matched to an optical flash
    flash_id : int
        Index of the optical flash the interaction was matched to
    flash_time : float
        Time at which the flash occurred in nanoseconds
    flash_total_pe : float
        Total number of photoelectrons associated with the flash
    flash_hypo_pe : float
        Total number of photoelectrons expected to be produced by the interaction
    topology : str
        String representing the interaction topology
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
    topology: str = ''

    # Private derived attributes
    _topology: str = field(init=False, repr=False)

    # Fixed-length attributes
    _fixed_length_attrs = {'vertex': 3}

    # Variable-length attributes as (key, dtype) pairs
    _var_length_attrs = {
            'particles': object, 'particle_ids': np.int32
    }

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
        match = self.match_ids[0] if len(self.match_ids) > 0 else -1
        info = (f"Interaction(ID: {self.id:<3} "
                f"| Size: {self.size:<5} | Topology: {self.topology:<10} "
                f"| Match: {match:<3})")
        if len (self.particles):
            info += '\n' + len(info) * '-'
            for particle in self.particles:
                info += '\n' + str(particle)

        return info

    @property
    def num_particles(self):
        """Number of particles that make up this interaction.

        Returns
        -------
        int
            Number of particles that make up the interaction instance
        """
        return len(self.particle_ids)

    @property
    def particle_counts(self):
        """Number of particles of each shape in this interaction.

        Returns
        -------
        np.ndarray
            (C) Number of particles of each class
        """
        counts = np.zeros(len(PID_LABELS) - 1, dtype=int)
        for part in self.particles:
            if part.pid > -1 and part.is_valid:
                counts[part.pid] += 1

        return counts

    @property
    def primary_particle_counts(self):
        """Number of primary particles of each shape in this interaction.

        Returns
        -------
        np.ndarray
            (C) Number of primary particles of each class
        """
        counts = np.zeros(len(PID_LABELS) - 1, dtype=int)
        for part in self.particles:
            if part.pid > -1 and part.is_primary and part.is_valid:
                counts[part.pid] += 1

        return counts

    @property
    def topology(self):
        """String representing the interaction topology.

        Returns
        -------
        str
            String listing the number of primary particles in this interaction
        """
        topology = ''
        for i, count in enumerate(self.primary_particle_counts):
            if count > 0:
                topology += f'{count}{PID_TAGS[i]}'

        return topology

    @topology.setter
    def topology(self, topology):
        self._topology = topology

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
        interaction.particle_ids = np.array([p.id for p in particles])

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

    # Variable-length attributes
    _var_length_attrs = {
            **RecoBase._var_length_attrs, **InteractionBase._var_length_attrs
    }

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
    reco_vertex : np.ndarray
        (3) Coordinates of the reconstructed interaction vertex
    """
    nu_id: int = -1
    reco_vertex: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = {
            **Neutrino._fixed_length_attrs,
            **InteractionBase._fixed_length_attrs,
            'reco_vertex': 3, 
    }

    # Variable-length attributes
    _var_length_attrs = {
            **TruthBase._var_length_attrs,
            **InteractionBase._var_length_attrs
    }

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
        # Transfer all the neutrino attributes
        for attr, val in neutrino.as_dict().items():
            if attr != 'id':
                setattr(self, attr, val)
            else:
                self.nu_id = val

        # Set the interaction vertex position
        self.vertex = neutrino.position
