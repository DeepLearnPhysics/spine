"""Module with a data class objects which represent output interactions."""

from typing import List
from dataclasses import dataclass, field
from warnings import warn

import numpy as np

from spine.utils.globals import PID_LABELS, PID_TAGS
from spine.utils.decorators import inherit_docstring

from spine.data.neutrino import Neutrino

from .base import RecoBase, TruthBase

__all__ = ['RecoInteraction', 'TruthInteraction']


@dataclass(eq=False)
class InteractionBase:
    """Base interaction-specific information.

    Attributes
    ----------
    particles : List[object]
        List of particles that make up the interaction
    particle_ids : np.ndarray, 
        List of Particle IDs that make up this interaction
    num_particles : int
        Number of particles that make up this interaction
    particle_counts : np.ndarray
        (P) Number of particles of each species in this interaction
    primary_particle_counts : np.ndarray
        (P) Number of primary particles of each species in this interaction
    vertex : np.ndarray
        (3) Coordinates of the interaction vertex
    is_fiducial : bool
        Whether this interaction vertex is inside the fiducial volume
    is_flash_matched : bool
        True if the interaction was matched to an optical flash
    flash_ids : np.ndarray
        (F) Indices of the optical flashes the interaction is matched to
    flash_volume_ids : np.ndarray
        (F) Indices of the optical volumes the flashes where recorded in
    flash_times : np.ndarray
        (F) Times at which the flashes occurred in microseconds
    flash_total_pe : float
        Total number of photoelectrons associated with the flash
    flash_hypo_pe : float
        Total number of photoelectrons expected to be produced by the interaction
    topology : str
        String representing the interaction topology
    """
    particles: List[object] = None
    particle_ids: np.ndarray = None
    num_particles: int = None
    particle_counts: np.ndarray = None
    primary_particle_counts: np.ndarray = None
    vertex: np.ndarray = None
    is_fiducial: bool = False
    is_flash_matched: bool = False
    flash_ids: np.ndarray = None
    flash_volume_ids: np.ndarray = None
    flash_times: np.ndarray = None
    flash_scores: np.ndarray = None
    flash_total_pe: float = -1.
    flash_hypo_pe: float = -1.
    topology: str = None

    # Fixed-length attributes
    _fixed_length_attrs = (
            ('vertex', 3), ('particle_counts', len(PID_LABELS) - 1),
            ('primary_particle_counts', len(PID_LABELS) - 1)
    )

    # Variable-length attributes as (key, dtype) pairs
    _var_length_attrs = (
            ('particles', object), ('particle_ids', np.int32),
            ('flash_ids', np.int32), ('flash_volume_ids', np.int32),
            ('flash_times', np.float32), ('flash_scores', np.float32)
    )

    # Attributes specifying coordinates
    _pos_attrs = ('vertex',)

    # Boolean attributes
    _bool_attrs = ('is_fiducial', 'is_flash_matched')

    # Attributes that must never be stored to file
    _skip_attrs = ('particles',)

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

    @num_particles.setter
    def num_particles(self, num_particles):
        pass

    @property
    def particle_counts(self):
        """Number of particles of each PID species in this interaction.

        Returns
        -------
        np.ndarray
            (P) Number of particles of each PID
        """
        counts = np.zeros(len(PID_LABELS) - 1, dtype=int)
        for part in self.particles:
            if part.pid > -1 and part.is_valid:
                counts[part.pid] += 1

        return counts

    @particle_counts.setter
    def particle_counts(self, particle_counts):
        pass

    @property
    def primary_particle_counts(self):
        """Number of primary particles of each PID species in this interaction.

        Returns
        -------
        np.ndarray
            (P) Number of primary particles of each PID
        """
        counts = np.zeros(len(PID_LABELS) - 1, dtype=int)
        for part in self.particles:
            if part.pid > -1 and part.is_primary and part.is_valid:
                counts[part.pid] += 1

        return counts

    @primary_particle_counts.setter
    def primary_particle_counts(self, primary_particle_counts):
        pass

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
        pass

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


@dataclass(eq=False)
@inherit_docstring(RecoBase, InteractionBase)
class RecoInteraction(InteractionBase, RecoBase):
    """Reconstructed interaction information."""
    
    vertex_distance: float = -1.
    shower_split_angle: float = -1.

    # Attributes that must never be stored to file
    _skip_attrs = (
            *RecoBase._skip_attrs,
            *InteractionBase._skip_attrs
    )

    # Variable-length attributes
    _var_length_attrs = (
            *RecoBase._var_length_attrs,
            *InteractionBase._var_length_attrs
    )

    # Boolean attributes
    _bool_attrs = (
            *RecoBase._bool_attrs,
            *InteractionBase._bool_attrs
    )

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return 'Reco' + super().__str__()


@dataclass(eq=False)
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
    _fixed_length_attrs = (
            ('reco_vertex', 3),
            *Neutrino._fixed_length_attrs,
            *InteractionBase._fixed_length_attrs
    )

    # Variable-length attributes
    _var_length_attrs = (
            *TruthBase._var_length_attrs,
            *InteractionBase._var_length_attrs
    )

    # Attributes specifying coordinates
    _pos_attrs = (
            'reco_vertex',
            *InteractionBase._pos_attrs,
            *Neutrino._pos_attrs
    )

    # Boolean attributes
    _bool_attrs = (
            *TruthBase._bool_attrs,
            *InteractionBase._bool_attrs
    )

    # Attributes that must never be stored to file
    _skip_attrs = (
            *TruthBase._skip_attrs,
            *InteractionBase._skip_attrs
    )

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
                if neutrino.id != self.nu_id:
                    warn("The neutrino ID as stored in the larcv.Neutrino "
                         "object does not match its index.")

        # Set the interaction vertex position
        self.vertex = neutrino.position
