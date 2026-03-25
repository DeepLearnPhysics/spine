"""Module with a data class objects which represent output interactions."""

from dataclasses import dataclass, field
from typing import List, Optional, cast
from warnings import warn

import numpy as np

from spine.data.decorator import stored_property
from spine.data.field import FieldMetadata
from spine.data.larcv.neutrino import Neutrino
from spine.utils.globals import PID_LABELS, PID_TAGS, SHOWR_SHP

from .base import OutBase, RecoBase, TruthBase
from .particle import ParticleBase, RecoParticle, TruthParticle

__all__ = ["RecoInteraction", "TruthInteraction"]


@dataclass(eq=False)
class InteractionBase(OutBase):
    """Base interaction-specific information.

    Attributes
    ----------
    particles: List[ParticleBase]
        List of particles in the interaction (defined in subclasses with specific types)
    primary_particles: List[ParticleBase]
        List of primary particles associated with the interaction
    particle_ids : np.ndarray
        List of Particle IDs that make up this interaction
    primary_particle_ids : np.ndarray
        List of primary Particle IDs associated with this interaction
    num_particles : int
        Number of particles that make up this interaction
    num_primary_particles : int
        Number of primary particles associated with this interaction
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
    flash_scores : np.ndarray
        (F) Flash matching quality scores reported for each match
    flash_total_pe : float
        Total number of photoelectrons associated with the flash
    flash_hypo_pe : float
        Total number of photoelectrons expected to be produced by the interaction
    is_crt_matched : bool
        True if any particle in the interaction was matched to a CRT hit
    crt_ids : np.ndarray
        (C) Indices of the CRT hits the interaction is matched to
    crt_times : np.ndarray
        (C) Times at which the CRT hits occurred in microseconds
    crt_scores : np.ndarray
        (C) Quality metric associated with the CRT matches
    topology : str
        String representing the interaction topology
    """

    # Scalar attributes
    is_fiducial: bool = False
    is_flash_matched: bool = False

    flash_total_pe: float = np.nan
    flash_hypo_pe: float = np.nan

    # Object list attributes
    # Note: Subclasses override this with specific List[RecoParticle/TruthParticle]
    particles: List[ParticleBase] = field(
        default_factory=list,
        metadata=FieldMetadata(skip=True),
    )

    # Vector attributes
    particle_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64),
    )

    vertex: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="instance",
        ),
    )

    flash_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64),
    )
    flash_volume_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64),
    )
    flash_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, units="us"),
    )
    flash_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32),
    )

    def __str__(self) -> str:
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        match = self.match_ids[0] if len(self.match_ids) > 0 else -1
        info = (
            f"Interaction(ID: {self.id:<3} "
            f"| Size: {self.size:<5} | Topology: {self.topology:<10} "
            f"| Match: {match:<3})"
        )
        if len(self.particles) > 0:
            info += "\n" + len(info) * "-"
            for particle in self.particles:
                info += "\n" + str(particle)

        return info

    def reset_flash_match(self) -> None:
        """Reset all the flash matching attributes."""
        self.is_flash_matched = False
        self.flash_total_pe = np.nan
        self.flash_hypo_pe = np.nan
        self.flash_ids = np.empty(0, dtype=np.int64)
        self.flash_volume_ids = np.empty(0, dtype=np.int64)
        self.flash_times = np.empty(0, dtype=np.float32)
        self.flash_scores = np.empty(0, dtype=np.float32)

    @property
    def primary_particles(self) -> List[ParticleBase]:
        """List of primary particles associated with this interaction.

        Returns
        -------
        List[ParticleBase]
            List of primary Particle objects associated with this interaction
        """
        return [part for part in self.particles if part.is_primary]

    @property
    @stored_property(dtype=np.int64)
    def primary_particle_ids(self) -> np.ndarray:
        """List of primary Particle IDs associated with this interaction.

        Returns
        -------
        np.darray
            List of primary Particle IDs associated with this interaction
        """
        return np.array([part.id for part in self.primary_particles], dtype=np.int64)

    @property
    @stored_property
    def num_particles(self) -> int:
        """Number of particles that make up this interaction.

        Returns
        -------
        int
            Number of particles that make up the interaction instance
        """
        return len(self.particle_ids)

    @property
    @stored_property
    def num_primary_particles(self) -> int:
        """Number of primary particles associated with this interaction.

        Returns
        -------
        int
            Number of primary particles associated with the interaction instance
        """
        return len(self.primary_particle_ids)

    @property
    @stored_property(dtype=np.int64, length=len(PID_LABELS) - 1)
    def particle_counts(self) -> np.ndarray:
        """Number of particles of each PID species in this interaction.

        Returns
        -------
        np.ndarray
            (P) Number of particles of each PID
        """
        counts = np.zeros(len(PID_LABELS) - 1, dtype=np.int64)
        for part in self.particles:
            if part.pid > -1 and part.is_valid:
                counts[part.pid] += 1

        return counts

    @property
    @stored_property(dtype=np.int64, length=len(PID_LABELS) - 1)
    def primary_particle_counts(self) -> np.ndarray:
        """Number of primary particles of each PID species in this interaction.

        Returns
        -------
        np.ndarray
            (P) Number of primary particles of each PID
        """
        counts = np.zeros(len(PID_LABELS) - 1, dtype=np.int64)
        for part in self.primary_particles:
            if part.pid > -1 and part.is_valid:
                counts[part.pid] += 1

        return counts

    @property
    @stored_property
    def is_crt_matched(self) -> bool:
        """Checks if any particle in the interaction was matched to a CRT hit.

        Returns
        -------
        bool
            `True` if any of the particle was matched to a CRT hit
        """
        return bool(np.any([part.is_crt_matched for part in self.particles]))

    @property
    @stored_property(dtype=np.int64)
    def crt_ids(self) -> np.ndarray:
        """Returns the list of CRT hit IDs matched to this interaction.

        Returns
        -------
        np.ndarray
            (C) List of CRT hit IDs matched to this interaction
        """
        if len(self.particles) > 0:
            return np.concatenate([part.crt_ids for part in self.particles])

        return np.empty(0, dtype=np.int64)

    @property
    @stored_property(dtype=np.float32, units="us")
    def crt_times(self) -> np.ndarray:
        """Returns the list of CRT hit times matched to this interaction.

        Returns
        -------
        np.ndarray
            (C) List of CRT hit times matched to this interaction
        """
        if len(self.particles) > 0:
            return np.concatenate([part.crt_times for part in self.particles])

        return np.empty(0, dtype=np.float32)

    @property
    @stored_property(dtype=np.float32)
    def crt_scores(self) -> np.ndarray:
        """Returns the list of quality metrics of CRT hits matched to this interaction.

        Returns
        -------
        np.ndarray
            (C) List of quality metrics of CRT hits matched to this interaction
        """
        if len(self.particles) > 0:
            return np.concatenate([part.crt_scores for part in self.particles])

        return np.empty(0, dtype=np.float32)

    @property
    @stored_property
    def topology(self) -> str:
        """String representing the interaction topology.

        Returns
        -------
        str
            String listing the number of primary particles in this interaction
        """
        topology = ""
        for i, count in enumerate(self.primary_particle_counts):
            if count > 0:
                topology += f"{count}{PID_TAGS[i]}"

        return topology

    @classmethod
    def from_particles(cls, particles: List[ParticleBase]):
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
        unique_attrs = ("is_truth", "units")
        for attr in unique_attrs:
            if hasattr(particles[0], attr):
                assert (
                    len(np.unique([getattr(p, attr) for p in particles])) < 2
                ), f"{attr} must be unique in the list of particles."

        # Attach particle list
        interaction.particles = particles
        interaction.particle_ids = np.array([p.id for p in particles])

        # Build long-form attributes
        for attr in interaction._cat_attrs:
            val_list = [getattr(p, attr) for p in particles]
            setattr(interaction, attr, np.concatenate(val_list))

        return interaction


@dataclass(eq=False)
class RecoInteraction(InteractionBase, RecoBase):
    """Reconstructed interaction information.

    Attributes
    ----------
    particles : List[RecoParticle]
        List of particles that make up the interaction
    """

    # Object list attributes
    particles: List[RecoParticle] = field(  # type: ignore[assignment]
        default_factory=lambda: [],
        metadata=FieldMetadata(skip=True),
    )

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return "Reco" + super().__str__()

    @property
    def leading_shower(self) -> Optional[RecoParticle]:
        """Leading primary shower of this interaction.

        Returns
        -------
        RecoParticle
            Primary shower with the highest kinetic energy
        """
        showers = [
            part
            for part in self.particles
            if part.is_primary and part.shape == SHOWR_SHP
        ]
        if len(showers) == 0:
            return None

        return max(showers, key=lambda x: cast(float, x.ke))


@dataclass(eq=False)
class TruthInteraction(Neutrino, InteractionBase, TruthBase):
    """Truth interaction information.

    This inherits all of the attributes of :class:`Interaction`, which contains
    the G4 truth information for the interaction.

    Attributes
    ----------
    particles : List[TruthParticle]
        List of particles that make up the interaction
    nu_id : int
        Index of the neutrino matched to this interaction
    reco_vertex : np.ndarray
        (3) Coordinates of the reconstructed interaction vertex
    """

    # Scalar attributes
    nu_id: int = -1

    # Object list attributes
    particles: List[TruthParticle] = field(  # type: ignore[assignment]
        default_factory=lambda: [],
        metadata=FieldMetadata(skip=True),
    )

    # Vector attributes
    reco_vertex: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="instance",
        ),
    )

    def __str__(self) -> str:
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return "Truth" + super().__str__()

    def attach_neutrino(self, neutrino) -> None:
        """Attach neutrino generator information to this interaction.

        Parameters
        ----------
        neutrino : Neutrino
            Neutrino to fetch the attributes from
        """
        # Transfer all the neutrino attributes
        for attr, val in neutrino.as_dict().items():
            if attr != "id":
                setattr(self, attr, val)
            else:
                if neutrino.id != self.nu_id:
                    warn(
                        "The neutrino ID as stored in the larcv.Neutrino "
                        "object does not match its index."
                    )

        # Set the interaction vertex position
        self.vertex = neutrino.position
