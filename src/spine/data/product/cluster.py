"""Structured voxel-cluster and particle-label data products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spine.utils.conditional import torch

from ..base import DataBase
from ..field import FieldMetadata
from .base import DataProduct, TensorSchema

__all__ = ["ParticleLabel", "ClusterLabelData"]


@dataclass(eq=False, repr=False)
class ParticleLabel(DataBase):
    """Compact named particle information used by cluster labels.

    Each record represents one truth particle. Integer relationships such as
    ``ancestor`` are event-local indexes into the surrounding particle table;
    physical identifiers such as ``particle`` retain their source values.

    Attributes
    ----------
    particle : int
        Source particle identifier.
    group : int
        Particle-group identifier.
    ancestor : int
        Event-local index of the ancestor particle.
    interaction : int
        Interaction identifier.
    nu : int
        Neutrino identifier, or ``-1`` for non-neutrino interactions.
    pid : int
        Particle-type label.
    group_primary : int
        Whether the particle is primary within its particle group.
    interaction_primary : int
        Whether the particle is primary within its interaction.
    vertex : numpy.ndarray
        Ancestor vertex in image coordinates.
    momentum : float
        Initial momentum magnitude.
    energy_init : float
        Initial total energy.
    shape : int
        Semantic-shape label.
    """

    particle: int = -1
    group: int = -1
    ancestor: int = -1
    interaction: int = -1
    nu: int = -1
    pid: int = -1
    group_primary: int = -1
    interaction_primary: int = -1
    vertex: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, position=True),
    )
    momentum: float = np.nan
    energy_init: float = np.nan
    shape: int = -1


def _field_name(name: str) -> str:
    """Normalize accepted field-name aliases."""
    aliases = {
        "part": "particle",
        "ancst": "ancestor",
        "inter": "interaction",
        "type": "pid",
        "pgroup": "group_primary",
        "pinter": "interaction_primary",
        "p": "momentum",
    }
    return aliases.get(name, name)


class _ClusterLabelFieldAccessor:
    """Shared named-field interface for event and batch cluster labels.

    Subclasses implement :meth:`voxel_field`; the properties below provide a
    discoverable, statically named interface for fields used throughout the
    reconstruction chain.
    """

    def voxel_field(self, name: str) -> Any:
        """Return a named field expanded to the voxel rows.

        Parameters
        ----------
        name : str
            Compact voxel field, particle field, accepted alias or virtual
            ancestor field.
        """
        raise NotImplementedError

    @property
    def values(self) -> Any:
        """Return the value associated with each voxel."""
        return self.voxel_field("value")

    @property
    def cluster_ids(self) -> Any:
        """Return the cluster identifier associated with each voxel."""
        return self.voxel_field("cluster")

    @property
    def particle_indexes(self) -> Any:
        """Return event-local particle-table indexes for each voxel."""
        return self.voxel_field("particle_index")

    @property
    def particle_ids(self) -> Any:
        """Return source particle identifiers expanded to the voxels."""
        return self.voxel_field("particle")

    @property
    def group_ids(self) -> Any:
        """Return particle-group identifiers expanded to the voxels."""
        return self.voxel_field("group")

    @property
    def ancestor_indexes(self) -> Any:
        """Return event-local ancestor table indexes expanded to the voxels."""
        return self.voxel_field("ancestor")

    @property
    def interaction_ids(self) -> Any:
        """Return interaction identifiers expanded to the voxels."""
        return self.voxel_field("interaction")

    @property
    def neutrino_ids(self) -> Any:
        """Return neutrino identifiers expanded to the voxels."""
        return self.voxel_field("nu")

    @property
    def pids(self) -> Any:
        """Return particle-type labels expanded to the voxels."""
        return self.voxel_field("pid")

    @property
    def group_primaries(self) -> Any:
        """Return group-primary labels expanded to the voxels."""
        return self.voxel_field("group_primary")

    @property
    def interaction_primaries(self) -> Any:
        """Return interaction-primary labels expanded to the voxels."""
        return self.voxel_field("interaction_primary")

    @property
    def vertices(self) -> Any:
        """Return ancestor vertices expanded to the voxels."""
        return self.voxel_field("vertex")

    @property
    def momenta(self) -> Any:
        """Return initial momentum magnitudes expanded to the voxels."""
        return self.voxel_field("momentum")

    @property
    def energies(self) -> Any:
        """Return initial energies expanded to the voxels."""
        return self.voxel_field("energy_init")

    @property
    def shapes(self) -> Any:
        """Return semantic-shape labels expanded to the voxels."""
        return self.voxel_field("shape")

    @property
    def ancestor_pids(self) -> Any:
        """Return ancestor particle-type labels expanded to the voxels."""
        return self.voxel_field("ancestor_pid")

    @property
    def ancestor_momenta(self) -> Any:
        """Return ancestor momentum magnitudes expanded to the voxels."""
        return self.voxel_field("ancestor_momentum")


@dataclass(eq=False, init=False)
class ClusterLabelData(_ClusterLabelFieldAccessor, DataProduct):
    """Cluster labels and optional particle information for one event.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Compact voxel table containing batch-free coordinates, value, cluster
        index and, when available, particle index.
    particles : dict[str, numpy.ndarray or torch.Tensor], optional
        Named particle-level fields. Every field has one row per particle.
    meta : object, optional
        Image metadata associated with the voxel coordinates.
    remove_duplicates : bool, default True
        Whether overlapping voxel rows should be merged during collation.
    sum_cols : numpy.ndarray, optional
        Compact feature columns summed during duplicate merging.
    """

    data: Any
    particles: dict[str, Any] | None = None
    meta: Any | None = None
    remove_duplicates: bool = True
    sum_cols: np.ndarray | None = None

    product_type = "cluster_label"

    # Compact event-level layout: x, y, z, value, cluster, particle.
    coord_cols = np.arange(3, dtype=np.int64)
    value_col = 3
    cluster_col = 4
    particle_col = 5

    def __init__(
        self,
        data: Any | None = None,
        particles: dict[str, Any] | None = None,
        meta: Any | None = None,
        *,
        coords: Any | None = None,
        features: Any | None = None,
        remove_duplicates: bool = True,
        sum_cols: np.ndarray | None = None,
    ) -> None:
        """Initialize compact labels from packed or split event arrays.

        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor, optional
            Packed ``[x, y, z, value, cluster, particle_index?]`` rows.
        particles : dict[str, numpy.ndarray or torch.Tensor], optional
            Named particle fields sharing a common row count.
        meta : object, optional
            Event image metadata.
        coords : numpy.ndarray or torch.Tensor, optional
            Coordinates used with ``features`` instead of packed ``data``.
        features : numpy.ndarray or torch.Tensor, optional
            Compact features used with ``coords``.
        remove_duplicates : bool, default True
            Whether collation should merge duplicate coordinates.
        sum_cols : numpy.ndarray, optional
            Compact feature columns summed during duplicate merging.

        Raises
        ------
        ValueError
            If packed and split forms are mixed or dimensions, particle fields
            and voxel associations are inconsistent.
        """
        # Normalize the two supported construction forms to one packed table
        if data is not None and (coords is not None or features is not None):
            raise ValueError("Provide packed `data` or `coords`/`features`, not both.")
        if data is None:
            if coords is None or features is None:
                raise ValueError("Must provide either `data` or both split arrays.")
            if isinstance(coords, torch.Tensor):
                data = torch.cat((coords, features), dim=1)
            else:
                data = np.concatenate((coords, features), axis=1)

        # Store compact voxel rows alongside the optional normalized table
        self.data = data
        self.particles = particles
        self.meta = meta
        self.remove_duplicates = remove_duplicates
        self.sum_cols = sum_cols
        self.__post_init__()

    @property
    def coords(self) -> Any:
        """Return the event coordinate matrix."""
        return self.data[:, self.coord_cols]

    @property
    def features(self) -> Any:
        """Return compact value, cluster and particle-association features."""
        return self.data[:, self.value_col :]

    @classmethod
    def metadata(cls, has_particles: bool = True) -> dict[str, Any]:
        """Return the stable serialized cluster-label schema.

        Parameters
        ----------
        has_particles : bool, default True
            Include the event-local particle-index field.
        """
        metadata = super().metadata()
        schema = cls.tensor_schema(has_particles)
        metadata.update(schema.to_dict())

        return metadata

    @staticmethod
    def tensor_schema(has_particles: bool) -> TensorSchema:
        """Return the compact voxel-table schema.

        Parameters
        ----------
        has_particles : bool
            Whether the table includes event-local particle associations.
        """
        # Particle-free labels contain only value and cluster association
        fields = {"value": (0,), "cluster": (1,)}
        if has_particles:
            fields["particle_index"] = (2,)

        return TensorSchema(
            coordinate_groups={"points": (0, 1, 2)},
            feature_fields=fields,
        )

    def __post_init__(self) -> None:
        """Validate association and particle-table dimensions."""
        # Validate the compact voxel layout first
        if self.data.ndim != 2 or self.data.shape[1] not in (5, 6):
            raise ValueError(
                "ClusterLabelData data must have columns [x, y, z, value, cluster] "
                "and optionally particle."
            )

        # Association-free products must omit the otherwise meaningless index
        if self.particles is None:
            if self.data.shape[1] != 5:
                raise ValueError(
                    "Particle indexes must be omitted when particle information "
                    "is unavailable."
                )
            return

        # Particle-backed labels require one event-local association per voxel
        if self.data.shape[1] != 6:
            raise ValueError("Particle information requires a particle-index column.")
        lengths = {len(value) for value in self.particles.values()}
        if len(lengths) > 1:
            raise ValueError("All particle fields must have the same length.")
        num_particles = next(iter(lengths), 0)

        # Check only valid associations; -1 deliberately represents no particle
        raw_index = self.data[:, self.particle_col]
        particle_index = (
            raw_index.detach().cpu().numpy().astype(np.int64, copy=False)
            if isinstance(raw_index, torch.Tensor)
            else np.asarray(raw_index, dtype=np.int64)
        )
        valid = particle_index >= 0
        if np.any(particle_index[valid] >= num_particles):
            raise ValueError("A voxel particle index lies outside the particle table.")

    def __len__(self) -> int:
        """Return the number of voxels in the event."""
        return len(self.data)

    def particle_field(self, name: str) -> Any:
        """Return one named field with one value per particle.

        Parameters
        ----------
        name : str
            Particle-table field name.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Field values in particle-table order.

        Raises
        ------
        ValueError
            If particle information is unavailable.
        KeyError
            If the requested field is not stored or virtual.
        """
        name = _field_name(name)
        if self.particles is None:
            raise ValueError(
                f"Particle field `{name}` is unavailable because this cluster "
                "label was parsed without particle information."
            )

        # Resolve virtual vertex components without duplicating table columns
        if name in {"vertex_x", "vertex_y", "vertex_z"}:
            return self.particles["vertex"][:, "xyz".index(name[-1])]

        # Resolve ancestor-derived values through the local relationship table
        if name == "ancestor_pid":
            return self._ancestor_field("pid")
        if name == "ancestor_momentum":
            return self._ancestor_field("momentum")

        # All remaining fields must be present directly in the particle table
        if name not in self.particles:
            raise KeyError(f"Particle field `{name}` is unavailable.")

        return self.particles[name]

    def _ancestor_field(self, name: str) -> Any:
        """Gather a particle field through event-local ancestor indexes.

        Unresolved ancestors retain the ``-1`` sentinel instead of indexing
        from the end of the particle table.
        """
        particles = self.particles
        if particles is None:
            raise ValueError("Particle information is unavailable.")

        # Allocate with the invalid sentinel so unresolved ancestors stay invalid
        values = particles[name]
        indexes = particles["ancestor"]
        output = (
            values.new_full((len(indexes), *values.shape[1:]), -1)
            if isinstance(values, torch.Tensor)
            else np.full((len(indexes), *values.shape[1:]), -1, dtype=values.dtype)
        )

        # Gather only resolved ancestors so negative sentinels remain intact
        valid = indexes >= 0
        output[valid] = values[
            (
                indexes[valid].long()
                if isinstance(indexes, torch.Tensor)
                else indexes[valid]
            )
        ]

        return output

    def voxel_field(self, name: str) -> Any:
        """Return one named field expanded to one value per voxel.

        Parameters
        ----------
        name : str
            Compact voxel field, particle field, alias or virtual field.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Values aligned one-to-one with compact voxel rows. Voxels without
            a particle association contain ``-1`` for particle-level fields.
        """
        name = _field_name(name)

        # Value and cluster ID live directly in the compact voxel table
        if name == "value":
            return self.data[:, self.value_col]
        if name == "cluster":
            return self.data[:, self.cluster_col]
        if name == "particle_index":
            if self.particles is None:
                raise ValueError("Particle indexes are unavailable.")
            return self.data[:, self.particle_col]

        # All other fields are gathered through the local particle association
        values = self.particle_field(name)
        raw_index = self.data[:, self.particle_col]
        particle_index = (
            raw_index.long()
            if isinstance(raw_index, torch.Tensor)
            else raw_index.astype(np.int64, copy=False)
        )
        output_shape = (len(self), *values.shape[1:])
        output = (
            values.new_full(output_shape, -1)
            if isinstance(values, torch.Tensor)
            else np.full(output_shape, -1, dtype=values.dtype)
        )

        # Gather valid local associations and preserve -1 for unassigned voxels
        valid = particle_index >= 0
        output[valid] = values[particle_index[valid]]

        return output
