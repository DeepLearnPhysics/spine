"""Batched structured voxel-cluster and particle-label data products."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ..cluster import ClusterLabelData, _ClusterLabelFieldAccessor, _field_name
from .tensor import TensorBatch

__all__ = ["ClusterLabelBatch"]


class ClusterLabelBatch(_ClusterLabelFieldAccessor):
    """Batched compact voxel labels and normalized particle information.

    The voxel table and particle table remain independently batched. Voxel
    rows carry event-local particle indexes, while each named particle field
    has one row per particle. Named accessors resolve this indirection without
    materializing dense particle information for every voxel.

    Parameters
    ----------
    data : TensorBatch
        Compact rows arranged as ``[batch, x, y, z, value, cluster]`` or
        ``[batch, x, y, z, value, cluster, particle_index]``.
    particles : dict[str, TensorBatch], optional
        Named particle fields. Every field must use the same backend, batch
        size and per-event counts.
    meta : list, optional
        Image metadata for each event. When omitted, metadata attached to
        ``data`` is retained.
    """

    def __init__(
        self,
        data: TensorBatch,
        particles: dict[str, TensorBatch] | None = None,
        meta: list[Any] | None = None,
    ) -> None:
        """Validate and initialize a cluster-label batch.

        Parameters
        ----------
        data : TensorBatch
            Compact rows ``[batch, x, y, z, value, cluster, particle?]``.
        particles : dict[str, TensorBatch], optional
            Named particle fields with common per-event counts.
        meta : list, optional
            Per-event image metadata. Defaults to ``data.meta``.

        Raises
        ------
        ValueError
            If the compact layout, coordinates, particle counts,
            associations or metadata are inconsistent.
        """
        # Validate the compact batch layout
        if not data.has_batch_col:
            raise ValueError("ClusterLabelBatch data must carry a batch column.")
        expected_width = 7 if particles is not None else 6
        if data.data.ndim != 2 or data.data.shape[1] != expected_width:
            raise ValueError(
                f"ClusterLabelBatch data must have {expected_width} columns."
            )
        expected_coord_cols = np.asarray([1, 2, 3], dtype=np.int64)
        if data.coord_cols is not None and not np.array_equal(
            data.coord_cols, expected_coord_cols
        ):
            raise ValueError(
                "ClusterLabelBatch coordinates must occupy packed columns 1, 2 and 3."
            )

        # Normalize the generic tensor schema to the stable cluster-label schema
        expected_schema = ClusterLabelData.tensor_schema(particles is not None)
        if data.schema != expected_schema or data.coord_cols is None:
            data = TensorBatch(
                data.data,
                data.counts,
                has_batch_col=True,
                coord_cols=expected_coord_cols,
                schema=expected_schema,
                meta=data.meta,
            )
        if particles:
            # Every field must describe the same particle rows in every event
            reference_counts = np.asarray(
                next(iter(particles.values())).to_numpy().counts
            )
            for name, particle_field in particles.items():
                if particle_field.is_numpy != data.is_numpy:
                    raise ValueError(
                        f"Particle field `{name}` uses a different array backend."
                    )
                if particle_field.batch_size != data.batch_size:
                    raise ValueError(
                        f"Particle field `{name}` has the wrong batch size."
                    )
                if not np.array_equal(
                    particle_field.to_numpy().counts, reference_counts
                ):
                    raise ValueError("Particle fields must share event counts.")

            # Validate event-local associations rather than global batch indexes
            data_numpy = data.to_numpy()
            particle_index = data_numpy.data[:, 6].astype(np.int64, copy=False)
            for batch_id in range(data.batch_size):
                lower, upper = data_numpy.edges[batch_id : batch_id + 2]
                local = particle_index[lower:upper]
                valid = local >= 0
                if np.any(local[valid] >= reference_counts[batch_id]):
                    raise ValueError(
                        "A voxel particle index is outside its event table."
                    )

        # Metadata is event-aligned and may be supplied by either layer
        if meta is None:
            meta = data.meta
        if meta is not None and len(meta) != data.batch_size:
            raise ValueError("Cluster-label metadata must contain one item per event.")

        # Store the independently batched voxel and particle products
        self.data = data
        self.particles = particles
        self.meta = meta

    def __len__(self) -> int:
        """Return the batch size."""
        return self.data.batch_size

    def __getitem__(self, batch_id: int) -> ClusterLabelData:
        """Return one event with event-local particle indexes.

        Parameters
        ----------
        batch_id : int
            Event position in the batch.

        Returns
        -------
        ClusterLabelData
            Self-describing event label with its particle table and metadata.
        """
        # Event extraction removes the packed batch column and retains schema.
        particles = None
        if self.particles is not None:
            particles = {
                name: field[batch_id] for name, field in self.particles.items()
            }
        data = self.data.event(batch_id)
        meta = None if self.meta is None else self.meta[batch_id]

        return ClusterLabelData(
            coords=data.coords,
            features=data.features,
            particles=particles,
            meta=meta,
        )

    @property
    def tensor(self) -> Any:
        """Return the compact batched voxel table."""
        return self.data.tensor

    @property
    def counts(self) -> Any:
        """Return voxel counts per event."""
        return self.data.counts

    @property
    def dtype(self) -> Any:
        """Return the compact voxel-table dtype."""
        return self.data.dtype

    @property
    def device(self) -> Any:
        """Return the compact voxel-table device."""
        return self.data.device

    @property
    def is_numpy(self) -> bool:
        """Whether the compact voxel table is NumPy-backed."""
        return self.data.is_numpy

    @property
    def batch_size(self) -> int:
        """Return the number of events in the batch."""
        return self.data.batch_size

    def to_tensor_batch(self) -> TensorBatch:
        """Return the compact voxel table for sparse network input."""
        return self.data

    @property
    def coords(self) -> TensorBatch:
        """Return voxel coordinates as a tensor batch."""
        return self.data.coords

    def coordinates(self, name: str | None = None) -> TensorBatch:
        """Return one named voxel-coordinate group.

        Parameters
        ----------
        name : str, optional
            Coordinate group. Cluster labels expose the sole ``points`` group,
            so this may normally be omitted.
        """
        return self.data.coordinates(name)

    @property
    def coordinate_data(self) -> TensorBatch:
        """Return the complete voxel coordinate matrix."""
        return cast(TensorBatch, self.data.coordinate_data)

    def select(self, index: Any, counts: Any) -> "ClusterLabelBatch":
        """Select voxel rows while retaining the event particle tables.

        Parameters
        ----------
        index : numpy.ndarray or torch.Tensor
            Global row indexes into the compact voxel table.
        counts : array-like
            Number of selected voxel rows in each event.

        Returns
        -------
        ClusterLabelBatch
            Restricted voxel associations sharing the unchanged particle
            tables and metadata.
        """
        data = TensorBatch(
            self.tensor[index],
            counts,
            has_batch_col=self.data.has_batch_col,
            coord_cols=self.data.coord_cols,
            schema=self.data.schema,
            meta=self.meta,
        )

        return ClusterLabelBatch(data, self.particles, self.meta)

    def numpy_tensor(self) -> np.ndarray:
        """Return the compact voxel table as a NumPy array.

        Raises
        ------
        TypeError
            If this batch is PyTorch-backed.
        """
        return self.data.numpy_tensor()

    def torch_tensor(self) -> Any:
        """Return the compact voxel table as a PyTorch tensor.

        Raises
        ------
        TypeError
            If this batch is NumPy-backed.
        """
        return self.data.torch_tensor()

    def particle_field(self, name: str) -> TensorBatch:
        """Return one named field with one row per particle.

        Parameters
        ----------
        name : str
            Stored field, accepted alias or virtual ancestor/vertex field.

        Returns
        -------
        TensorBatch
            Field values using particle counts, rather than voxel counts.

        Raises
        ------
        ValueError
            If particle information was not stored.
        KeyError
            If the requested field is unknown.
        """
        name = _field_name(name)
        if self.particles is None:
            raise ValueError(
                f"Particle field `{name}` is unavailable because this cluster "
                "label was parsed without particle information."
            )

        # Resolve virtual coordinate and ancestor fields on demand
        if name in {"vertex_x", "vertex_y", "vertex_z"}:
            vertex = self.particle_field("vertex")
            return TensorBatch(vertex.data[:, "xyz".index(name[-1])], vertex.counts)

        # Ancestor fields require conversion from local to global batch indexes
        if name in {"ancestor_pid", "ancestor_momentum"}:
            source = "pid" if name == "ancestor_pid" else "momentum"
            values = self.particle_field(source)
            indexes = self.particle_field("ancestor")

            # Convert event-local ancestor indexes to global batched indexes
            index = (
                indexes.data.long()
                if not indexes.is_numpy
                else indexes.data.astype(np.int64)
            )
            output = (
                values.data.new_full((len(index), *values.data.shape[1:]), -1)
                if not values.is_numpy
                else np.full(
                    (len(index), *values.data.shape[1:]), -1, dtype=values.dtype
                )
            )
            valid = index >= 0
            offsets = values.edges[:-1]
            if not indexes.is_numpy:
                # Structural boundaries stay on CPU; lookup indexes follow data
                offsets = offsets.to(index.device)
            global_index = index.clone() if not indexes.is_numpy else index.copy()
            global_index[valid] += offsets[indexes.batch_ids[valid]]
            output[valid] = values.data[global_index[valid]]

            return TensorBatch(output, indexes.counts)

        # All remaining particle fields must be stored directly
        if name not in self.particles:
            raise KeyError(f"Particle field `{name}` is unavailable.")

        return self.particles[name]

    def voxel_field(self, name: str) -> TensorBatch:
        """Expand one compact or particle-level field to the voxel rows.

        Invalid particle associations remain filled with ``-1``. The returned
        batch always has the same per-event counts as the voxel table.

        Parameters
        ----------
        name : str
            Compact voxel field, particle field, alias or virtual field.

        Returns
        -------
        TensorBatch
            Requested values aligned one-to-one with voxel rows.
        """
        name = _field_name(name)

        # Value and cluster ID live directly in the compact voxel table
        if name == "value":
            return self.data.feature("value").values
        if name == "cluster":
            return self.data.feature("cluster").values
        if name == "particle_index":
            if self.particles is None:
                raise ValueError("Particle indexes are unavailable.")
            return self.data.feature("particle_index").values

        # Expand particle-sized values through the event-local association
        field = self.particle_field(name)
        raw_particle_index = self.data.feature("particle_index").values.data
        particle_index = (
            raw_particle_index.long()
            if not self.data.is_numpy
            else raw_particle_index.astype(np.int64)
        )
        values = field.data
        output_shape = (len(self.tensor), *values.shape[1:])

        # Allocate invalid sentinels on the same backend as the particle field
        if self.data.is_numpy:
            output = np.full(output_shape, -1, dtype=values.dtype)
            valid = particle_index >= 0
        else:
            output = values.new_full(output_shape, -1)
            valid = particle_index >= 0

        # Convert event-local associations into the flattened particle namespace
        particle_offsets = field.edges[:-1]
        if not self.data.is_numpy:
            # Structural boundaries stay on CPU; lookup indexes follow data
            particle_offsets = particle_offsets.to(particle_index.device)
        global_index = (
            particle_index.copy() if self.data.is_numpy else particle_index.clone()
        )
        global_index[valid] += particle_offsets[self.data.batch_ids[valid]]
        output[valid] = values[global_index[valid]]

        return TensorBatch(output, self.counts)

    def to_numpy(self) -> "ClusterLabelBatch":
        """Return a NumPy-backed batch, preserving tables and metadata."""
        # Convert both independently batched products to keep their backends aligned
        particles = None
        if self.particles is not None:
            particles = {
                name: field.to_numpy() for name, field in self.particles.items()
            }

        return ClusterLabelBatch(self.data.to_numpy(), particles, self.meta)

    def to_tensor(self, dtype: Any = None, device: Any = None) -> "ClusterLabelBatch":
        """Return a PyTorch-backed batch.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Requested tensor dtype.
        device : torch.device, optional
            Requested tensor device.
        """
        # Convert both independently batched products to keep their backends aligned
        particles = None
        if self.particles is not None:
            particles = {
                name: field.to_tensor(dtype=dtype, device=device)
                for name, field in self.particles.items()
            }

        return ClusterLabelBatch(
            self.data.to_tensor(dtype=dtype, device=device), particles, self.meta
        )
