"""Parser for structured cluster labels stored in SPINE HDF5 files."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import ClusterLabelData

from ..base import ParserBase

__all__ = ["HDF5ClusterLabelParser"]


class HDF5ClusterLabelParser(ParserBase):
    """Reconstruct a compact cluster-label event from cached products."""

    name = "cluster_label"

    def __init__(
        self,
        dtype: str,
        cluster_label_event: str,
        particle_event: str | None = None,
        meta_event: str | None = None,
    ) -> None:
        """Initialize the cached cluster-label parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype used for voxel values.
        cluster_label_event : str
            Compact voxel-association product stored in the HDF5 file.
        particle_event : str, optional
            Compound HDF5 product containing the named particle table.
        meta_event : str, optional
            Cached image metadata product.
        """
        super().__init__(
            dtype,
            cluster_label_event=cluster_label_event,
            particle_event=particle_event,
            meta_event=meta_event,
        )

    def __call__(self, trees: dict[str, Any]) -> ClusterLabelData:
        """Parse one cached cluster-label entry.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ClusterLabelData
            Compact cluster-label product with optional particles and metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self,
        cluster_label_event: np.ndarray,
        particle_event: Any | None = None,
        meta_event: Any | None = None,
    ) -> ClusterLabelData:
        """Split compact voxel rows and restore named particle fields.

        Parameters
        ----------
        cluster_label_event : np.ndarray
            Compact voxel table containing coordinates and association fields.
        particle_event : object collection, optional
            Serialized row-oriented particle-label objects.
        meta_event : object, optional
            Spatial metadata associated with the voxel table.

        Returns
        -------
        ClusterLabelData
            Fused voxel associations, named particle columns, and metadata.

        Raises
        ------
        ValueError
            If the compact voxel table does not have the expected shape.
        """
        # Validate the compact association product before splitting its columns
        data = np.asarray(cluster_label_event)
        expected_width = 6 if particle_event is not None else 5
        if data.ndim != 2 or data.shape[1] != expected_width:
            raise ValueError(
                f"Cached cluster labels must have {expected_width} columns, "
                f"received {data.shape}."
            )

        # Restore the optional compound particle sidecar as named arrays
        particles = None
        if particle_event is not None:
            fields = tuple(particle_event[0].as_dict()) if len(particle_event) else ()
            if not fields and hasattr(particle_event, "default"):
                fields = tuple(particle_event.default.as_dict())
            particles = {
                name: np.asarray(
                    [getattr(particle, name) for particle in particle_event]
                )
                for name in fields
            }

        # Fuse the physical voxel and particle products into one event product
        return ClusterLabelData(
            coords=data[:, :3].astype(self.itype, copy=False),
            features=data[:, 3:].astype(self.ftype, copy=False),
            particles=particles,
            meta=meta_event,
        )
