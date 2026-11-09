"""PIMM detector-hit dataset using the common HDF5 parsing machinery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from spine.config.factory import module_dict

from ..parse import pimm
from ..read.pimm import PIMMReader
from .hdf5 import HDF5Dataset

__all__ = ["PIMMDataset"]


class PIMMDataset(HDF5Dataset):
    """Parse aligned PIMM hits and truth into aligned SPINE training products."""

    name: ClassVar[str] = "pimm"
    parser_dict = module_dict(pimm)

    def _build_reader(
        self, stage: str | None, stage_map: Mapping[str, str], kwargs: Mapping[str, Any]
    ) -> PIMMReader:
        """Select the native reader and enforce a common voxelization grid.

        Parameters
        ----------
        stage : str, optional
            Reserved cache-stage selection; always None for this dataset.
        stage_map : mapping
            Reserved cache product routing; empty for this dataset.
        kwargs : mapping
            Physical-reader and entry-selection options.

        Returns
        -------
        PIMMReader
            Native reader projected onto the products required by the schema.

        Raises
        ------
        ValueError
            If parsers request inconsistent grid settings.
        """
        del stage, stage_map

        # Every supervision product must use the same voxel coordinates.
        grids = {p.grid for p in self.parsers.values() if hasattr(p, "grid")}
        if len(grids) > 1:
            raise ValueError(
                "PIMM parsers must use the same voxel size, origin and unlabelled policy."
            )

        # Reuse the parent dataset's schema-driven physical projection.
        return PIMMReader(
            keys=None if self.keys is None else tuple(self.keys), **kwargs
        )
