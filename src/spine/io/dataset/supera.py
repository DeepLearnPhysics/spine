"""Native pysupera dataset using the common HDF5 parsing machinery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from spine.config.factory import module_dict

from ..parse import supera
from ..read.supera import SuperaReader
from .hdf5 import HDF5Dataset

__all__ = ["SuperaDataset"]


class SuperaDataset(HDF5Dataset):
    """Parse pysupera truth deposits into aligned SPINE training products."""

    name: ClassVar[str] = "supera"
    parser_dict = module_dict(supera)

    def _build_reader(
        self, stage: str | None, stage_map: Mapping[str, str], kwargs: Mapping[str, Any]
    ) -> SuperaReader:
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
        SuperaReader
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
            raise ValueError("Supera parsers must use the same voxel size and origin.")

        # Reuse the parent dataset's schema-driven physical projection.
        return SuperaReader(
            keys=None if self.keys is None else tuple(self.keys), **kwargs
        )
