"""Dataset wrapper around a manifest-backed SPINE cache repository."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from ..read import CacheReader
from ..read.base import ReaderBase
from .hdf5 import HDF5Dataset

__all__ = ["CacheDataset"]


class CacheDataset(HDF5Dataset):
    """Expose a logical sharded cache through the standard HDF5 parsers.

    Cache repositories reuse the ordinary V2 HDF5 product representation, so
    parsing and augmentation are identical. Physical shard discovery and
    stage composition are delegated to :class:`CacheReader`.
    """

    name: ClassVar[str] = "cache"
    supports_stages: ClassVar[bool] = True

    def __init__(self, path: str, **kwargs: Any) -> None:
        """Initialize a dataset from a ``.spine-cache`` repository path.

        Parameters
        ----------
        path : str
            Logical cache repository directory.
        **kwargs : Any
            Parser, projection, augmentation and reader options accepted by
            :class:`HDF5Dataset` and :class:`CacheReader`.
        """
        self.repository_path = path
        super().__init__(path=path, **kwargs)

    def _build_reader(
        self,
        stage: str | None,
        stage_map: Mapping[str, str],
        kwargs: Mapping[str, Any],
    ) -> ReaderBase:
        """Build the manifest-aware reader used by this dataset.

        Parameters
        ----------
        stage : str, optional
            Default cache stage selected by the dataset configuration.
        stage_map : mapping
            Explicit product-to-stage routing assembled from the dataset and
            parser schemas.
        kwargs : mapping
            Remaining physical-reader and entry-selection options.

        Returns
        -------
        ReaderBase
            Cache reader exposing the requested raw product projection.
        """
        selected_keys = tuple(self.keys) if self.keys is not None else None
        return CacheReader(
            stage=stage,
            stage_map=stage_map,
            keys=selected_keys,
            **kwargs,
        )
