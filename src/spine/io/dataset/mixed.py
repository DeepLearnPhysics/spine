"""Dataset that merges aligned LArCV and HDF5-backed samples."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from .base import BaseDataset, DataDict
from .hdf5 import HDF5Dataset
from .larcv import LArCVDataset

__all__ = ["MixedDataset"]


class MixedDataset(BaseDataset):
    """Torch dataset that merges aligned samples from LArCV and HDF5.

    The LArCV dataset is treated as the primary source of iteration order and
    truth products. The HDF5 dataset acts as an aligned cache or augmentation
    source whose products are merged into the primary sample only after
    metadata and source provenance checks pass.
    """

    name: ClassVar[str] = "mixed"
    primary: LArCVDataset
    cache: HDF5Dataset
    reader: Any

    def __init__(
        self,
        larcv: Mapping[str, Any],
        hdf5: Mapping[str, Any],
        dtype: str,
        augment: Mapping[str, Any] | None = None,
        geo: Mapping[str, Any] | None = None,
        align_keys: Sequence[str] = ("file_index", "file_entry_index"),
        hdf5_align_keys: Mapping[str, str] | None = None,
        hdf5_key_map: Mapping[str, str] | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        """Instantiate the mixed dataset.

        Parameters
        ----------
        larcv : dict
            Configuration block for the LArCV-backed sample source
        hdf5 : dict
            Configuration block for the HDF5-backed cache source
        dtype : str
            Floating-point dtype used by parser factories
        augment : dict, optional
            Augmentation configuration applied once to the merged sample
        geo : dict, optional
            Geometry configuration forwarded to the LArCV dataset/augmenter
        align_keys : sequence[str], default ("file_index", "file_entry_index")
            Keys that must match between the LArCV and HDF5 samples
        hdf5_align_keys : dict, optional
            Optional mapping from LArCV alignment keys to HDF5 alignment keys.
            If not provided, the dataset uses `source_<key>` when that key is
            present in the HDF5 sample, and otherwise falls back to `<key>`.
        hdf5_key_map : dict, optional
            Optional rename map applied to HDF5 product keys before merging
        allow_overwrite : bool, default False
            If `True`, allow HDF5 products to overwrite colliding LArCV keys
        """
        super().__init__()

        self.align_keys = tuple(align_keys)
        self.hdf5_align_keys = dict(hdf5_align_keys or {})
        self.hdf5_key_map = dict(hdf5_key_map or {})
        self.allow_overwrite = allow_overwrite

        self.primary = LArCVDataset(**dict(larcv), dtype=dtype, augment=None, geo=geo)
        self.cache = HDF5Dataset(**dict(hdf5), dtype=dtype, augment=None)
        self.reader = self.primary.reader
        if len(self.primary) != len(self.cache):
            raise ValueError(
                "The LArCV and HDF5 sources must expose the same number of entries "
                f"to be mixed safely. Got {len(self.primary)} and {len(self.cache)}."
            )

        self.build_augmenter(augment, geo=geo)

    def __len__(self) -> int:
        """Return the number of aligned entries.

        Returns
        -------
        int
            Number of samples shared by the primary and cache datasets.
        """
        return len(self.primary)

    def __getitem__(self, idx: int) -> DataDict:
        """Return one merged sample from the aligned sources.

        Parameters
        ----------
        idx : int
            Dataset entry index shared by both underlying sources.

        Returns
        -------
        dict
            Merged sample dictionary containing primary LArCV products plus
            non-metadata HDF5 cache products.
        """
        primary = self.primary[idx]
        cache = self.cache[idx]
        self.validate_alignment(idx, primary, cache)
        merged = dict(primary)
        self.merge_cache(merged, cache)
        return self.apply_augmenter(merged)

    def validate_alignment(self, idx: int, primary: DataDict, cache: DataDict) -> None:
        """Ensure the configured alignment keys match between both sources.

        Parameters
        ----------
        idx : int
            Dataset entry index being validated.
        primary : dict
            Sample returned by the primary LArCV dataset.
        cache : dict
            Sample returned by the HDF5 cache dataset.
        """
        self.validate_source_alignment(idx, primary, cache)
        for key in self.align_keys:
            if key == "file_index" and "source_file_name" in cache:
                continue
            cache_key = self.resolve_cache_align_key(key, cache)
            if primary.get(key) != cache.get(cache_key):
                raise ValueError(
                    "MixedDataset source alignment failed at dataset index "
                    f"{idx}: LArCV key '{key}' and HDF5 key '{cache_key}' differ "
                    f"({primary.get(key)!r} != {cache.get(cache_key)!r})."
                )

    def validate_source_alignment(
        self, idx: int, primary: DataDict, cache: DataDict
    ) -> None:
        """Validate cache-file provenance against the current LArCV source file.

        This check is only applied when the HDF5 sample exposes staged-cache
        provenance keys. In that case the cache is expected to correspond to
        exactly one original source file, identified by file name, file size,
        and modification time.
        """
        if "source_file_name" not in cache:
            return

        file_idx = primary.get("file_index")
        assert isinstance(file_idx, int), "Primary file index should be an integer."
        source_path = self.primary.reader.file_paths[file_idx]
        source_stat = os.stat(source_path)
        source_name = os.path.basename(source_path)

        expected = {
            "source_file_name": source_name,
            "source_file_size": int(source_stat.st_size),
            "source_file_mtime_ns": int(source_stat.st_mtime_ns),
        }
        for key, value in expected.items():
            if key in cache and cache[key] != value:
                raise ValueError(
                    f"MixedDataset source provenance mismatch at dataset index {idx}: "
                    f"HDF5 '{key}' is {cache[key]!r}, expected {value!r}."
                )

    def resolve_cache_align_key(self, key: str, cache: DataDict) -> str:
        """Return the HDF5 key used to align one LArCV index field.

        Parameters
        ----------
        key : str
            Alignment key expected on the primary dataset side.
        cache : dict
            Cache sample dictionary used to determine whether a
            ``source_<key>`` variant is available.

        Returns
        -------
        str
            HDF5-side key name that should match the primary ``key``.
        """
        if key in self.hdf5_align_keys:
            return self.hdf5_align_keys[key]

        source_key = f"source_{key}"
        if source_key in cache:
            return source_key

        return key

    def merge_cache(self, merged: DataDict, cache: DataDict) -> None:
        """Merge one cached HDF5 sample into an existing LArCV sample.

        Parameters
        ----------
        merged : dict
            Mutable sample dictionary initially populated from the primary
            dataset.
        cache : dict
            HDF5 cache sample to merge into ``merged``.
        """
        for key, value in cache.items():
            if key in self._index_keys or key in self._source_keys:
                continue

            target_key = self.hdf5_key_map.get(key, key)
            if target_key in merged and not self.allow_overwrite:
                raise ValueError(
                    f"MixedDataset key collision for '{target_key}'. "
                    "Use `hdf5_key_map` or `allow_overwrite=True` to resolve it."
                )

            merged[target_key] = value

    @property
    def data_types(self) -> dict[str, str]:
        """Return the collate type for each merged product.

        Returns
        -------
        dict[str, str]
            Mapping from merged output key to collate type.
        """
        data_types = dict(self.primary.data_types)
        for key, value in self.cache.data_types.items():
            if key in self._index_keys or key in self._source_keys:
                continue

            target_key = self.hdf5_key_map.get(key, key)
            if target_key in data_types and data_types[target_key] != value:
                raise ValueError(
                    f"MixedDataset data type collision for '{target_key}': "
                    f"{data_types[target_key]!r} vs {value!r}."
                )

            data_types[target_key] = value

        return data_types

    @property
    def overlay_methods(self) -> dict[str, str]:
        """Return the overlay method for each merged product.

        Returns
        -------
        dict[str, str]
            Mapping from merged output key to overlay strategy.
        """
        overlay_methods = dict(self.primary.overlay_methods)
        for key, value in self.cache.overlay_methods.items():
            if key in self._index_keys or key in self._source_keys:
                continue

            target_key = self.hdf5_key_map.get(key, key)
            if target_key in overlay_methods and overlay_methods[target_key] != value:
                raise ValueError(
                    f"MixedDataset overlay collision for '{target_key}': "
                    f"{overlay_methods[target_key]!r} vs {value!r}."
                )

            overlay_methods[target_key] = value

        return overlay_methods

    @property
    def data_keys(self) -> tuple[str, ...]:
        """Return the names of all merged data products.

        Returns
        -------
        tuple[str, ...]
            Ordered tuple of keys exposed by the merged dataset.
        """
        return tuple(self.data_types.keys())
