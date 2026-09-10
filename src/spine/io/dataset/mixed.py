"""Dataset that merges aligned primary and cache-backed samples."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from .base import BaseDataset, DataDict
from .hdf5 import HDF5Dataset
from .larcv import LArCVDataset

__all__ = ["MixedDataset"]

ENTRY_SELECTION_KEYS = (
    "n_entry",
    "n_skip",
    "entry_list",
    "skip_entry_list",
    "run_event_list",
    "skip_run_event_list",
    "entry_fraction_range",
)
CACHE_ENTRY_DOMAINS = ("auto", "source", "filtered")


class MixedDataset(BaseDataset):
    """Torch dataset that merges samples from aligned primary and cache sources.

    The primary dataset owns iteration order and usually supplies raw or truth
    products. The cache dataset supplies materialized products from an earlier
    processing stage. A sample is merged only after its configured event keys
    and, when available, immutable source-file provenance agree.

    The canonical configuration blocks are ``primary`` and ``cache``. The
    older ``larcv`` and ``hdf5`` spellings remain accepted for ordinary flat
    HDF5 workflows.
    """

    name: ClassVar[str] = "mixed"
    primary: Any
    cache: Any
    reader: Any

    def __init__(
        self,
        larcv: Mapping[str, Any] | None = None,
        hdf5: Mapping[str, Any] | None = None,
        dtype: str | None = None,
        augment: Mapping[str, Any] | None = None,
        align_keys: Sequence[str] = ("file_index", "file_entry_index"),
        hdf5_align_keys: Mapping[str, str] | None = None,
        hdf5_key_map: Mapping[str, str] | None = None,
        allow_overwrite: bool = False,
        entry_filter: str | None = None,
        cache_entry_domain: str = "auto",
        primary: Mapping[str, Any] | None = None,
        cache: Mapping[str, Any] | None = None,
        cache_align_keys: Mapping[str, str] | None = None,
        cache_key_map: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate the mixed dataset.

        Parameters
        ----------
        larcv : dict, optional
            Legacy configuration block for a LArCV primary source.
        hdf5 : dict, optional
            Legacy configuration block for a flat HDF5 cache source.
        dtype : str, optional
            Floating-point dtype used by parser factories. This value is
            required by the dataset factory.
        augment : dict, optional
            Augmentation configuration applied once to the merged sample
        align_keys : sequence[str], default ("file_index", "file_entry_index")
            Keys that must match between the primary and cache samples.
        hdf5_align_keys : dict, optional
            Legacy mapping from primary alignment keys to HDF5 alignment keys.
            If not provided, the dataset uses `source_<key>` when that key is
            present in the HDF5 sample, and otherwise falls back to `<key>`.
        hdf5_key_map : dict, optional
            Legacy rename map applied to HDF5 product keys before merging.
        allow_overwrite : bool, default False
            If `True`, allow cache products to overwrite colliding primary keys.
        entry_filter : str, optional
            File-aware primary-source eligibility manifest. The cache is assumed
            to contain either the original source domain or the accepted
            entries in compact order, according to ``cache_entry_domain``.
        cache_entry_domain : {"auto", "source", "filtered"}, default "auto"
            Entry domain represented by the cache when ``entry_filter`` is active.
            ``source`` retains all original entries, ``filtered`` contains only
            accepted entries, and ``auto`` infers the layout from cardinality.
        primary, cache : dict, optional
            Canonical child dataset configurations. Their ``name`` fields
            default to ``larcv`` and ``cache``, respectively. These replace
            the legacy ``larcv`` and ``hdf5`` block names.
        cache_align_keys, cache_key_map : dict, optional
            Canonical mapping of primary alignment keys to cache keys and
            canonical rename map for cache products, respectively.
        **kwargs : Any
            Shared keyword arguments forwarded to both underlying dataset
            constructors. This is primarily used for reader-level options such
            as entry-list filtering that must remain aligned across sources.
        """
        # Initialize the parent class
        super().__init__()

        # Accept the established configuration vocabulary while making the
        # child roles explicit for the new first-class cache dataset.
        if dtype is None:
            raise ValueError("MixedDataset requires an explicit `dtype`.")
        if primary is not None and larcv is not None:
            raise ValueError("Provide either `primary` or `larcv`, not both.")
        if cache is not None and hdf5 is not None:
            raise ValueError("Provide either `cache` or `hdf5`, not both.")
        if primary is None and larcv is None:
            raise ValueError("MixedDataset requires a `primary` dataset block.")
        if cache is None and hdf5 is None:
            raise ValueError("MixedDataset requires a `cache` dataset block.")
        if cache_align_keys is not None and hdf5_align_keys is not None:
            raise ValueError(
                "Provide either `cache_align_keys` or `hdf5_align_keys`, not both."
            )
        if cache_key_map is not None and hdf5_key_map is not None:
            raise ValueError(
                "Provide either `cache_key_map` or `hdf5_key_map`, not both."
            )

        # Store the alignment and merge configuration for use when samples are
        # fetched.
        self.align_keys = tuple(align_keys)
        self.cache_align_keys = dict(cache_align_keys or hdf5_align_keys or {})
        self.cache_key_map = dict(cache_key_map or hdf5_key_map or {})
        self.allow_overwrite = allow_overwrite
        if cache_entry_domain not in CACHE_ENTRY_DOMAINS:
            raise ValueError(
                f"Unknown `cache_entry_domain` `{cache_entry_domain}`; expected "
                f"one of {CACHE_ENTRY_DOMAINS}."
            )

        # Initialize the aligned sources through the ordinary dataset factory.
        # Shared positional selectors are forwarded to both children so their
        # exposed ordering remains one-to-one.
        from ..factories import (  # pylint: disable=import-outside-toplevel
            dataset_factory,
        )

        primary_config = dict(primary if primary is not None else larcv or {})
        primary_config.setdefault("name", "larcv")
        if entry_filter is not None:
            primary_config["entry_filter"] = entry_filter
        primary_config.update(kwargs)
        primary_config["augment"] = None
        resolved_entry_filter = primary_config.get("entry_filter")
        if primary is None:
            legacy_primary = dict(primary_config)
            legacy_primary.pop("name", None)
            self.primary = LArCVDataset(**legacy_primary, dtype=dtype)
        else:
            self.primary = dataset_factory(primary_config, dtype=dtype)

        # The primary manifest is not blindly forwarded to the cache. The
        # cache may already represent the compact filtered domain, in which
        # case its physical indexes no longer match raw-source indexes.
        cache_kwargs = dict(kwargs)
        nested_cache = dict(cache if cache is not None else hdf5 or {})
        if resolved_entry_filter is not None:
            nested_selectors = [
                key
                for key in ENTRY_SELECTION_KEYS
                if key in nested_cache and nested_cache[key] is not None
            ]
            if nested_selectors:
                raise ValueError(
                    "When using `entry_filter`, configure mixed-dataset entry "
                    "selection at the mixed root, not inside `cache`."
                )
            for key in ENTRY_SELECTION_KEYS:
                cache_kwargs.pop(key, None)

        nested_cache.setdefault("name", "cache" if cache is not None else "hdf5")
        nested_cache.update(cache_kwargs)
        nested_cache["augment"] = None
        if cache is None:
            legacy_cache = dict(nested_cache)
            legacy_cache.pop("name", None)
            self.cache = HDF5Dataset(**legacy_cache, dtype=dtype)
        else:
            self.cache = dataset_factory(nested_cache, dtype=dtype)

        if resolved_entry_filter is not None:
            self._select_cache_entries(cache_entry_domain)

        self.reader = self.primary.reader
        if len(self.primary) != len(self.cache):
            raise ValueError(
                "The primary and cache sources must expose the same number of entries "
                f"to be mixed safely. Got {len(self.primary)} and {len(self.cache)}."
            )

        # Initialize the augmenter
        self.build_augmenter(augment)

    def _select_cache_entries(self, cache_entry_domain: str) -> None:
        """Project the cache onto the final filtered primary selection.

        Parameters
        ----------
        cache_entry_domain : {"auto", "source", "filtered"}
            Configured cache entry-domain policy. Auto detection compares the
            cache cardinality with the complete raw and eligible populations.

        Raises
        ------
        ValueError
            If the requested domain is inconsistent with cache cardinality or
            auto detection cannot identify either supported layout.
        """
        primary_reader = self.primary.reader
        cache_reader = self.cache.reader
        source_count = int(primary_reader.num_entries)
        eligible = np.asarray(primary_reader.eligible_entry_index, dtype=np.int64)
        cache_count = int(cache_reader.num_entries)

        # Cardinality identifies whether the cache retained rejected entries.
        domain = cache_entry_domain
        if domain == "auto":
            if cache_count == len(eligible):
                domain = "filtered"
            elif cache_count == source_count:
                domain = "source"
            else:
                raise ValueError(
                    "Could not infer the mixed cache entry domain: cache "
                    f"has {cache_count} entries, while the raw and filtered "
                    f"domains contain {source_count} and {len(eligible)}."
                )

        expected_count = source_count if domain == "source" else len(eligible)
        if cache_count != expected_count:
            raise ValueError(
                f"Cache `cache_entry_domain: {domain}` requires {expected_count} "
                f"entries, found {cache_count}."
            )

        selected = np.asarray(primary_reader.entry_index, dtype=np.int64)
        if domain == "filtered":
            # Eligible indexes are source ordered. Their positions are exactly
            # the compact entry indexes written by a filtered cache job.
            compact = np.searchsorted(eligible, selected)
            if np.any(compact >= len(eligible)) or not np.array_equal(
                eligible[compact], selected
            ):
                raise ValueError(
                    "The final primary selection is not contained in its eligible "
                    "entry domain."
                )
            selected = compact

        cache_reader.process_entry_list(entry_list=selected.tolist())

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
            Merged sample dictionary containing primary products plus
            non-administrative cache products.
        """
        return self._merge_sample(idx, self.primary[idx], self.cache[idx])

    def __getitems__(self, indices: Sequence[int]) -> list[DataDict]:
        """Load and merge a batch of aligned primary and cache samples.

        Parameters
        ----------
        indices : sequence[int]
            Shared dataset indexes requested from both sources.

        Returns
        -------
        list[dict]
            Validated merged samples in the same order as ``indices``.

        Raises
        ------
        RuntimeError
            If either child dataset returns an incomplete batch.
        """
        # Normalize once so both sources receive identical indexes and ordering
        indices = [int(idx) for idx in indices]
        primary_batch = self.load_batch(self.primary, indices)
        cache_batch = self.load_batch(self.cache, indices)

        # A partial child result would make source alignment ambiguous
        if len(primary_batch) != len(cache_batch) or len(primary_batch) != len(indices):
            raise RuntimeError("MixedDataset sources returned an incomplete batch.")

        return [
            self._merge_sample(idx, primary, cache)
            for idx, primary, cache in zip(indices, primary_batch, cache_batch)
        ]

    def _merge_sample(self, idx: int, primary: DataDict, cache: DataDict) -> DataDict:
        """Validate and merge one already-loaded aligned source pair.

        Parameters
        ----------
        idx : int
            Shared dataset index used for alignment diagnostics.
        primary : dict
            Sample loaded from the primary source.
        cache : dict
            Corresponding sample loaded from the cache source.

        Returns
        -------
        dict
            Merged and augmented sample.
        """
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
            Sample returned by the primary dataset.
        cache : dict
            Sample returned by the cache dataset.
        """
        self.validate_source_alignment(idx, primary, cache)
        for key in self.align_keys:
            if key == "file_index" and "source_file_name" in cache:
                continue
            cache_key = self.resolve_cache_align_key(key, cache)
            if primary.get(key) != cache.get(cache_key):
                raise ValueError(
                    "MixedDataset source alignment failed at dataset index "
                    f"{idx}: primary key '{key}' and cache key '{cache_key}' differ "
                    f"({primary.get(key)!r} != {cache.get(cache_key)!r})."
                )

    def validate_source_alignment(
        self, idx: int, primary: DataDict, cache: DataDict
    ) -> None:
        """Validate cache provenance against the current primary source file.

        This check is only applied when the cache sample exposes source
        provenance keys. In that case the cache is expected to correspond to
        exactly one original source file, identified by file name, file size,
        and modification time.

        Parameters
        ----------
        idx : int
            Dataset entry index being validated.
        primary : dict
            Sample returned by the primary dataset.
        cache : dict
            Sample returned by the cache dataset.
        """
        if "source_file_name" not in cache:
            return

        # Resolve the primary file identity from its active reader.
        file_idx = primary.get("file_index")
        assert isinstance(file_idx, int), "Primary file index should be an integer."
        source_path = self.primary.reader.file_paths[file_idx]
        source_stat = os.stat(source_path)
        source_name = os.path.basename(source_path)

        # Compare every provenance component advertised by the cache
        expected = {
            "source_file_name": source_name,
            "source_file_size": int(source_stat.st_size),
            "source_file_mtime_ns": int(source_stat.st_mtime_ns),
        }
        for key, value in expected.items():
            if key in cache and cache[key] != value:
                raise ValueError(
                    f"MixedDataset source provenance mismatch at dataset index {idx}: "
                    f"cache '{key}' is {cache[key]!r}, expected {value!r}."
                )

    def resolve_cache_align_key(self, key: str, cache: DataDict) -> str:
        """Return the cache key used to align one primary index field.

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
            Cache-side key name that should match the primary ``key``.
        """
        if key in self.cache_align_keys:
            return self.cache_align_keys[key]

        source_key = f"source_{key}"
        if source_key in cache:
            return source_key

        return key

    def merge_cache(self, merged: DataDict, cache: DataDict) -> None:
        """Merge one cache sample into an existing primary sample.

        Parameters
        ----------
        merged : dict
            Mutable sample dictionary initially populated from the primary
            dataset.
        cache : dict
            Cache sample to merge into ``merged``.
        """
        for key, value in cache.items():
            # Administrative identity remains authoritative on the primary side
            if key in self._index_keys or key in self._source_keys:
                continue

            # Apply public renames and reject accidental product replacement
            target_key = self.cache_key_map.get(key, key)
            if target_key in merged and not self.allow_overwrite:
                raise ValueError(
                    f"MixedDataset key collision for '{target_key}'. "
                    "Use `cache_key_map` or `allow_overwrite=True` to resolve it."
                )

            merged[target_key] = value

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

            target_key = self.cache_key_map.get(key, key)
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
        keys = list(self.primary.data_keys)
        for key in self.cache.data_keys:
            if key in self._index_keys or key in self._source_keys:
                continue
            target_key = self.cache_key_map.get(key, key)
            if target_key not in keys:
                keys.append(target_key)
        return tuple(keys)
