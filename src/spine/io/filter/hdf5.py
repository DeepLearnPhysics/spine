"""SPINE HDF5 and cache-repository inspectors for entry filtering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..cache import CacheManifest, CacheRepository, CacheSource
from .base import EntryInspector, SourceFingerprint

__all__ = ["CacheEntryInspector", "HDF5EntryInspector"]


def _require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """Return a named HDF5 group after validating its physical type.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        Container expected to own the group.
    name : str
        Child name relative to ``parent``.

    Returns
    -------
    h5py.Group
        Validated child group.

    Raises
    ------
    TypeError
        If the named child is not an HDF5 group.
    """
    child = parent[name]
    if not isinstance(child, h5py.Group):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 group.")
    return child


def _require_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    """Return a named HDF5 dataset after validating its physical type.

    Parameters
    ----------
    parent : h5py.Group
        Container expected to own the dataset.
    name : str
        Child name relative to ``parent``.

    Returns
    -------
    h5py.Dataset
        Validated child dataset.

    Raises
    ------
    TypeError
        If the named child is not an HDF5 dataset.
    """
    child = parent[name]
    if not isinstance(child, h5py.Dataset):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 dataset.")
    return child


def _measurement_product(name: str, request: Mapping[str, Any]) -> str:
    """Resolve the physical product name for one measurement.

    Parameters
    ----------
    name : str
        Public measurement name, used as the product name by default.
    request : mapping
        Normalized measurement configuration.

    Returns
    -------
    str
        Nonempty physical product name.

    Raises
    ------
    TypeError
        If an explicit product name is not a nonempty string.
    """
    product = request.get("product", name)
    if not isinstance(product, str) or not product:
        raise TypeError(f"Measurement `{name}` `product` must be a nonempty string.")
    return product


def _offset_counts(group: h5py.Group, product: str, num_entries: int) -> list[int]:
    """Read event row counts for one V2 product without loading its payload.

    Parameters
    ----------
    group : h5py.Group
        V2 ``products`` group containing the requested logical product.
    product : str
        Logical product name.
    num_entries : int
        Expected length of the event axis.

    Returns
    -------
    list[int]
        Number of product rows stored for each event.

    Raises
    ------
    KeyError
        If the requested product is absent.
    TypeError
        If the product does not use the V2 offset layout.
    ValueError
        If the offsets do not match the event axis or are not monotonic.
    """
    if product not in group:
        raise KeyError(f"Missing requested HDF5 product `{product}`.")
    product_group = group[product]
    if (
        not isinstance(product_group, h5py.Group)
        or "event_offsets" not in product_group
    ):
        raise TypeError(f"HDF5 product `{product}` does not expose V2 `event_offsets`.")
    offsets = product_group["event_offsets"]
    if not isinstance(offsets, h5py.Dataset) or offsets.ndim != 1:
        raise TypeError(f"HDF5 product `{product}` has invalid `event_offsets`.")
    if len(offsets) != num_entries + 1:
        raise ValueError(
            f"HDF5 product `{product}` has {len(offsets) - 1} entries; "
            f"expected {num_entries}."
        )
    # Adjacent offsets encode the first-axis row count. Reading this small
    # metadata column avoids materializing the potentially enormous payload.
    values = np.asarray(offsets[:], dtype=np.int64)
    counts = np.diff(values)
    if np.any(counts < 0):
        raise ValueError(f"HDF5 product `{product}` has non-monotonic offsets.")
    return counts.tolist()


def _legacy_counts(in_file: h5py.File, product: str, num_entries: int) -> list[int]:
    """Measure V1 region selections without materializing product arrays.

    Parameters
    ----------
    in_file : h5py.File
        Open legacy SPINE HDF5 file.
    product : str
        Top-level product dataset name.
    num_entries : int
        Expected length of the event axis.

    Returns
    -------
    list[int]
        First-axis extent of the product selection for each event.

    Raises
    ------
    KeyError
        If the event table does not reference the requested product.
    TypeError
        If the referenced product is not a dataset.
    """
    events = in_file["events"]
    if not isinstance(events, h5py.Dataset) or product not in (
        events.dtype.names or ()
    ):
        raise KeyError(f"Missing requested HDF5 product `{product}`.")
    dataset = in_file[product]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Legacy HDF5 product `{product}` is not a dataset.")

    counts = []
    for entry in range(num_entries):
        # ``selection`` asks HDF5 for the region shape; unlike ``dataset[ref]``,
        # it does not read the selected edge rows into memory.
        reference = events[entry][product]
        selection = dataset.regionref.selection(reference)
        counts.append(int(selection[0]) if selection else 0)
    return counts


class HDF5EntryInspector(EntryInspector):
    """Measure logical product sizes in ordinary SPINE HDF5 files.

    Notes
    -----
    Both supported physical layouts are metadata-only: V2 uses event offsets
    and V1 uses region-reference selection extents. Product values are never
    deserialized during a scan.
    """

    name = "hdf5"
    version = 1

    def inspect(
        self, source: str, measurements: Mapping[str, Mapping[str, Any]]
    ) -> tuple[int, dict[str, list[int]]]:
        """Read V2 offsets, or V1 region extents, for requested products.

        Parameters
        ----------
        source : str
            Path to one ordinary SPINE HDF5 file.
        measurements : mapping
            Measurement names mapped to normalized ``product_size`` requests.
            Each request may name a physical ``product``; otherwise the
            measurement name is used.

        Returns
        -------
        num_entries : int
            Number of events in the file.
        values : dict[str, list[int]]
            Per-event row counts for every requested measurement.

        Raises
        ------
        KeyError
            If the event axis, product root or requested product is absent.
        TypeError
            If the event axis, product root or information node has an
            unexpected HDF5 type.
        ValueError
            If the physical format version or product offsets are invalid.
        """
        with h5py.File(source, "r") as in_file:
            events = in_file.get("events")
            if events is None:
                raise KeyError(f"HDF5 source `{source}` has no event axis.")
            if not isinstance(events, h5py.Dataset):
                raise TypeError(f"HDF5 source `{source}` has an invalid event axis.")
            num_entries = len(events)

            info = in_file.get("info")
            if info is not None and not isinstance(info, h5py.Group):
                raise TypeError(f"HDF5 source `{source}` has an invalid info node.")
            version = 1
            if info is not None:
                version = int(info.attrs.get("format_version", 1))

            products = in_file.get("products")
            values = {}
            for name, request in measurements.items():
                product = _measurement_product(name, request)
                if version == 2:
                    if products is None:
                        raise KeyError(f"HDF5 source `{source}` has no product root.")
                    if not isinstance(products, h5py.Group):
                        raise TypeError(
                            f"HDF5 source `{source}` has an invalid product root."
                        )
                    values[name] = _offset_counts(products, product, num_entries)
                elif version == 1:
                    values[name] = _legacy_counts(in_file, product, num_entries)
                else:
                    raise ValueError(
                        f"Unsupported HDF5 format version {version} in `{source}`."
                    )
        return num_entries, values


class CacheEntryInspector(EntryInspector):
    """Measure product sizes across a logical SPINE cache repository.

    The repository manifest supplies the stable source and stage order. Each
    requested product is resolved to exactly one published stage, then its V2
    event offsets are read from that stage's immutable per-source shards.
    """

    name = "cache"
    version = 1

    def fingerprint(self, source: str) -> SourceFingerprint:
        """Fingerprint the published repository manifest, not its directory.

        Parameters
        ----------
        source : str
            Cache repository directory.

        Returns
        -------
        SourceFingerprint
            Repository identity based on the canonical directory and current
            published manifest metadata.

        Raises
        ------
        FileNotFoundError
            If the directory has no published ``manifest.json``.

        Notes
        -----
        Cache stage replacement atomically changes ``manifest.json`` while
        leaving the repository directory itself in place. Fingerprinting the
        manifest therefore invalidates stale scan records correctly.
        """
        repository = Path(self.canonical_source(source))
        manifest = repository / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Cache repository `{repository}` has no manifest.json."
            )
        stat_result = manifest.stat()
        return SourceFingerprint(
            path=str(repository),
            size=int(stat_result.st_size),
            mtime_ns=int(stat_result.st_mtime_ns),
        )

    def inspect(
        self, source: str, measurements: Mapping[str, Mapping[str, Any]]
    ) -> tuple[int, dict[str, list[int]]]:
        """Read product offsets from the owning stage's published shards.

        Parameters
        ----------
        source : str
            Logical ``.spine-cache`` repository directory.
        measurements : mapping
            Measurement names mapped to normalized ``product_size`` requests.
            Requests may specify ``product`` and, when ownership is ambiguous,
            ``stage``.

        Returns
        -------
        num_entries : int
            Total number of logical events across repository sources.
        values : dict[str, list[int]]
            Per-event row counts in canonical repository source order.
        """
        repository = CacheRepository(source)
        manifest = repository.load()
        values = {name: [] for name in measurements}
        total_entries = sum(item.num_entries for item in manifest.sources)
        owners = self._resolve_owners(manifest, measurements)

        # Preserve the manifest's source order so recorded indexes match the
        # logical event axis later exposed by CacheReader.
        for cache_source in manifest.sources:
            source_values = self._inspect_source(
                repository, manifest, cache_source, owners
            )
            for name, counts in source_values.items():
                values[name].extend(counts)

        return total_entries, values

    @staticmethod
    def _resolve_owners(
        manifest: CacheManifest,
        measurements: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, tuple[str, str]]:
        """Resolve each requested product to one published cache stage.

        Parameters
        ----------
        manifest : CacheManifest
            Immutable repository snapshot used for the complete scan.
        measurements : mapping
            Normalized measurement requests.

        Returns
        -------
        dict[str, tuple[str, str]]
            Mapping from measurement name to ``(stage, product)``.

        Raises
        ------
        KeyError
            If an explicitly selected stage or product is absent.
        ValueError
            If inferred product ownership is missing or ambiguous.
        """
        owners: dict[str, tuple[str, str]] = {}
        for name, request in measurements.items():
            product = _measurement_product(name, request)
            configured_stage = request.get("stage")
            # Explicit ownership resolves intentionally duplicated product
            # names; otherwise require one unambiguous published owner.
            if configured_stage is not None:
                if not isinstance(configured_stage, str) or not configured_stage:
                    raise TypeError(
                        f"Measurement `{name}` `stage` must be a nonempty string."
                    )
                candidates = [configured_stage]
            else:
                candidates = [
                    stage_name
                    for stage_name, stage in manifest.stages.items()
                    if product in stage.products
                ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Product `{product}` must resolve to exactly one cache stage; "
                    f"found {candidates}."
                )
            stage_name = candidates[0]
            if stage_name not in manifest.stages:
                raise KeyError(f"Cache repository has no stage `{stage_name}`.")
            stage = manifest.stages[stage_name]
            if product not in stage.products:
                raise KeyError(
                    f"Cache stage `{stage_name}` has no product `{product}`."
                )
            owners[name] = (stage_name, product)
        return owners

    @staticmethod
    def _inspect_source(
        repository: CacheRepository,
        manifest: CacheManifest,
        cache_source: CacheSource,
        owners: Mapping[str, tuple[str, str]],
    ) -> dict[str, list[int]]:
        """Read all requested offset columns for one logical source shard.

        Parameters
        ----------
        repository : CacheRepository
            Repository used to resolve immutable shard paths safely.
        manifest : CacheManifest
            Published snapshot containing stage and shard routing.
        cache_source : CacheSource
            Source whose stage shards should be inspected.
        owners : mapping
            Measurement names mapped to ``(stage, product)`` pairs.

        Returns
        -------
        dict[str, list[int]]
            Per-event counts for this source, keyed by measurement name.

        Raises
        ------
        TypeError
            If a selected shard does not expose the expected V2 groups.
        ValueError
            If a stage shard's event count disagrees with the repository.
        """
        values = {}
        opened: dict[str, h5py.File] = {}
        try:
            for name, (stage_name, product) in owners.items():
                # Multiple measurements may share a stage. Reuse its read-only
                # handle for this source, then close all handles in ``finally``.
                if stage_name not in opened:
                    stage = manifest.stages[stage_name]
                    shard = repository.resolve_shard(stage.shards[cache_source.id])
                    opened[stage_name] = h5py.File(shard, "r")
                stage_group = _require_group(
                    _require_group(opened[stage_name], "stages"), stage_name
                )
                events = _require_dataset(stage_group, "events")
                if len(events) != cache_source.num_entries:
                    raise ValueError(
                        f"Cache stage `{stage_name}` has {len(events)} entries "
                        f"for source `{cache_source.id}`; expected "
                        f"{cache_source.num_entries}."
                    )
                products = _require_group(stage_group, "products")
                values[name] = _offset_counts(
                    products, product, cache_source.num_entries
                )
        finally:
            for in_file in opened.values():
                in_file.close()

        return values
