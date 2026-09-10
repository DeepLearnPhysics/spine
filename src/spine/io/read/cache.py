"""Reader for manifest-backed SPINE cache repositories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..cache import CacheManifest, CacheRepository
from ..cache.backend.hdf5.reader import HDF5ShardReader
from .base import ReaderBase

__all__ = ["CacheReader"]


class CacheReader(ReaderBase):
    """Read a consistent snapshot of stage shards as one logical dataset.

    The manifest is read once during initialization. Each selected logical
    stage is backed by a private HDF5 V2 shard reader, and their products
    are merged by aligned event index. A later manifest publication therefore
    cannot change the files observed by an active reader.
    """

    name = "cache"

    def __init__(
        self,
        path: str,
        stage: str | None = None,
        stage_map: Mapping[str, str] | None = None,
        keys: Sequence[str] | None = None,
        n_entry: int | None = None,
        n_skip: int | None = None,
        entry_list: str | list[int] | None = None,
        skip_entry_list: str | list[int] | None = None,
        build_classes: bool = True,
        skip_unknown_attrs: bool = False,
        allow_missing: bool = False,
        keep_open: bool = True,
        swmr: bool = False,
        entry_fraction_range: Sequence[float] | None = None,
        entry_filter: str | None = None,
        max_print_files: int = 10,
    ) -> None:
        """Initialize a reader from one published repository snapshot.

        Parameters
        ----------
        path : str
            Logical ``.spine-cache`` repository directory.
        stage : str, optional
            Default stage from which requested products are resolved.
        stage_map : mapping, optional
            Explicit product-to-stage assignments.
        keys : sequence[str], optional
            Raw products to expose. If omitted, expose all products from the
            default stage, or all uniquely named products across every stage.
        n_entry, n_skip, entry_list, skip_entry_list, entry_fraction_range : optional
            Standard entry-selection controls.
        build_classes, skip_unknown_attrs, allow_missing, keep_open, swmr : optional
            HDF5 decoding and handle-lifetime controls.
        entry_filter : str, optional
            File-aware eligibility manifest applied through source provenance
            persisted in every stage shard.
        max_print_files : int, default 10
            Maximum physical shard names printed by each stage reader.

        Raises
        ------
        ValueError
            If the repository has no stages, selected stages have incompatible
            event axes, or product ownership is ambiguous.
        KeyError
            If a requested stage or product is absent from the manifest.
        """
        self.repository = CacheRepository(path)
        self.manifest = self.repository.load()
        if len(self.manifest.stages) == 0:
            raise ValueError(f"Cache repository '{path}' contains no stages.")

        routing = self._resolve_products(self.manifest, stage, stage_map, keys)
        reader_options = {
            "n_entry": n_entry,
            "n_skip": n_skip,
            "entry_list": entry_list,
            "skip_entry_list": skip_entry_list,
            "build_classes": build_classes,
            "skip_unknown_attrs": skip_unknown_attrs,
            "allow_missing": allow_missing,
            "keep_open": keep_open,
            "swmr": swmr,
            "entry_fraction_range": entry_fraction_range,
            "entry_filter": entry_filter,
            "max_print_files": max_print_files,
        }

        # Use source IDs as shard basenames; sorting inside the shard reader then
        # produces identical physical ordering for every independently written
        # stage generation.
        source_ids = sorted(source.id for source in self.manifest.sources)
        self._readers: dict[str, HDF5ShardReader] = {}
        for stage_name, stage_keys in routing.items():
            stage_record = self.manifest.stages[stage_name]
            shards = [
                self.repository.resolve_shard(stage_record.shards[source_id])
                for source_id in source_ids
            ]
            self._readers[stage_name] = HDF5ShardReader(
                stage=stage_name,
                keys=stage_keys,
                file_keys=shards,
                **reader_options,
            )

        # Every selected stage must expose the same projected logical event
        # axis before their products can be merged safely.
        readers = list(self._readers.values())
        reference = readers[0]
        for other in readers[1:]:
            if other.num_entries != reference.num_entries or not np.array_equal(
                other.entry_index, reference.entry_index
            ):
                raise ValueError("Published cache stages have misaligned event axes.")

        self.num_entries = reference.num_entries
        self.entry_index = reference.entry_index.copy()
        self.eligible_entry_index = reference.eligible_entry_index.copy()
        self.file_index = reference.file_index.copy()
        self.file_offsets = reference.file_offsets.copy()
        self.run_info = None
        self.run_map = None
        self.file_paths = list(reference.file_paths)
        self.cfg = self._collect_attribute("cfg")
        self.version = self._collect_attribute("version")
        self.post_processors = None

    @staticmethod
    def _resolve_products(
        manifest: CacheManifest,
        stage: str | None,
        stage_map: Mapping[str, str] | None,
        keys: Sequence[str] | None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve requested products to their owning published stages.

        Parameters
        ----------
        manifest : CacheManifest
            Repository snapshot containing available stages and products.
        stage : str, optional
            Default owning stage for products without an explicit mapping.
        stage_map : mapping, optional
            Per-product stage overrides.
        keys : sequence[str], optional
            Product projection. If omitted, infer the projection from the
            selected stage or from all published stages.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Requested products grouped by owning stage.

        Raises
        ------
        KeyError
            If a configured stage or requested product does not exist.
        ValueError
            If automatic discovery finds the same product in multiple stages.
        """
        explicit = dict(stage_map or {})
        if stage is not None and stage not in manifest.stages:
            raise KeyError(f"Cache repository does not contain stage '{stage}'.")

        if keys is None and stage is not None and not explicit:
            return {stage: manifest.stages[stage].products}

        requested = (
            tuple(keys)
            if keys is not None
            else tuple(
                key
                for stage_record in manifest.stages.values()
                for key in stage_record.products
            )
        )
        # Explicit routes take precedence, followed by the default stage. Only
        # otherwise is ownership inferred from the published product schemas.
        result: dict[str, list[str]] = {}
        for key in requested:
            if key in explicit:
                owner = explicit[key]
                if owner not in manifest.stages:
                    raise KeyError(
                        f"Cache repository does not contain stage '{owner}'."
                    )
                if key not in manifest.stages[owner].products:
                    raise KeyError(
                        f"Cache stage '{owner}' does not contain product '{key}'."
                    )
            elif stage is not None:
                owner = stage
                if key not in manifest.stages[owner].products:
                    raise KeyError(
                        f"Cache stage '{owner}' does not contain product '{key}'."
                    )
            else:
                candidates = [
                    name
                    for name, record in manifest.stages.items()
                    if key in record.products
                ]
                if len(candidates) == 0:
                    raise KeyError(f"No cache stage contains product '{key}'.")
                if len(candidates) > 1:
                    raise ValueError(
                        f"Product '{key}' appears in multiple cache stages: "
                        f"{candidates}. Specify its stage explicitly."
                    )
                owner = candidates[0]
            result.setdefault(owner, []).append(key)

        return {name: tuple(stage_keys) for name, stage_keys in result.items()}

    def get(self, idx: int) -> dict[str, Any]:
        """Load and merge one event from every selected stage shard.

        Parameters
        ----------
        idx : int
            User-facing index into the projected cache entry sequence.

        Returns
        -------
        dict[str, Any]
            Combined products and administrative metadata for the event.
        """
        return self._merge_entries(
            {name: reader[idx] for name, reader in self._readers.items()}
        )

    def get_many(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Load several events through each stage's batched HDF5 V2 path.

        Parameters
        ----------
        indices : sequence[int]
            User-facing entry indexes to read in the requested order.

        Returns
        -------
        list[dict[str, Any]]
            Merged event dictionaries in the same order as ``indices``.
        """
        batches = {
            name: reader.get_many(indices) for name, reader in self._readers.items()
        }
        return [
            self._merge_entries(
                {name: batch[position] for name, batch in batches.items()}
            )
            for position in range(len(indices))
        ]

    @staticmethod
    def _merge_entries(entries: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Merge aligned stage entries while validating shared metadata.

        Parameters
        ----------
        entries : mapping
            Event dictionaries keyed by their owning stage.

        Returns
        -------
        dict[str, Any]
            One logical event containing the union of stage products.

        Raises
        ------
        ValueError
            If stages expose the same product or disagree on administrative
            event metadata.
        """
        metadata = {
            "index",
            "file_index",
            "file_entry_index",
            "source_file_name",
            "source_file_size",
            "source_file_mtime_ns",
            "source_file_entry_index",
        }
        # Metadata may legitimately be repeated across shards, whereas model
        # products must have exactly one selected owner.
        merged: dict[str, Any] = {}
        for entry in entries.values():
            for key, value in entry.items():
                if key in merged:
                    if key not in metadata:
                        raise ValueError(
                            f"Cache product '{key}' is exposed by multiple selected stages."
                        )
                    if merged[key] != value:
                        raise ValueError(
                            f"Cache stages disagree on event metadata '{key}'."
                        )
                    continue
                merged[key] = value
        return merged

    def process_entry_list(self, *args: Any, **kwargs: Any) -> None:
        """Apply one entry projection to every selected stage reader.

        Parameters
        ----------
        *args : Any
            Positional entry-selection arguments accepted by
            :meth:`ReaderBase.process_entry_list`.
        **kwargs : Any
            Named entry-selection arguments accepted by the same method.

        Notes
        -----
        All stage readers share an event axis by manifest contract. Applying
        the identical projection preserves that alignment.
        """
        for reader in self._readers.values():
            reader.process_entry_list(*args, **kwargs)
        reference = next(iter(self._readers.values()))
        self.entry_index = reference.entry_index.copy()
        self.eligible_entry_index = reference.eligible_entry_index.copy()

    def close(self) -> None:
        """Close all process-local physical shard handles.

        This method delegates to every selected stage reader and may be called
        repeatedly during explicit cleanup or object destruction.
        """
        for reader in self._readers.values():
            reader.close()

    def _collect_attribute(self, name: str) -> Any:
        """Collect one reader attribute across the selected stages.

        Parameters
        ----------
        name : str
            Attribute to retrieve from each physical stage reader.

        Returns
        -------
        Any
            The sole value for a single-stage reader, or a stage-keyed mapping
            when multiple stages contribute to the logical dataset.
        """
        values = {
            stage_name: getattr(reader, name, None)
            for stage_name, reader in self._readers.items()
        }
        if len(values) == 1:
            return next(iter(values.values()))
        return values
