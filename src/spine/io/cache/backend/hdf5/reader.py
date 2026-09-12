"""Stage-aware HDF5 cache reader."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, cast
from warnings import warn

import h5py
import numpy as np
import yaml
from yaml.parser import ParserError

from spine.io.filter import eligible_cache_entries_from_manifest
from spine.logging import logger

from ....read.hdf5 import HDF5Reader
from ....read.hdf5.common import (
    decode_string_attribute,
    require_dataset,
    require_group,
)
from ...manifest import CacheSource
from .common import source_id

__all__ = [
    "HDF5ShardReader",
    "inspect_stage_shard",
    "read_source_entry_index",
]

SOURCE_ENTRY_KEY = "source_file_entry_index"

StageConfig = dict[str, object]
StageConfigMap = dict[str, StageConfig | None]


class SourceInfo(TypedDict, total=False):
    """Describe source-file provenance stored once in a physical shard.

    Attributes
    ----------
    source_file_name : str
        Original file name recorded by the upstream reader.
    source_file_size : int
        Source-file size in bytes at cache-production time.
    source_file_mtime_ns : int
        Source-file modification timestamp in nanoseconds.

    Notes
    -----
    Fields are optional only to represent legacy shards without a ``source``
    group. Newly produced repository shards always store all three values.
    """

    source_file_name: str
    source_file_size: int
    source_file_mtime_ns: int


class HDF5ShardReader(HDF5Reader):
    """Read products stored under one or more stage groups in a cache file.

    The reader exposes the same event-level interface as :class:`HDF5Reader`,
    but resolves requested product keys under ``/stages/<stage>`` instead of
    the flat top-level namespace.
    """

    name = "cache"

    def __init__(
        self,
        stage: str | None = None,
        file_keys: str | list[str] | None = None,
        file_list: str | None = None,
        limit_num_files: int | None = None,
        max_print_files: int = 10,
        n_entry: int | None = None,
        n_skip: int | None = None,
        entry_list: list[int] | None = None,
        skip_entry_list: list[int] | None = None,
        build_classes: bool = True,
        skip_unknown_attrs: bool = False,
        allow_missing: bool = False,
        keep_open: bool = True,
        swmr: bool = False,
        ignore_incomplete: bool = False,
        stage_map: Mapping[str, str] | None = None,
        keys: Sequence[str] | None = None,
        entry_fraction_range: Sequence[float] | None = None,
        entry_filter: str | None = None,
        preserve_file_order: bool = False,
    ) -> None:
        """Initialize the cache-shard reader.

        Parameters
        ----------
        stage : str, optional
            Default stage from which to load products. If omitted, keys are
            searched across all stages and must resolve uniquely.
        stage_map : mapping, optional
            Explicit map from product keys to stage names. This overrides the
            default stage on a per-product basis.
        keys : sequence[str], optional
            Product keys that should be exposed by the reader. If omitted, all
            products from the selected stage(s) are exposed.
        file_keys, file_list, limit_num_files, max_print_files, n_entry, n_skip, \
        entry_list, skip_entry_list, build_classes, skip_unknown_attrs, \
        allow_missing, keep_open, swmr, ignore_incomplete, \
        entry_fraction_range, entry_filter : optional
            See :class:`spine.io.read.HDF5Reader`. These options control file
            discovery, entry selection, object reconstruction, file-handle
            lifetime, incomplete-stage handling, and source-manifest filtering.
        preserve_file_order : bool, default False
            Preserve the order of an explicit ``file_keys`` list instead of
            sorting paths. Cache repository projections use this to match the
            authoritative primary dataset's source order.
        """
        # Store routing policy before inspecting the available stage schemas
        self.stage = stage
        self.stage_map = dict(stage_map or {})
        self.requested_keys = tuple(keys) if keys is not None else None
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)
        if preserve_file_order:
            if file_list is not None or not isinstance(file_keys, list):
                raise ValueError(
                    "Preserving cache shard order requires an explicit file list."
                )
            requested_order = {path: index for index, path in enumerate(file_keys)}
            self.file_paths.sort(key=requested_order.__getitem__)
        self.keep_open = keep_open
        self.swmr = swmr
        self.ignore_incomplete = ignore_incomplete
        self._handle_pid = None
        self._file_handles = {}
        self.fixed_only = False
        self._initialize_product_backend()
        self._resolved_products: dict[int, dict[str, str]] = {}
        self._source_info: dict[int, SourceInfo] = {}
        self.file_format_versions: list[int] = []

        # Build the global event axis and resolve products independently per file
        file_index = []
        source_provenance: list[dict[str, object]] = []
        self.num_entries = 0
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as in_file:
                self.validate_stage_file(in_file, path)
                self.file_format_versions.append(2)
                self._source_info[i] = self.read_source_info(in_file)
                product_stage_map = self.resolve_product_stages(in_file, path)
                self._resolved_products[i] = product_stage_map
                stage_lengths = self.get_stage_lengths(in_file, path, product_stage_map)
                num_entries = self.validate_stage_lengths(path, stage_lengths)
                if entry_filter is not None:
                    source_provenance.extend(
                        self._read_stage_source_manifest_provenance(
                            in_file,
                            i,
                            num_entries,
                        )
                    )
                    self._clear_product_handles()
                file_index.append(i * np.ones(num_entries, dtype=np.int64))
                self.file_offsets[i] = self.num_entries
                self.num_entries += num_entries

        logger.info("Total number of entries in the file(s): %d\n", self.num_entries)

        self.file_index = (
            np.concatenate(file_index) if file_index else np.empty(0, dtype=np.int64)
        )
        self.run_info = None
        self.run_map = None

        # Apply the standard reader entry projection to the merged event axis
        eligible_entries = None
        if entry_filter is not None:
            eligible_entries = eligible_cache_entries_from_manifest(
                entry_filter,
                source_provenance,
            )
        self.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            None,
            None,
            allow_missing,
            entry_fraction_range,
            eligible_entries,
        )

        # Finish the inherited object reconstruction and file metadata setup
        self.build_classes = build_classes
        self.skip_unknown_attrs = skip_unknown_attrs
        self.cfg = self.process_cfg()
        self.version = self.process_version()

    def _read_stage_source_manifest_provenance(
        self,
        in_file: h5py.File,
        file_idx: int,
        num_entries: int,
    ) -> list[dict[str, object]]:
        """Read the canonical source identity of each cache-shard entry.

        A cache shard stores file identity once under ``/source`` and stores
        original entry indexes as a stage product. Both are required here;
        the legacy physical-index fallback is intentionally unsafe for mapping
        an external source manifest.

        Parameters
        ----------
        in_file : h5py.File
            Open cache shard.
        file_idx : int
            Cache-file index in this reader.
        num_entries : int
            Number of physical entries in the cache file.

        Returns
        -------
        list[dict]
            Per-entry source file identity and original source entry index.

        Raises
        ------
        KeyError
            If file identity or source-entry indexes were not persisted.
        """
        source_info = self._source_info[file_idx]
        missing = set(self.source_keys[:-1]) - set(source_info)
        if missing:
            raise KeyError(
                "Cannot apply an entry-filter manifest to a cache shard "
                f"without source identity; missing fields: {sorted(missing)}."
            )

        stage_names = set(self._resolved_products[file_idx].values())
        has_source_entries = any(
            SOURCE_ENTRY_KEY
            in self.get_stage_products(
                self.get_stage_group(in_file, self.file_paths[file_idx], stage_name)
            )
            for stage_name in stage_names
        )
        if not has_source_entries:
            raise KeyError(
                "Cannot apply an entry-filter manifest to a cache shard "
                f"without the '{SOURCE_ENTRY_KEY}' product."
            )

        source_entries = self._load_source_entry_indices(
            in_file,
            file_idx,
            0,
            num_entries,
        )
        return [
            {**source_info, SOURCE_ENTRY_KEY: source_entry}
            for source_entry in source_entries
        ]

    @classmethod
    def validate_stage_file(cls, in_file: h5py.File, path: str) -> None:
        """Require the internal cache-shard V2 container format.

        Parameters
        ----------
        in_file : h5py.File
            Open cache shard.
        path : str
            Cache path included in validation errors.

        Raises
        ------
        ValueError
            If the shard does not declare the cache format and HDF5 V2 schema.
        """
        if "info" not in in_file:
            raise ValueError(f"Cache shard '{path}' is missing its info group.")
        info = require_group(in_file, "info")
        version = int(info.attrs.get("format_version", 1))
        if version != 2:
            raise ValueError(
                f"Cache shard '{path}' uses HDF5 format version {version}; "
                "rebuild it with version 2."
            )
        file_format = decode_string_attribute(info.attrs.get("format"), "format")
        if file_format != cls.name:
            raise ValueError(
                f"Cache shard '{path}' has format '{file_format}', expected "
                f"'{cls.name}'."
            )
        cls.get_stages_group(in_file, path)

    @staticmethod
    def get_stages_group(in_file: h5py.File, path: str) -> h5py.Group:
        """Return the top-level ``stages`` group.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.
        path : str
            File path used to build informative error messages.

        Returns
        -------
        h5py.Group
            Top-level group containing the cache-shard products.
        """
        assert "stages" in in_file, f"Stage-cache file '{path}' is missing 'stages'."
        stages = in_file["stages"]
        assert isinstance(
            stages, h5py.Group
        ), f"'stages' in '{path}' must be a group, got {type(stages)}."
        return stages

    @classmethod
    def get_stage_group(cls, in_file: h5py.File, path: str, stage: str) -> h5py.Group:
        """Return one named stage group.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.
        path : str
            File path used to build informative error messages.
        stage : str
            Name of the stage group to load under ``/stages``.

        Returns
        -------
        h5py.Group
            Requested stage group.
        """
        stages = cls.get_stages_group(in_file, path)
        assert (
            stage in stages
        ), f"Stage-cache file '{path}' does not contain stage '{stage}'."
        stage_group = stages[stage]
        assert isinstance(
            stage_group, h5py.Group
        ), f"Stage '{stage}' in '{path}' must be a group, got {type(stage_group)}."
        return stage_group

    @staticmethod
    def read_source_info(in_file: h5py.File) -> SourceInfo:
        """Return top-level source provenance stored in the cache file.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.

        Returns
        -------
        dict[str, object]
            File-level provenance dictionary. If the cache predates the source
            group convention, this returns an empty dictionary.
        """
        if "source" not in in_file:
            return {}
        source_group = in_file["source"]
        assert isinstance(
            source_group, h5py.Group
        ), f"Expected 'source' to be a group, got {type(source_group)}."
        file_name = source_group.attrs["file_name"]
        if isinstance(file_name, bytes):
            file_name = file_name.decode()
        if not isinstance(file_name, str):
            raise TypeError("Source attribute 'file_name' must be a string.")

        # Provenance sizes are serialized as scalar integer attributes. Make
        # that contract explicit before converting NumPy integer scalars.
        file_size = HDF5ShardReader.read_integer_attribute(source_group, "file_size")
        file_mtime_ns = HDF5ShardReader.read_integer_attribute(
            source_group, "file_mtime_ns"
        )
        return {
            "source_file_name": file_name,
            "source_file_size": file_size,
            "source_file_mtime_ns": file_mtime_ns,
        }

    @staticmethod
    def read_integer_attribute(group: h5py.Group, name: str) -> int:
        """Return one required scalar integer group attribute.

        Parameters
        ----------
        group : h5py.Group
            Group containing the requested attribute.
        name : str
            Attribute name.

        Returns
        -------
        int
            Normalized Python integer value.

        Raises
        ------
        TypeError
            If the attribute is not a scalar integer.
        """
        value = np.asarray(group.attrs[name])
        if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"Source attribute '{name}' must be a scalar integer.")
        return int(value.item())

    @staticmethod
    def list_stage_names(stages: h5py.Group) -> tuple[str, ...]:
        """Return validated names from the top-level stage group.

        Parameters
        ----------
        stages : h5py.Group
            Top-level ``/stages`` namespace.

        Returns
        -------
        tuple[str, ...]
            Stage names in their physical HDF5 iteration order.

        Raises
        ------
        TypeError
            If the HDF5 backend exposes a non-string child name.
        """
        names = []
        for name in stages:
            if not isinstance(name, str):
                raise TypeError("HDF5 stage names must be strings.")
            names.append(name)
        return tuple(names)

    def list_stage_keys(self, stage_group: h5py.Group) -> tuple[str, ...]:
        """List product keys stored in one stage group.

        Products are exposed from the stage-local V2 ``products`` namespace;
        private reconstruction children remain nested under their owner.

        Parameters
        ----------
        stage_group : h5py.Group
            Stage group whose logical product keys should be listed.

        Returns
        -------
        tuple[str, ...]
            Stored logical product names.
        """
        return tuple(self.get_stage_products(stage_group).keys())

    @staticmethod
    def get_stage_products(stage_group: h5py.Group) -> h5py.Group:
        """Return the V2 logical-product root for one stage.

        Parameters
        ----------
        stage_group : h5py.Group
            Stage namespace containing a required ``products`` group.

        Returns
        -------
        h5py.Group
            Stage-local logical product namespace.
        """
        return require_group(stage_group, "products")

    def resolve_product_stages(self, in_file: h5py.File, path: str) -> dict[str, str]:
        """Resolve each requested product key to one stage.

        Resolution order is:

        1. explicit ``stage_map`` entry for the key
        2. dataset-level default ``stage``
        3. automatic discovery across all available stages

        Automatic discovery requires a unique match. If the same product name
        appears in multiple stages, the caller must disambiguate it.

        Parameters
        ----------
        in_file : h5py.File
            Open cache-shard file.
        path : str
            File path used in diagnostics.

        Returns
        -------
        dict[str, str]
            Mapping from exposed product names to their owning stages.

        Raises
        ------
        KeyError
            If a requested product cannot be found in its configured stage.
        ValueError
            If automatic discovery finds a product in multiple stages.
        """
        if (
            self.stage is not None
            and self.requested_keys is None
            and not self.stage_map
        ):
            stage_group = self.get_stage_group(in_file, path, self.stage)
            self.check_stage_complete(stage_group, path, self.stage)
            return {key: self.stage for key in self.list_stage_keys(stage_group)}

        # Derive the complete logical key set when no projection was requested
        stages = self.get_stages_group(in_file, path)
        stage_names = self.list_stage_names(stages)
        required_keys = (
            tuple(self.requested_keys)
            if self.requested_keys is not None
            else tuple(
                key
                for stage_name in stage_names
                for key in self.list_stage_keys(
                    self.get_stage_group(in_file, path, stage_name)
                )
            )
        )

        # Apply explicit mappings first, then the default or unique discovery
        resolved: dict[str, str] = {}
        for key in required_keys:
            if key in self.stage_map:
                stage_name = self.stage_map[key]
                stage_group = self.get_stage_group(in_file, path, stage_name)
                self.check_stage_complete(stage_group, path, stage_name)
                if key not in self.get_stage_products(stage_group):
                    raise KeyError(
                        f"Requested product '{key}' does not exist in stage "
                        f"'{stage_name}' of '{path}'."
                    )
                resolved[key] = stage_name
                continue

            if self.stage is not None:
                stage_group = self.get_stage_group(in_file, path, self.stage)
                self.check_stage_complete(stage_group, path, self.stage)
                if key not in self.get_stage_products(stage_group):
                    raise KeyError(
                        f"Requested product '{key}' does not exist in stage "
                        f"'{self.stage}' of '{path}'."
                    )
                resolved[key] = self.stage
                continue

            # Without a configured stage, require exactly one owning stage
            candidates = []
            for stage_name in stage_names:
                stage_group = self.get_stage_group(in_file, path, stage_name)
                if key in self.get_stage_products(stage_group):
                    self.check_stage_complete(stage_group, path, stage_name)
                    candidates.append(stage_name)

            if not candidates:
                raise KeyError(
                    f"Could not find requested product '{key}' in any stage of '{path}'."
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Requested product '{key}' appears in multiple stages of '{path}': "
                    f"{candidates}. Specify its stage explicitly."
                )

            resolved[key] = candidates[0]

        return resolved

    def check_stage_complete(
        self, stage_group: h5py.Group, path: str, stage: str
    ) -> None:
        """Reject incomplete stages unless explicitly allowed.

        Parameters
        ----------
        stage_group : h5py.Group
            Resolved stage group.
        path : str
            Cache file path.
        stage : str
            Stage name used in the error message.

        Raises
        ------
        RuntimeError
            If the stage is marked incomplete and ``ignore_incomplete`` is
            disabled.
        """
        if (
            "info" in stage_group
            and "complete" in stage_group["info"].attrs
            and not stage_group["info"].attrs["complete"]
            and not self.ignore_incomplete
        ):
            raise RuntimeError(
                f"Stage '{stage}' in '{path}' is marked incomplete. "
                "Pass ignore_incomplete=True to override."
            )

    def get_stage_lengths(
        self, in_file: h5py.File, path: str, product_stage_map: Mapping[str, str]
    ) -> dict[str, int]:
        """Return the event count of each referenced stage.

        Parameters
        ----------
        in_file : h5py.File
            Open cache file handle.
        path : str
            Cache file path.
        product_stage_map : mapping
            Mapping from requested raw product key to resolved stage name.

        Returns
        -------
        dict[str, int]
            Event count for every referenced stage.
        """
        stage_lengths: dict[str, int] = {}
        for stage_name in set(product_stage_map.values()):
            stage_group = self.get_stage_group(in_file, path, stage_name)
            events = stage_group["events"]
            assert isinstance(
                events, h5py.Dataset
            ), f"Stage '{stage_name}' in '{path}' is missing an 'events' dataset."
            stage_lengths[stage_name] = len(events)
        return stage_lengths

    @staticmethod
    def validate_stage_lengths(path: str, stage_lengths: Mapping[str, int]) -> int:
        """Ensure all referenced stages in one file have the same length.

        Parameters
        ----------
        path : str
            Cache path used in mismatch diagnostics.
        stage_lengths : mapping
            Event counts keyed by stage name.

        Returns
        -------
        int
            Shared number of entries across all referenced stages.

        Raises
        ------
        ValueError
            If referenced stages expose different event counts.
        """
        lengths = list(stage_lengths.values())
        if not lengths:
            return 0
        if any(length != lengths[0] for length in lengths[1:]):
            raise ValueError(
                f"Referenced stages in '{path}' do not expose the same number of entries: "
                f"{dict(stage_lengths)}."
            )
        return lengths[0]

    def process_cfg(self) -> StageConfig | StageConfigMap | None:
        """Return the stored configuration for the referenced stage(s), if any.

        Returns
        -------
        dict[str, object] or dict[str, dict or None] or None
            Parsed YAML configuration stored under stage metadata. A single
            stage yields its configuration directly; multiple stages return a
            mapping from stage name to configuration.
        """
        with h5py.File(self.file_paths[0], "r") as in_file:
            # Decode one configuration per stage referenced by the first file
            stage_names = sorted(set(self._resolved_products[0].values()))
            cfg_map: StageConfigMap = {}
            for stage_name in stage_names:
                stage_group = self.get_stage_group(
                    in_file, self.file_paths[0], stage_name
                )
                if "info" not in stage_group or "cfg" not in stage_group["info"].attrs:
                    cfg_map[stage_name] = None
                    continue
                cfg_str = stage_group["info"].attrs["cfg"]
                try:
                    if not isinstance(cfg_str, str):
                        raise TypeError("Stage 'cfg' attribute must be a string.")
                    cfg = yaml.safe_load(cfg_str)
                    if cfg is not None and not isinstance(cfg, dict):
                        raise TypeError("Stage configuration must decode to a mapping.")
                    if cfg is not None and not all(isinstance(key, str) for key in cfg):
                        raise TypeError("Stage configuration keys must be strings.")
                    cfg_map[stage_name] = cfg
                except ParserError:
                    warn(
                        "Parsing stage configuration failed, returning None for "
                        f"stage '{stage_name}'."
                    )
                    cfg_map[stage_name] = None

        # Preserve the simple flat-reader interface for single-stage caches
        if len(cfg_map) == 1:
            return next(iter(cfg_map.values()))
        return cfg_map

    def _load_entry(
        self,
        idx: int,
        file_idx: int,
        entry_idx: int,
        in_file: h5py.File,
    ) -> dict[str, object]:
        """Decode one resolved cache-shard entry from an open file.

        Parameters
        ----------
        idx : int
            User-facing reader index written into the returned metadata.
        file_idx : int
            Index of the physical cache-shard file containing the event.
        entry_idx : int
            Event index local to that physical file.
        in_file : h5py.File
            Open readable handle for ``file_idx``.

        Returns
        -------
        dict[str, object]
            Raw merged event dictionary containing standard metadata plus all
            requested stage products for the selected entry.

        Notes
        -----
        Handle ownership remains with the inherited scalar or batch access
        method. This decoder only reads from ``in_file`` and never closes it.
        """
        # Load the raw source entry index for this compact cache event. This is
        # administrative metadata rather than a user-facing product, so it is
        # always read even if the user did not request it.
        source_entry_idx = self._load_source_entry_indices(
            in_file,
            file_idx,
            entry_idx,
            entry_idx + 1,
        )[0]

        # Keep the compact cache position distinct from its raw source entry.
        data: dict[str, object] = {
            "file_index": file_idx,
            "file_entry_index": entry_idx,
            SOURCE_ENTRY_KEY: source_entry_idx,
        }
        data.update(self._source_info.get(file_idx, {}))
        product_stage_map = self._resolved_products[file_idx]

        # Read each physical stage once and select only its resolved products
        for stage_name in sorted(set(product_stage_map.values())):
            stage_group = self.get_stage_group(
                in_file, self.file_paths[file_idx], stage_name
            )
            products = self.get_stage_products(stage_group)
            for key, resolved_stage in product_stage_map.items():
                if resolved_stage == stage_name and key != SOURCE_ENTRY_KEY:
                    self.load_product(products, entry_idx, data, key)
            self.reconstruct_products(products, entry_idx, data)

        # Expose the user-facing global entry after all stage products are merged
        data["index"] = idx
        return data

    def _load_v2_run(
        self,
        file_idx: int,
        entries: list[tuple[int, int, int]],
        in_file: h5py.File,
    ) -> list[dict[str, object]]:
        """Decode one contiguous run from the resolved stage products.

        This is the cache-shard counterpart to the flat V2 reader path. Each
        selected stage product, including its private reconstruction children,
        is loaded once for the complete event run.

        Parameters
        ----------
        file_idx : int
            Index of the physical shard containing the run.
        entries : list[tuple[int, int, int]]
            Contiguous run descriptors containing file index, user-facing
            index, and physical file-entry index.
        in_file : h5py.File
            Open readable handle for the physical shard.

        Returns
        -------
        list[dict[str, object]]
            Decoded events in the same order as ``entries``.
        """
        # Load the raw source entry indexes for the entire contiguous run. This is
        # administrative metadata rather than a user-facing product, so it is
        # always read even if the user did not request it.
        first = entries[0][2]
        last = entries[-1][2] + 1
        source_entry_indices = self._load_source_entry_indices(
            in_file,
            file_idx,
            first,
            last,
        )
        data: list[dict[str, object]] = []
        for (_, idx, entry_idx), source_entry_idx in zip(entries, source_entry_indices):
            event: dict[str, object] = {
                "file_index": file_idx,
                "file_entry_index": entry_idx,
                SOURCE_ENTRY_KEY: source_entry_idx,
            }
            event.update(self._source_info.get(file_idx, {}))
            data.append(event)

        # Read each physical stage once and select only its resolved products
        product_stage_map = self._resolved_products[file_idx]
        for stage_name in sorted(set(product_stage_map.values())):
            stage_group = self.get_stage_group(
                in_file, self.file_paths[file_idx], stage_name
            )
            products = self.get_stage_products(stage_group)
            for key, resolved_stage in product_stage_map.items():
                if resolved_stage == stage_name and key != SOURCE_ENTRY_KEY:
                    self.load_product_many(products, first, last, data, key)
            self.reconstruct_products_many(products, first, last, data)

        # Stored products cannot override the reader-facing global index.
        for (_, idx, _), event in zip(entries, data):
            event["index"] = idx
        return data

    def _load_source_entry_indices(
        self,
        in_file: h5py.File,
        file_idx: int,
        first: int,
        last: int,
    ) -> list[int]:
        """Load and validate raw-source entry indexes for a cache event run.

        Source entry provenance is administrative metadata rather than an
        optional model product. It must therefore be read even when the HDF5
        dataset projects a narrow product schema. Every referenced stage that
        stores the provenance axis must agree with its siblings.

        Parameters
        ----------
        in_file : h5py.File
            Open cache-shard file containing the requested event run.
        file_idx : int
            Index of the physical cache file in the reader file list.
        first, last : int
            Inclusive-exclusive compact event range within the cache file.

        Returns
        -------
        list[int]
            Original source-file entry index for every compact cache event.
            Physical cache indexes are returned for older stages which do not
            persist this provenance product.

        Raises
        ------
        TypeError
            If a stored source-entry product is not scalar integer data.
        ValueError
            If multiple referenced stages report different source entries.
        """
        source_entries: np.ndarray | None = None
        source_stage: str | None = None
        product_stage_map = self._resolved_products[file_idx]

        for stage_name in sorted(set(product_stage_map.values())):
            stage_group = self.get_stage_group(
                in_file, self.file_paths[file_idx], stage_name
            )
            products = self.get_stage_products(stage_group)
            if SOURCE_ENTRY_KEY not in products:
                continue

            # Reuse the contiguous V2 product reader for the provenance axis.
            stage_data: list[dict[str, object]] = [{} for _ in range(last - first)]
            self.load_product_many(
                products,
                first,
                last,
                stage_data,
                SOURCE_ENTRY_KEY,
            )
            stage_entries = np.asarray(
                [event[SOURCE_ENTRY_KEY] for event in stage_data]
            )
            if stage_entries.ndim != 1 or not np.issubdtype(
                stage_entries.dtype, np.integer
            ):
                raise TypeError(
                    f"Stage '{stage_name}' in '{self.file_paths[file_idx]}' must "
                    f"store '{SOURCE_ENTRY_KEY}' as scalar integers."
                )
            stage_entries = stage_entries.astype(np.int64, copy=False)

            if source_entries is None:
                source_entries = stage_entries
                source_stage = stage_name
            elif not np.array_equal(source_entries, stage_entries):
                raise ValueError(
                    f"Stages '{source_stage}' and '{stage_name}' in "
                    f"'{self.file_paths[file_idx]}' report different "
                    f"'{SOURCE_ENTRY_KEY}' values for cache entries "
                    f"[{first}, {last})."
                )

        if source_entries is None:
            return np.arange(first, last, dtype=np.int64).tolist()
        return source_entries.tolist()


def inspect_stage_shard(path: str, stage: str) -> tuple[CacheSource, tuple[str, ...]]:
    """Validate a completed shard and return its source and product schema.

    Parameters
    ----------
    path : str
        Physical HDF5 shard path.
    stage : str
        Stage expected inside the shard.

    Returns
    -------
    tuple[CacheSource, tuple[str, ...]]
        Immutable source record and sorted public product schema.

    Raises
    ------
    RuntimeError
        If the requested stage was not finalized successfully.
    ValueError
        If the file does not use the cache-shard V2 format.
    """
    with h5py.File(path, "r") as shard:
        HDF5ShardReader.validate_stage_file(shard, path)
        source_info = HDF5ShardReader.read_source_info(shard)
        file_name = source_info.get("source_file_name")
        file_size = source_info.get("source_file_size")
        file_mtime_ns = source_info.get("source_file_mtime_ns")
        if file_name is None or file_size is None or file_mtime_ns is None:
            raise ValueError(
                f"Pending cache shard '{path}' has incomplete source provenance."
            )

        normalized = {
            "file_name": file_name,
            "file_size": file_size,
            "file_mtime_ns": file_mtime_ns,
        }
        shard_source_id = source_id(normalized)
        stage_group = HDF5ShardReader.get_stage_group(shard, path, stage)
        info = stage_group.get("info")
        if not isinstance(info, h5py.Group) or not bool(
            info.attrs.get("complete", False)
        ):
            raise RuntimeError(f"Pending cache shard '{path}' is not complete.")

        # Administrative event-axis products are part of the physical schema,
        # but are not advertised as stage outputs in the manifest.
        product_names: list[str] = []
        for raw_key in HDF5ShardReader.get_stage_products(stage_group):
            # h5py group iteration is documented to yield names, although its
            # type stubs retain the possibility of an absent link.
            key = cast(str, raw_key)
            if key not in ("index", SOURCE_ENTRY_KEY):
                product_names.append(key)
        products = tuple(sorted(product_names))
        events = require_dataset(stage_group, "events")
        source = CacheSource(
            id=shard_source_id,
            file_name=file_name,
            file_size=file_size,
            file_mtime_ns=file_mtime_ns,
            num_entries=len(events),
        )
        return source, products


def read_source_entry_index(path: str, stage: str) -> np.ndarray:
    """Read the compact-to-source entry axis from one validated V2 shard.

    Parameters
    ----------
    path : str
        Physical HDF5 shard path.
    stage : str
        Stage containing the provenance product.

    Returns
    -------
    numpy.ndarray
        One-dimensional compact-to-source entry mapping. For manually imported
        V2 shards without explicit provenance, this is the physical event axis.
    """
    with h5py.File(path, "r") as shard:
        stage_group = HDF5ShardReader.get_stage_group(shard, path, stage)
        products = HDF5ShardReader.get_stage_products(stage_group)
        if SOURCE_ENTRY_KEY not in products:
            # Repository shards normally persist provenance. Retain a physical
            # index fallback so manually imported V2 shards remain legible.
            events = require_dataset(stage_group, "events")
            return np.arange(len(events), dtype=np.int64)

        product = products[SOURCE_ENTRY_KEY]
        assert isinstance(product, h5py.Group)
        return np.asarray(product["values"], dtype=np.int64)
