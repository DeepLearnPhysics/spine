"""Private HDF5 shard writer and stage-serialization orchestration."""

from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from typing import Any

import h5py
import numpy as np
import yaml

from ....write.hdf5 import HDF5Writer
from ....write.hdf5.common import decode_string_attribute, require_group
from .common import source_id
from .file import StageFileMixin
from .state import StageState

__all__ = ["HDF5ShardWriter"]


class HDF5ShardWriter(StageFileMixin, HDF5Writer):
    """Serialize one cache stage into one HDF5 V2 shard per source file.

    The public cache writer gives this backend a transaction-private directory.
    Incoming batches are partitioned by immutable source provenance, and each
    partition is written under ``/stages/<stage>`` in its own shard. The
    repository manifest—not this physical writer—composes independently
    produced stage generations into one logical dataset.

    Unlike :class:`HDF5Writer`, this class does not use one flat product
    namespace. The stage-local namespace keeps the physical schema explicit
    and lets the shard reader reuse the normal V2 product codec.
    """

    name = "cache"
    _file_source_keys = {"source_file_name", "source_file_size", "source_file_mtime_ns"}

    def __init__(
        self,
        file_name: str | None = None,
        directory: str | None = None,
        prefix: str | list[str] | None = None,
        suffix: str = "stage",
        stage: str | None = None,
        keys: list[str] | None = None,
        skip_keys: list[str] | None = None,
        split: bool = True,
        lite: bool = False,
        keep_open: bool = True,
        flush_frequency: int | None = None,
        overwrite: bool = False,
        overwrite_stage: bool = False,
        source_id_names: bool = False,
    ) -> None:
        """Initialize the cache-shard writer.

        Parameters
        ----------
        file_name : str, optional
            Output cache file name. When ``directory`` is not provided, this
            path also provides the parent directory for source-derived cache
            files. If omitted, the base output path is built from ``prefix``
            and ``suffix`` using the same naming rules as :class:`HDF5Writer`.
        directory : str, optional
            Output directory used for all source-derived cache files. When
            provided, it overrides the directory encoded in ``file_name``.
        prefix : str or list[str], optional
            Input file prefix used to derive the base cache-shard file name
            when ``file_name`` is not specified.
        suffix : str, default "stage"
            Suffix appended to source file basenames when deriving split cache
            file names.
        stage : str, optional
            Stage name to use for the standard driver-facing writer contract.
            When provided, :meth:`__call__` writes to this stage and
            :meth:`finalize` marks it complete. If omitted, use
            :meth:`write_stage` and :meth:`finalize_stage` directly.
        keys : list[str], optional
            List of data-product keys to persist in each stage. If omitted,
            store every product present in the batch apart from administrative
            source-file metadata.
        skip_keys : list[str], optional
            List of data-product keys to exclude from each stage.
        split : bool, default True
            Cache shards are always written one file per source file. This
            argument is accepted for compatibility with generic writer
            configuration, but it must remain `True`.
        lite : bool, default False
            If `True`, store lite object representations when applicable
        keep_open : bool, default True
            If `True`, keep one append handle open per process
        flush_frequency : int, optional
            Flush the file after this many appended entries per stage. If
            `None`, only flush on explicit requests or close/finalize.
        overwrite : bool, default False
            If `True`, replace the entire cache file if it already exists.
        overwrite_stage : bool, default False
            If `True`, replace a completed stage encountered in a directly
            managed shard. Repository transactions normally write fresh files
            and handle logical stage replacement at manifest publication.
        source_id_names : bool, default False
            If `True`, name shards from their immutable source identities.
            The cache repository enables this mode for transaction-private
            output; direct codec tests may retain an explicitly configured
            file name.
        Notes
        -----
        Cache shards exclusively use the offset-based HDF5 V2 product layout.
        They are internal, reproducible artifacts, so incompatible shards must
        be rebuilt rather than upgraded in place.
        """
        # Validate the configuration before initializing the shared HDF5 backend
        if not split:
            raise ValueError(
                "The cache HDF5 backend requires `split=True` because shards "
                "are written one file per source file."
            )
        # Reuse the shared V2 product codec while leaving shard lifecycle and
        # stage completion to this backend. ``append=True`` prevents the flat
        # writer constructor from claiming the transaction-private path.
        name_split = split if prefix is not None else False
        super().__init__(
            file_name=file_name,
            directory=directory,
            prefix=prefix,
            suffix=suffix,
            keys=keys,
            skip_keys=skip_keys,
            overwrite=False,
            append=True,
            split=name_split,
            lite=lite,
            keep_open=keep_open,
            flush_frequency=flush_frequency,
            format_version=self.current_format_version,
        )

        self.file_name = self.file_names[0]
        self._route_by_source = isinstance(prefix, list) and len(prefix) > 1
        self.directory = directory
        self.suffix = suffix

        # Add stage-specific routing and publication policy.
        self.stage = stage
        self.overwrite_stage = overwrite_stage
        self.source_id_names = source_id_names
        self.source_info: dict[str, Any] | None = None

        self._configured_keys = None if self.keys is None else set(self.keys)
        # Cache shards always route by source, even when the initial base name
        # was resolved from a single prefix.
        self.split = True

        # Track schemas, completion state, and handles independently per file
        self._handles: dict[str, h5py.File] = {}
        self._initialized_files: set[str] = set()
        self._stage_states: dict[str, StageState] = {}
        self._completed_stages: dict[str, set[str]] = defaultdict(set)
        self._active_stages: set[tuple[str, str]] = set()
        self._known_files: set[str] = set()

        if overwrite and os.path.exists(self.file_name):
            os.remove(self.file_name)

    def __call__(self, data: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
        """Append one batch to the configured stage.

        Parameters
        ----------
        data : dict
            Scalar or batched products with reader-provided source provenance.
        cfg : dict, optional
            Complete SPINE configuration stored with the stage metadata.

        Raises
        ------
        RuntimeError
            If no default stage was configured for the standard writer path.
        """
        if self.stage is None:
            raise RuntimeError(
                "The cache HDF5 backend requires a configured `stage` "
                "through the standard writer call path."
            )

        self.write_stage(
            self.stage,
            data,
            cfg=cfg,
            overwrite_stage=self.overwrite_stage,
        )

    def finalize(self) -> None:
        """Mark the configured stage complete across all touched shards.

        Raises
        ------
        RuntimeError
            If no default stage was configured for the standard writer path.

        Notes
        -----
        Physical completion is distinct from repository publication. The
        public cache writer validates these finalized shards before committing
        their generation to the manifest.
        """
        if self.stage is None:
            raise RuntimeError(
                "The cache HDF5 backend requires a configured `stage` to finalize "
                "through the standard writer interface."
            )

        self.finalize_stage(self.stage)

    def _prepare_batch(
        self,
        data: dict[str, Any],
        state: StageState | None,
    ) -> tuple[dict[str, Any], int, StageState]:
        """Normalize one batch and resolve its stage-local V2 schema.

        The inherited V2 preparation backend stores schema metadata on the
        writer instance. Cache shards project the selected stage into that
        state temporarily, then restore the driver-facing writer state before
        returning.

        Parameters
        ----------
        data : dict
            Raw scalar or batched products to prepare.
        state : StageState, optional
            Existing schema for this stage, if it has already been written.

        Returns
        -------
        tuple[dict[str, Any], int, StageState]
            Prepared batch, batch size, and resolved stage-local schema.
        """
        original_ready = self.ready
        original_keys = self.keys
        original_metadata = self.product_metadata
        original_children = self.product_children
        try:
            # Start from the configured public projection on every stage. V2
            # preparation may add private child keys owned by typed products.
            self.ready = state is not None
            self.keys = (
                None if self._configured_keys is None else set(self._configured_keys)
            )
            self.product_metadata = (
                {} if state is None else deepcopy(state.product_metadata)
            )
            self.product_children = (
                {} if state is None else dict(state.product_children)
            )

            prepared = self.with_source_provenance(data)
            prepared = self.prepare_products(prepared)

            # Normalize scalar entries to the list-like batch representation
            # consumed by the collective product append backend.
            if np.isscalar(prepared["index"]):
                for key in prepared:
                    prepared[key] = [prepared[key]]
                batch_size = 1
            else:
                batch_size = len(prepared["index"])

            if state is None:
                state = self._create_stage_state(prepared)
            return prepared, batch_size, state
        finally:
            self.ready = original_ready
            self.keys = original_keys
            self.product_metadata = original_metadata
            self.product_children = original_children

    def _create_stage_state(self, data: dict[str, Any]) -> StageState:
        """Infer the schema of one stage from the first written batch.

        Parameters
        ----------
        data : dict
            Normalized batch dictionary used as the schema template.

        Returns
        -------
        StageState
            Immutable product definitions and mutable append progress for the
            stage.
        """
        keys = self.get_stored_keys(data)
        if "source_file_entry_index" in data:
            keys.add("source_file_entry_index")
        keys.difference_update(self._file_source_keys)

        # Infer and retain one immutable serialization schema per named stage
        type_dict, object_dtypes = self.get_data_formats(data, keys)
        state = StageState(
            keys=keys,
            type_dict=type_dict,
            object_dtypes=object_dtypes,
            product_metadata=deepcopy(self.product_metadata),
            product_children=dict(self.product_children),
        )
        return state

    def get_output_path(
        self, source_info: dict[str, Any], multiple_sources: bool = False
    ) -> str:
        """Resolve the cache-file path for one source file.

        Parameters
        ----------
        source_info : dict
            File-level source identity returned by :meth:`get_batch_source_info`.
        multiple_sources : bool, default False
            If `True`, derive one output path from the source file basename.
            Otherwise reuse ``self.file_name`` directly unless this writer is
            already in source-routed mode.

        Returns
        -------
        str
            Destination path for the source-specific cache shard.

        Raises
        ------
        ValueError
            If source-identity naming is enabled without an output directory.
        """
        if self.source_id_names:
            if self.directory is None:
                raise ValueError(
                    "Source-identity shard naming requires an output directory."
                )
            return os.path.join(self.directory, f"{source_id(source_info)}.h5")

        if not (self._route_by_source or multiple_sources):
            if self.directory is None:
                return self.file_name
            return os.path.join(self.directory, os.path.basename(self.file_name))

        dir_name = (
            self.directory
            if self.directory is not None
            else os.path.dirname(self.file_name)
        )
        base_name = os.path.splitext(str(source_info["file_name"]))[0]
        return os.path.join(dir_name, f"{base_name}_{self.suffix}.h5")

    def split_batch_by_source(
        self, data: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        """Split one normalized batch into one subset per source file.

        Parameters
        ----------
        data : dict
            Normalized batch containing per-event source provenance.

        Returns
        -------
        list[tuple[str, dict, dict]]
            One tuple per source file containing the resolved output file path,
            the batch subset that belongs to that source file, and the
            file-level source provenance dictionary.

        Raises
        ------
        KeyError
            If required source-file provenance is absent from the batch.
        """
        required = ("source_file_name", "source_file_size", "source_file_mtime_ns")
        for key in required:
            if key not in data:
                raise KeyError(
                    "The cache HDF5 backend requires reader-provided source "
                    "provenance. "
                    f"Missing key: {key}."
                )

        # Group event positions by their complete source-file identity
        batch_size = len(data["index"])
        groups: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
        for batch_id in range(batch_size):
            groups[
                (
                    data["source_file_name"][batch_id],
                    data["source_file_size"][batch_id],
                    data["source_file_mtime_ns"][batch_id],
                )
            ].append(batch_id)

        multiple_sources = len(groups) > 1
        self._route_by_source = self._route_by_source or multiple_sources

        # Materialize one independently writable batch subset per source file
        result = []
        for (file_name, file_size, file_mtime_ns), batch_ids in groups.items():
            source_info = {
                "file_name": (
                    file_name.item() if hasattr(file_name, "item") else file_name
                ),
                "file_size": int(
                    file_size.item() if hasattr(file_size, "item") else file_size
                ),
                "file_mtime_ns": int(
                    file_mtime_ns.item()
                    if hasattr(file_mtime_ns, "item")
                    else file_mtime_ns
                ),
            }
            subset = {}
            for key, value in data.items():
                if np.isscalar(value):
                    subset[key] = value
                    continue
                subset[key] = [value[i] for i in batch_ids]
            result.append(
                (
                    self.get_output_path(source_info, multiple_sources),
                    subset,
                    source_info,
                )
            )

        return result

    def _ensure_stage_group(
        self,
        out_file: h5py.File,
        file_path: str,
        stage: str,
        state: StageState,
        cfg: dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
        overwrite_stage: bool = False,
    ) -> h5py.Group:
        """Create or fetch one stage group.

        Parameters
        ----------
        out_file : h5py.File
            Open cache-file handle.
        file_path : str
            Output cache-file path used for error messages and bookkeeping.
        stage : str
            Stage name to create or reopen.
        state : StageState
            Inferred schema state for the stage.
        cfg : dict, optional
            Stage configuration to serialize into metadata.
        attrs : dict, optional
            Additional stage metadata attributes.
        overwrite_stage : bool, default False
            If `True`, delete any existing stage group and rebuild it.

        Returns
        -------
        h5py.Group
            New or reopened incomplete stage group ready for appends.

        Raises
        ------
        RuntimeError
            If the stage is already complete and replacement was not enabled.
        """
        stages = out_file["stages"]
        assert isinstance(stages, h5py.Group), "'stages' must be an HDF5 group."

        # Recovery and explicit replacement apply only when this writer first
        # encounters a file-stage pair. Later batches append to the active
        # incomplete stage normally, even when overwrite_stage is configured.
        stage_key = (file_path, stage)
        first_use = stage_key not in self._active_stages
        if stage in stages:
            stage_group = stages[stage]
            assert isinstance(
                stage_group, h5py.Group
            ), f"Stage '{stage}' is expected to be a group, got {type(stage_group)}."
            info = stage_group.get("info")
            complete = bool(
                isinstance(info, h5py.Group) and info.attrs.get("complete", False)
            )

            replace = (first_use and not complete) or (overwrite_stage and complete)
            if replace:
                del stages[stage]
                self._completed_stages[file_path].discard(stage)
            elif complete:
                raise RuntimeError(
                    f"Stage '{stage}' is already complete in '{file_path}'. "
                    "Set overwrite_stage=True to rebuild it."
                )

        if stage not in stages:
            # A new stage owns its metadata and complete V2 product schema.
            stage_group = stages.create_group(stage)
            info = stage_group.create_group("info")
            info.attrs["complete"] = False
            if cfg is not None:
                info.attrs["cfg"] = yaml.dump(cfg)
            if attrs is not None:
                for key, value in attrs.items():
                    info.attrs[key] = value

            self.type_dict = state.type_dict
            self.event_dtype = state.event_dtype
            self.product_metadata = state.product_metadata
            self.product_children = state.product_children
            self.initialize_product_datasets(stage_group, state.type_dict)
            state.event_dtype = self.event_dtype
            self._active_stages.add(stage_key)
            return stage_group

        stage_group = stages[stage]
        assert isinstance(
            stage_group, h5py.Group
        ), f"Stage '{stage}' is expected to be a group, got {type(stage_group)}."
        self._validate_stage_schema(stage_group, file_path, stage, state)

        # Reopened stages may refresh metadata but remain incomplete until finalized
        if "info" in stage_group and attrs is not None:
            for key, value in attrs.items():
                stage_group["info"].attrs[key] = value
        if "info" in stage_group and cfg is not None:
            stage_group["info"].attrs["cfg"] = yaml.dump(cfg)

        stage_group["info"].attrs["complete"] = False
        self._active_stages.add(stage_key)
        return stage_group

    @staticmethod
    def _validate_stage_schema(
        stage_group: h5py.Group,
        file_path: str,
        stage: str,
        state: StageState,
    ) -> None:
        """Validate an active stage against its in-memory V2 schema.

        Parameters
        ----------
        stage_group : h5py.Group
            Existing stage group being reopened for append.
        file_path : str
            Cache path used in validation errors.
        stage : str
            Stage name used in validation errors.
        state : StageState
            Expected physical and typed-product schema.
        """
        if "events" not in stage_group:
            raise ValueError(
                f"Stage '{stage}' in '{file_path}' is missing its V2 event axis."
            )
        products = require_group(stage_group, "products")
        expected_products = set(state.type_dict).difference(state.product_children)
        if set(products) != expected_products:
            raise ValueError(
                f"Stage '{stage}' in '{file_path}' has a different product schema."
            )

        # Typed products carry reconstruction metadata and private child groups.
        for key, metadata in state.product_metadata.items():
            product = require_group(products, key)
            if "product_metadata" not in product.attrs:
                raise ValueError(
                    f"Stage '{stage}' product '{key}' is missing V2 metadata."
                )
            encoded = decode_string_attribute(
                product.attrs["product_metadata"], "product_metadata"
            )
            expected_metadata = yaml.safe_load(yaml.safe_dump(metadata))
            if yaml.safe_load(encoded) != expected_metadata:
                raise ValueError(
                    f"Stage '{stage}' product '{key}' has a different schema."
                )

        for parent, name in state.product_children.values():
            parent_group = require_group(products, parent)
            if name not in parent_group:
                raise ValueError(
                    f"Stage '{stage}' product '{parent}' is missing child '{name}'."
                )

    def write_stage(
        self,
        stage: str,
        data: dict[str, Any],
        cfg: dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
        overwrite_stage: bool = False,
    ) -> None:
        """Append one batch of products to a named stage.

        Parameters
        ----------
        stage : str
            Stage group name under ``/stages``
        data : dict
            Dictionary of batched data products
        cfg : dict, optional
            Configuration to store alongside this stage
        attrs : dict, optional
            Additional stage metadata to persist under ``stage/info.attrs``
        overwrite_stage : bool, default False
            If `True`, delete any existing stage group with the same name and
            rebuild it from the provided data.

        Notes
        -----
        The input batch may span multiple source files. In that case the batch
        is partitioned by source provenance and written into one cache file per
        source file automatically.
        """
        # Normalize once and establish the stage schema from its first batch.
        state = self._stage_states.get(stage)
        normalized, _, state = self._prepare_batch(data, state)
        self._stage_states[stage] = state

        # Temporarily project the inherited flat writer onto this stage schema
        original_keys = self.keys
        original_type_dict = self.type_dict
        original_object_dtypes = self.object_dtypes
        original_product_metadata = self.product_metadata
        original_product_children = self.product_children
        original_event_dtype = self.event_dtype
        try:
            self.keys = state.keys
            self.type_dict = state.type_dict
            self.object_dtypes = state.object_dtypes
            self.product_metadata = state.product_metadata
            self.product_children = state.product_children
            self.event_dtype = state.event_dtype

            for target_path, subset, source_info in self.split_batch_by_source(
                normalized
            ):
                file_path = self._get_stage_write_path(
                    target_path,
                    stage,
                    source_info,
                    overwrite_stage,
                )
                out_file, should_close = self._open_handle(file_path)
                try:
                    self.ensure_source_group(out_file, subset, file_path)
                    stage_group = self._ensure_stage_group(
                        out_file,
                        file_path,
                        stage,
                        state,
                        cfg=cfg,
                        attrs=attrs,
                        overwrite_stage=overwrite_stage,
                    )

                    # Append the routed events collectively through the V2
                    # offset backend to minimize resize and write operations.
                    batch_ids = np.arange(len(subset["index"]), dtype=np.int64)
                    self.append_product_entries(stage_group, subset, batch_ids)

                    state.event_dtype = self.event_dtype
                    if self.flush_frequency is not None:
                        state.entries_since_flush += len(subset["index"])
                        if state.entries_since_flush >= self.flush_frequency:
                            out_file.flush()
                            state.entries_since_flush = 0
                finally:
                    if should_close:
                        out_file.close()
        finally:
            # Restore the driver-facing writer state after stage-local writes
            self.keys = original_keys
            self.type_dict = original_type_dict
            self.object_dtypes = original_object_dtypes
            self.product_metadata = original_product_metadata
            self.product_children = original_product_children
            self.event_dtype = original_event_dtype

    def finalize_stage(self, stage: str) -> None:
        """Finalize one stage in every touched cache file.

        Parameters
        ----------
        stage : str
            Stage name to finalize across all cache files written by this
            writer instance.

        Notes
        -----
        The repository transaction publishes these shards only after every
        touched file has been marked complete and validated.
        """
        for file_path in sorted(self._known_files):
            if (file_path, stage) not in self._active_stages:
                continue

            out_file, should_close = self._open_handle(file_path)
            try:
                stages = out_file["stages"]
                assert isinstance(stages, h5py.Group), "'stages' must be an HDF5 group."
                if stage not in stages:
                    continue

                # Completion is committed only after all preceding writes flush
                stage_group = stages[stage]
                assert isinstance(stage_group, h5py.Group)
                info = stage_group["info"]
                assert isinstance(info, h5py.Group)
                info.attrs["complete"] = True
                out_file.flush()
                self._completed_stages[file_path].add(stage)
            finally:
                if should_close:
                    out_file.close()
