"""Stage-aware HDF5 cache writer."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import yaml

from spine.version import __version__

from .hdf5 import HDF5Writer

__all__ = ["StageHDF5Writer"]


class StageHDF5Writer(HDF5Writer):
    """Write additive stage caches to one HDF5 file per source file.

    This writer is intended for sequential cache materialization workflows
    where each processing stage writes a self-contained set of products under
    ``/stages/<stage>`` while preserving previously completed stages. Cache
    files are split by source-file provenance automatically.

    Unlike :class:`HDF5Writer`, this class does not use one flat product
    namespace for the entire file. Each stage owns its own ``events`` dataset
    and product datasets, which allows failed later stages to be rewritten
    without modifying earlier completed stages.
    """

    name = "stage_hdf5"
    _file_source_keys = {"source_file_name", "source_file_size", "source_file_mtime_ns"}

    @dataclass
    class StageState:
        """In-memory description of one stage schema.

        The regular :class:`HDF5Writer` stores one flat schema for the whole
        file. Stage caches need one schema per stage, so this small dataclass
        carries the state required to keep appending consistently to a given
        stage group.
        """

        keys: set[str]
        type_dict: dict[str, HDF5Writer.DataFormat]
        object_dtypes: list[list[tuple[str, type]]]
        event_dtype: np.dtype | list[tuple[str, Any]] | None = None
        entries_since_flush: int = 0

    def __init__(
        self,
        file_name: str | None = None,
        directory: str | None = None,
        prefix: str | None = None,
        suffix: str = "stage",
        lite: bool = False,
        keep_open: bool = True,
        flush_frequency: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize the stage-cache writer.

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
        prefix : str, optional
            Input file prefix used to derive the base staged-cache file name
            when ``file_name`` is not specified.
        suffix : str, default "stage"
            Suffix appended to source file basenames when deriving split cache
            file names.
        lite : bool, default False
            If `True`, store lite object representations when applicable
        keep_open : bool, default True
            If `True`, keep one append handle open per process
        flush_frequency : int, optional
            Flush the file after this many appended entries per stage. If
            `None`, only flush on explicit requests or close/finalize.
        overwrite : bool, default False
            If `True`, replace the entire cache file if it already exists.
        """
        self.file_name = self.get_file_names(
            file_name=file_name,
            prefix=prefix,
            suffix=suffix,
            split=False,
            directory=directory,
        )[0]
        self.directory = directory
        self.suffix = suffix
        self.lite = lite
        self.keep_open = keep_open
        self.flush_frequency = flush_frequency
        self.source_info: dict[str, Any] | None = None

        self.keys = None
        self.skip_keys = None
        self.dummy_ds = None
        self.append = True
        self.split = False
        self.ready = False
        self.object_dtypes = []
        self.type_dict = None
        self.event_dtype = None

        self._cfg: dict[str, Any] | None = None
        self._handle_pid: int | None = None
        self._handles: dict[str, h5py.File] = {}
        self._initialized_files: set[str] = set()
        self._stage_states: dict[str, StageHDF5Writer.StageState] = {}
        self._completed_stages: dict[str, set[str]] = defaultdict(set)
        self._known_files: set[str] = set()

        if overwrite and os.path.exists(self.file_name):
            os.remove(self.file_name)

    def close(self) -> None:
        """Close any persistent cache-file handles.

        This only affects handles cached in the current process and may be
        called repeatedly.
        """
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass

        self._handles = {}
        self._handle_pid = None

    def _check_handle_pid(self) -> None:
        """Ensure persistent writer handles remain process-local.

        Stage caches are not safe to append to through a writer instance that
        has crossed a process boundary. This method enforces the same
        single-process handle ownership contract as :class:`HDF5Writer`.
        """
        current_pid = os.getpid()
        if self._handle_pid is None:
            self._handle_pid = current_pid
            return

        if self._handle_pid != current_pid:
            raise RuntimeError(
                "StageHDF5Writer file handles are process-local and cannot be "
                "reused across process boundaries."
            )

    def _open_handle(self, file_path: str) -> tuple[h5py.File, bool]:
        """Return an appendable cache-file handle for one output path.

        Returns
        -------
        tuple[h5py.File, bool]
            Open HDF5 handle and a flag indicating whether the caller is
            responsible for closing it immediately.
        """
        self._ensure_file(file_path)
        if not self.keep_open:
            return h5py.File(file_path, "a"), True

        self._check_handle_pid()
        handle = self._handles.get(file_path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(file_path, "a")
            self._handles[file_path] = handle

        return handle, False

    def _ensure_file(self, file_path: str) -> None:
        """Initialize one output cache file structure on first use.

        The top-level administrative groups are created lazily because staged
        cache files are derived from source provenance and may not all be
        touched by every write call.
        """
        if file_path in self._initialized_files:
            return

        mode = "a" if os.path.exists(file_path) else "w"
        if self.keep_open:
            self._check_handle_pid()
            out_file = h5py.File(file_path, mode)
            self._handles[file_path] = out_file
        else:
            out_file = h5py.File(file_path, mode)

        try:
            if "info" not in out_file:
                out_file.create_group("info")
            out_file["info"].attrs["version"] = __version__
            out_file["info"].attrs["format"] = self.name

            if "stages" not in out_file:
                out_file.create_group("stages")
        finally:
            if not self.keep_open:
                out_file.close()

        self._initialized_files.add(file_path)
        self._known_files.add(file_path)

    def get_batch_source_info(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract cache-file source provenance from one normalized batch.

        Parameters
        ----------
        data : dict
            Normalized batch dictionary prepared for writing.

        Returns
        -------
        dict[str, Any]
            File-level source identity stored under the cache file's top-level
            ``/source`` group.
        """
        required = ("source_file_name", "source_file_size", "source_file_mtime_ns")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                "StageHDF5Writer requires reader-provided source provenance. "
                f"Missing keys: {missing}."
            )

        values = {}
        for key in required:
            value = data[key]
            if np.isscalar(value):
                values[key] = value.item() if hasattr(value, "item") else value
                continue

            array = np.asarray(value)
            if array.ndim == 0:
                values[key] = array.item()
                continue
            if len(array) == 0:
                raise ValueError(f"Source provenance key '{key}' is empty.")

            first = array[0].item() if hasattr(array[0], "item") else array[0]
            if any(
                (el.item() if hasattr(el, "item") else el) != first for el in array[1:]
            ):
                raise ValueError(
                    "StageHDF5Writer expects one source file per cache file. "
                    f"Batch key '{key}' contains multiple values."
                )
            values[key] = first

        return {
            "file_name": values["source_file_name"],
            "file_size": int(values["source_file_size"]),
            "file_mtime_ns": int(values["source_file_mtime_ns"]),
        }

    def ensure_source_group(
        self, out_file: h5py.File, data: dict[str, Any], file_path: str
    ) -> None:
        """Create or validate the top-level source provenance group.

        This enforces the one-cache-file-per-source-file contract. If a later
        stage attempts to write into an existing cache file with mismatched
        source provenance, the writer raises immediately.
        """
        source_info = self.get_batch_source_info(data)
        self.source_info = source_info

        if "source" not in out_file:
            source_group = out_file.create_group("source")
            for key, value in source_info.items():
                source_group.attrs[key] = value
            return

        source_group = out_file["source"]
        assert isinstance(
            source_group, h5py.Group
        ), f"Expected 'source' to be a group, got {type(source_group)}."
        for key, value in source_info.items():
            cached_value = source_group.attrs.get(key)
            if cached_value != value:
                raise RuntimeError(
                    f"Cache source mismatch for '{file_path}': '{key}' differs "
                    f"({cached_value!r} != {value!r})."
                )

    def _prepare_batch(self, data: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Normalize one batch for stage writing.

        This mirrors the flat HDF5 writer behavior by accepting either scalar
        single-entry payloads or already batched payloads and returning a
        uniform list-like representation.
        """
        data = self.with_source_provenance(data)
        if np.isscalar(data["index"]):
            for key in data:
                data[key] = [data[key]]
            return data, 1

        return data, len(data["index"])

    def _create_stage_state(
        self, stage: str, data: dict[str, Any]
    ) -> StageHDF5Writer.StageState:
        """Infer the schema of one stage from the first written batch.

        Parameters
        ----------
        stage : str
            Stage name whose schema is being initialized.
        data : dict
            Normalized batch dictionary used as the schema template.
        """
        keys = {"index"}
        keys.update(data.keys())
        for key, source_key in self.source_index_keys.items():
            if key in data:
                keys.add(source_key)
        keys.difference_update(self._file_source_keys)
        if self.skip_keys is not None:
            keys.difference_update(self.skip_keys)
        type_dict, object_dtypes = self.get_data_types(data, keys)
        state = self.StageState(
            keys=keys, type_dict=type_dict, object_dtypes=object_dtypes
        )
        self._stage_states[stage] = state
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
            Otherwise reuse ``self.file_name`` directly.
        """
        if not multiple_sources:
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

        Returns
        -------
        list[tuple[str, dict, dict]]
            One tuple per source file containing the resolved output file path,
            the batch subset that belongs to that source file, and the
            file-level source provenance dictionary.
        """
        required = ("source_file_name", "source_file_size", "source_file_mtime_ns")
        for key in required:
            if key not in data:
                raise KeyError(
                    "StageHDF5Writer requires reader-provided source provenance. "
                    f"Missing key: {key}."
                )

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
        """
        stages = out_file["stages"]
        assert isinstance(stages, h5py.Group), "'stages' must be an HDF5 group."

        if stage in stages and overwrite_stage:
            del stages[stage]
            self._completed_stages[file_path].discard(stage)

        if stage not in stages:
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
            self.initialize_datasets(stage_group, state.type_dict)
            state.event_dtype = self.event_dtype
            return stage_group

        stage_group = stages[stage]
        assert isinstance(
            stage_group, h5py.Group
        ), f"Stage '{stage}' is expected to be a group, got {type(stage_group)}."

        if stage not in self._stage_states:
            raise RuntimeError(
                f"Stage '{stage}' already exists in '{self.file_name}'. Reopening and "
                "appending an existing stage across writer sessions is not supported "
                "in this first pass. Pass overwrite_stage=True to rebuild it."
            )

        if "info" in stage_group and attrs is not None:
            for key, value in attrs.items():
                stage_group["info"].attrs[key] = value
        if "info" in stage_group and cfg is not None:
            stage_group["info"].attrs["cfg"] = yaml.dump(cfg)

        stage_group["info"].attrs["complete"] = False
        return stage_group

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
        normalized, batch_size = self._prepare_batch(data)
        state = self._stage_states.get(stage)
        if state is None or overwrite_stage:
            state = self._create_stage_state(stage, normalized)

        for file_path, subset, _ in self.split_batch_by_source(normalized):
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

                self.keys = state.keys
                self.type_dict = state.type_dict
                self.object_dtypes = state.object_dtypes
                self.event_dtype = state.event_dtype

                for batch_id in range(len(subset["index"])):
                    self.append_entry(stage_group, subset, batch_id)

                state.event_dtype = self.event_dtype
                if self.flush_frequency is not None:
                    state.entries_since_flush += len(subset["index"])
                    if state.entries_since_flush >= self.flush_frequency:
                        out_file.flush()
                        state.entries_since_flush = 0
            finally:
                if should_close:
                    out_file.close()

    def finalize_stage(self, stage: str) -> None:
        """Mark one stage as complete in every touched cache file.

        Parameters
        ----------
        stage : str
            Stage name to finalize across all cache files written by this
            writer instance.
        """
        for file_path in sorted(self._known_files):
            out_file, should_close = self._open_handle(file_path)
            try:
                stages = out_file["stages"]
                assert isinstance(stages, h5py.Group), "'stages' must be an HDF5 group."
                if stage not in stages:
                    continue

                stage_group = stages[stage]
                stage_group["info"].attrs["complete"] = True
                out_file.flush()
                self._completed_stages[file_path].add(stage)
            finally:
                if should_close:
                    out_file.close()

    def list_stages(self) -> tuple[str, ...]:
        """Return the union of stage-group names across touched cache files.

        Returns
        -------
        tuple[str, ...]
            Sorted tuple of unique stage names seen in all output cache files
            touched by this writer instance.
        """
        stage_names: set[str] = set()
        for file_path in sorted(self._known_files):
            out_file, should_close = self._open_handle(file_path)
            try:
                stages = out_file["stages"]
                assert isinstance(stages, h5py.Group), "'stages' must be an HDF5 group."
                stage_names.update(stages.keys())
            finally:
                if should_close:
                    out_file.close()

        return tuple(sorted(stage_names))
