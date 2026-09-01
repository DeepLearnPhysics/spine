"""File lifecycle and source-provenance helpers for staged HDF5 output."""

from __future__ import annotations

import os
from typing import Any, Callable

import h5py
import numpy as np

from spine.version import __version__

from ..hdf5.common import decode_string_attribute, require_group

__all__ = ["StageFileMixin"]


class StageFileMixin:
    """Manage staged-cache files, handles, and source provenance.

    The mixin deliberately contains no output-product serialization. It owns
    the physical file contract shared by direct and sidecar writes: lazy file
    creation, process-local handles, container validation, and immutable
    source-file routing.
    """

    # Concrete-writer interface required by this mixin. These declarations
    # are type-only: ``StageHDF5Writer`` initializes the state, while the HDF5
    # base class supplies the shared path helper and format constants.
    name: str
    legacy_format_version: int
    current_format_version: int
    format_version: int
    keep_open: bool
    sidecar: bool
    source_info: dict[str, Any] | None

    _handle_pid: int | None
    _handles: dict[str, h5py.File]
    _initialized_files: set[str]
    _known_files: set[str]
    _sidecar_paths: dict[tuple[str, str], str]
    _sidecar_replace: dict[tuple[str, str], bool]
    _target_by_source: dict[tuple[str, int, int], str]

    _ensure_parent_dir: Callable[[str], None]

    def close(self) -> None:
        """Close persistent handles and discard uncommitted sidecars.

        Canonical caches are never modified by this cleanup. If finalization
        did not successfully merge a temporary stage, its sidecar is removed.
        This method may be called repeatedly.
        """
        # ``__del__`` may reach this method after constructor validation
        # rejected a partially initialized writer.
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except (OSError, RuntimeError, ValueError):
                pass

        self._handles = {}
        self._handle_pid = None

        # A sidecar that remains mapped was never committed successfully.
        sidecar_paths = getattr(self, "_sidecar_paths", {})
        for sidecar_path in set(sidecar_paths.values()):
            try:
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
            except OSError:
                pass

        self._sidecar_paths = {}
        self._sidecar_replace = {}

    def _check_handle_pid(self) -> None:
        """Ensure persistent writer handles remain process-local.

        Stage caches are not safe to append to through a writer instance that
        has crossed a process boundary. This method enforces the same
        single-process handle ownership contract as the regular HDF5 writer.
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
        self._ensure_stage_file(file_path)
        if not self.keep_open:
            return h5py.File(file_path, "a"), True

        self._check_handle_pid()
        handle = self._handles.get(file_path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(file_path, "a")
            self._handles[file_path] = handle

        return handle, False

    def _ensure_stage_file(self, file_path: str) -> None:
        """Initialize one staged-cache container on first use.

        Administrative groups are created lazily because output paths depend
        on source provenance and not every source is necessarily touched by a
        write call. A zero-length sidecar path reserved with ``mkstemp`` is
        treated as a new file rather than an existing HDF5 container.

        Parameters
        ----------
        file_path : str
            Physical direct-output or sidecar path to initialize.
        """
        if file_path in self._initialized_files:
            return

        file_exists = os.path.exists(file_path)
        reserved_sidecar = (
            file_exists
            and self.sidecar
            and file_path in self._sidecar_paths.values()
            and os.path.getsize(file_path) == 0
        )
        file_exists = file_exists and not reserved_sidecar
        mode = "a" if file_exists else "w"
        if mode == "w":
            self._ensure_parent_dir(file_path)

        if self.keep_open:
            self._check_handle_pid()
            out_file = h5py.File(file_path, mode)
            self._handles[file_path] = out_file
        else:
            out_file = h5py.File(file_path, mode)

        try:
            if file_exists:
                self._validate_stage_file(out_file, file_path)
            else:
                # These roots are shared by every independently owned stage.
                info = out_file.create_group("info")
                info.attrs["version"] = __version__
                info.attrs["spine_version"] = __version__
                info.attrs["format"] = self.name
                info.attrs["format_version"] = self.format_version
                out_file.create_group("stages")
        finally:
            if not self.keep_open:
                out_file.close()

        self._initialized_files.add(file_path)
        self._known_files.add(file_path)

    def _validate_stage_file(self, out_file: h5py.File, file_path: str) -> None:
        """Require an existing cache file to use the staged V2 layout.

        Staged caches are disposable internal products, so legacy files are
        rejected with an instruction to rebuild rather than upgraded in place.

        Parameters
        ----------
        out_file : h5py.File
            Existing cache file opened for reading or append.
        file_path : str
            Path included in validation errors.
        """
        if "info" not in out_file:
            raise ValueError(
                f"Cannot append staged cache '{file_path}': missing info group."
            )

        info = require_group(out_file, "info")
        raw_version = info.attrs.get("format_version", self.legacy_format_version)
        stored_version = int(np.asarray(raw_version).item())
        if stored_version != self.current_format_version:
            raise ValueError(
                f"Staged cache '{file_path}' uses HDF5 format version "
                f"{stored_version}; rebuild it with version 2."
            )

        stored_format = decode_string_attribute(info.attrs.get("format"), "format")
        if stored_format != self.name:
            raise ValueError(
                f"Cannot append staged cache '{file_path}': expected format "
                f"'{self.name}', found '{stored_format}'."
            )
        require_group(out_file, "stages")

    @staticmethod
    def _source_identity(source_info: dict[str, Any]) -> tuple[str, int, int]:
        """Return the immutable identity tuple used to route one source file.

        Parameters
        ----------
        source_info : dict
            Metadata containing ``file_name``, ``file_size``, and
            ``file_mtime_ns``.

        Returns
        -------
        tuple[str, int, int]
            Normalized source name, size, and modification timestamp.
        """
        return (
            str(source_info["file_name"]),
            int(source_info["file_size"]),
            int(source_info["file_mtime_ns"]),
        )

    @staticmethod
    def _read_source_info(in_file: h5py.File, file_path: str) -> dict[str, Any]:
        """Read canonical source provenance from a staged cache.

        Parameters
        ----------
        in_file : h5py.File
            Open staged-cache handle.
        file_path : str
            Cache path used in validation errors.

        Returns
        -------
        dict[str, Any]
            Normalized source name, size, and modification timestamp.

        Raises
        ------
        ValueError
            If the source group or any required attribute is absent.
        """
        if "source" not in in_file:
            raise ValueError(f"Staged cache '{file_path}' is missing its source group.")

        source = require_group(in_file, "source")
        required = ("file_name", "file_size", "file_mtime_ns")
        missing = [key for key in required if key not in source.attrs]
        if missing:
            raise ValueError(
                f"Staged cache '{file_path}' is missing source attributes {missing}."
            )

        return {
            "file_name": decode_string_attribute(
                source.attrs["file_name"], "file_name"
            ),
            # HDF5's typing permits array-valued attributes. These fields are
            # scalar by contract, so normalize their zero-dimensional NumPy
            # representation before converting to a Python integer.
            "file_size": int(np.asarray(source.attrs["file_size"]).item()),
            "file_mtime_ns": int(np.asarray(source.attrs["file_mtime_ns"]).item()),
        }

    def _index_target_files(self, file_paths: list[str]) -> None:
        """Index canonical staged caches by persisted source provenance.

        Parameters
        ----------
        file_paths : list[str]
            Existing staged-cache paths that may receive the new stage.

        Raises
        ------
        ValueError
            If a target is invalid or multiple targets claim the same source.
        """
        for file_path in file_paths:
            normalized_path = os.path.abspath(os.fspath(file_path))
            with h5py.File(normalized_path, "r") as in_file:
                self._validate_stage_file(in_file, normalized_path)
                source_info = self._read_source_info(in_file, normalized_path)

            identity = self._source_identity(source_info)
            previous = self._target_by_source.get(identity)
            if previous is not None and previous != normalized_path:
                raise ValueError(
                    "Multiple staged caches claim source provenance "
                    f"{identity}: '{previous}' and '{normalized_path}'."
                )
            self._target_by_source[identity] = normalized_path

    def get_batch_source_info(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract cache-file source provenance from one normalized batch.

        Parameters
        ----------
        data : dict
            Normalized batch dictionary prepared for writing.

        Returns
        -------
        dict[str, Any]
            File-level source identity stored under the top-level ``/source``
            group.
        """
        required = ("source_file_name", "source_file_size", "source_file_mtime_ns")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                "StageHDF5Writer requires reader-provided source provenance. "
                f"Missing keys: {missing}."
            )

        # Products arrive through several scalar and array-backed reader
        # representations, but normalization below reduces each one to a
        # Python scalar before the file-level identity is constructed.
        values: dict[str, Any] = {}
        for key in required:
            value = data[key]
            if np.isscalar(value):
                values[key] = value.item() if isinstance(value, np.generic) else value
                continue

            array = np.asarray(value)
            if array.ndim == 0:
                values[key] = array.item()
                continue
            if len(array) == 0:
                raise ValueError(f"Source provenance key '{key}' is empty.")

            first = array[0].item() if hasattr(array[0], "item") else array[0]
            if any(
                (element.item() if hasattr(element, "item") else element) != first
                for element in array[1:]
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
        """Create or validate the top-level source-provenance group.

        This enforces the one-cache-file-per-source-file contract. If a later
        stage attempts to write into an existing cache with mismatched source
        provenance, the writer raises immediately.

        Parameters
        ----------
        out_file : h5py.File
            Open staged-cache output file.
        data : dict
            Normalized batch containing source provenance.
        file_path : str
            Output path used in mismatch diagnostics.

        Raises
        ------
        RuntimeError
            If existing file provenance does not match the input batch.
        """
        source_info = self.get_batch_source_info(data)
        self.source_info = source_info

        # Record provenance on first use, then enforce it on every later stage.
        if "source" not in out_file:
            source_group = out_file.create_group("source")
            for key, value in source_info.items():
                source_group.attrs[key] = value
            return

        source_group = require_group(out_file, "source")
        for key, value in source_info.items():
            cached_value = source_group.attrs.get(key)
            if cached_value != value:
                raise RuntimeError(
                    f"Cache source mismatch for '{file_path}': '{key}' differs "
                    f"({cached_value!r} != {value!r})."
                )
