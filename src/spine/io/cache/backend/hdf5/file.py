"""File lifecycle and source-provenance helpers for cache HDF5 shards."""

from __future__ import annotations

import os
from typing import Any, Callable

import h5py
import numpy as np

from spine.version import __version__

from ....write.hdf5.common import decode_string_attribute, require_group

__all__ = ["StageFileMixin"]


class StageFileMixin:
    """Manage cache-shard files, handles, and source provenance.

    The mixin deliberately contains no output-product serialization. It owns
    the physical shard contract: lazy file creation, process-local handles,
    container validation, and immutable source-file routing.
    """

    # Concrete-writer interface required by this mixin. These declarations
    # are type-only: the shard writer initializes the state, while the HDF5
    # base class supplies the shared path helper and format constants.
    name: str
    legacy_format_version: int
    current_format_version: int
    format_version: int
    keep_open: bool
    source_info: dict[str, Any] | None

    _handle_pid: int | None
    _handles: dict[str, h5py.File]
    _initialized_files: set[str]
    _known_files: set[str]
    _ensure_parent_dir: Callable[[str], None]

    def close(self) -> None:
        """Close and forget all persistent shard handles.

        The method is idempotent and tolerates a partially initialized writer,
        which allows it to serve both explicit cleanup and destructor paths.
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

    def _close_path_handle(self, file_path: str) -> None:
        """Close and forget one persistent shard handle, if present.

        Parameters
        ----------
        file_path : str
            Physical shard path whose cached append handle should be released.
        """
        handle = self._handles.pop(file_path, None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, RuntimeError, ValueError):
                pass

    @staticmethod
    def _get_stage_write_path(
        target_path: str,
        stage: str,
        source_info: dict[str, Any],
        overwrite_stage: bool,
    ) -> str:
        """Return the transaction-private shard selected by source routing.

        The logical cache repository already owns publication and replacement
        policy, so the physical backend writes directly to its pending shard.
        The remaining arguments document the caller's complete routing state.

        Parameters
        ----------
        target_path : str
            Source-routed transaction-private output path.
        stage : str
            Logical stage being written.
        source_info : dict
            Immutable identity of the source represented by the shard.
        overwrite_stage : bool
            Requested replacement policy, handled by the repository layer.

        Returns
        -------
        str
            Unchanged transaction-private output path.
        """
        del stage, source_info, overwrite_stage
        return target_path

    def _check_handle_pid(self) -> None:
        """Ensure persistent writer handles remain process-local.

        Cache shards are not safe to append to through a writer instance that
        has crossed a process boundary. This method enforces the same
        single-process handle ownership contract as the regular HDF5 writer.
        """
        current_pid = os.getpid()
        if self._handle_pid is None:
            self._handle_pid = current_pid
            return

        if self._handle_pid != current_pid:
            raise RuntimeError(
                "Cache shard writer handles are process-local and cannot be "
                "reused across process boundaries."
            )

    def _open_handle(self, file_path: str) -> tuple[h5py.File, bool]:
        """Return an appendable cache-file handle for one output path.

        Parameters
        ----------
        file_path : str
            Physical shard path to open or reuse.

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
        """Initialize one cache-shard container on first use.

        Administrative groups are created lazily because output paths depend
        on source provenance and not every source is necessarily touched by a
        write call.

        Parameters
        ----------
        file_path : str
            Physical shard path to initialize.
        """
        if file_path in self._initialized_files:
            return

        file_exists = os.path.exists(file_path)
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
        """Require an existing cache shard to use the stage-aware V2 layout.

        Cache shards are disposable internal products, so legacy files are
        rejected with an instruction to rebuild rather than upgraded in place.

        Parameters
        ----------
        out_file : h5py.File
            Existing cache file opened for reading or append.
        file_path : str
            Path included in validation errors.

        Raises
        ------
        ValueError
            If the container metadata does not identify a compatible V2 cache
            shard with a top-level stage namespace.
        """
        if "info" not in out_file:
            raise ValueError(
                f"Cannot append cache shard '{file_path}': missing info group."
            )

        info = require_group(out_file, "info")
        raw_version = info.attrs.get("format_version", self.legacy_format_version)
        stored_version = int(np.asarray(raw_version).item())
        if stored_version != self.current_format_version:
            raise ValueError(
                f"Cache shard '{file_path}' uses HDF5 format version "
                f"{stored_version}; rebuild it with version 2."
            )

        stored_format = decode_string_attribute(info.attrs.get("format"), "format")
        if stored_format != self.name:
            raise ValueError(
                f"Cannot append cache shard '{file_path}': expected format "
                f"'{self.name}', found '{stored_format}'."
            )
        require_group(out_file, "stages")

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

        Raises
        ------
        KeyError
            If the batch does not contain complete source-file provenance.
        ValueError
            If a provenance field is empty or varies within the shard batch.
        """
        required = ("source_file_name", "source_file_size", "source_file_mtime_ns")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                "The cache HDF5 backend requires reader-provided source "
                "provenance. "
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
                    "The cache HDF5 backend expects one source per shard. "
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
            Open cache-shard output file.
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
