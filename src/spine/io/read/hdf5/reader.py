"""Contains a reader class dedicated to loading data from HDF5 files."""

import os
from collections.abc import Sequence
from typing import Any, cast
from warnings import warn

import h5py
import numpy as np
import yaml
from yaml.parser import ParserError

from spine.logging import logger

from ..base import ReaderBase
from .common import contiguous_runs, require_group, resolve_object_class
from .product import ProductGroupBackend
from .region import RegionReferenceBackend

__all__ = ["HDF5Reader"]


class HDF5Reader(ProductGroupBackend, RegionReferenceBackend, ReaderBase):
    """Read event data from versioned SPINE HDF5 files.

    This class inherits from the :class:`ReaderBase` class. It provides
    methods to load HDF5 files and extract their data products. Two physical
    layouts are supported:

    - Version 1 is the legacy layout. Each row of ``events`` is a compound
      record of HDF5 region references into top-level product datasets.
      Variable object attributes use HDF5 VLEN fields.
    - Version 2 stores logical products below ``/products`` and uses monotonic
      integer offset arrays to delimit events, objects, and variable fields.
      Product-owned auxiliary data is nested below the corresponding logical
      group. Its ``events`` dataset remains the authoritative event axis but
      contains no product references.

    The reader detects the layout independently for every input file. This
    allows one logical dataset to span legacy and V2 files without exposing
    layout differences to callers. Files which predate explicit
    ``info.attrs["format_version"]`` metadata are interpreted as V1.

    Product projection is performed before any product dataset is accessed.
    This is particularly useful for V2 because logical product names live
    below ``/products`` rather than in the ``events`` compound dtype.
    """

    name: str = "hdf5"

    # Retain these format-independent helpers on the reader for callers which
    # validate stored object schemas or construct columnar entry runs directly.
    _contiguous_runs = staticmethod(contiguous_runs)
    resolve_object_class = staticmethod(resolve_object_class)

    def __init__(
        self,
        file_keys: str | list[str] | None = None,
        file_list: str | None = None,
        limit_num_files: int | None = None,
        max_print_files: int = 10,
        n_entry: int | None = None,
        n_skip: int | None = None,
        entry_list: list[int] | None = None,
        skip_entry_list: list[int] | None = None,
        run_event_list: list[list[int]] | None = None,
        skip_run_event_list: list[list[int]] | None = None,
        create_run_map: bool = False,
        build_classes: bool = True,
        fixed_only: bool = False,
        columnar: bool = False,
        chunk_size: int = 1024,
        skip_unknown_attrs: bool = False,
        run_info_key: str = "run_info",
        allow_missing: bool = False,
        keep_open: bool = True,
        swmr: bool = False,
        ignore_incomplete: bool = False,
        keys: list[str] | tuple[str, ...] | None = None,
        entry_fraction_range: Sequence[float] | None = None,
    ) -> None:
        """Initalize the HDF5 file reader.

        Parameters
        ----------
        file_keys : str or list[str], optional
            Path or list of paths to the HDF5 files to be read
        file_list : str, optional
            Path to a text file containing a list of file paths to be read
        limit_num_files : int, optional
            Integer limiting number of files to be taken per data directory
        max_print_files : int, default 10
            Maximum number of loaded file names to be printed
        n_entry : int, optional
            Maximum number of entries to load
        n_skip : int, optional
            Number of entries to skip at the beginning
        entry_list : list[int], optional
            List of integer entry IDs to add to the index
        skip_entry_list : list[int], optional
            List of integer entry IDs to skip from the index
        run_event_list : list[list[int]], optional
            List of (run, subrun, event) triplets to add to the index
        skip_run_event_list : list[list[int]], optional
            List of (run, subrun, event) triplets to skip from the index
        create_run_map : bool, default False
            Initialize a map between (run, subrun, event) triplets and entries.
            For large files, this can be quite expensive (must load every entry).
        build_classes : bool, default True
            If the stored object is a class, build it back
        fixed_only : bool, default False
            If `True`, load only the fixed compound rows of V2 object
            products and do not access their variable-value pools. This is
            useful for high-level consumers which need scalar and fixed-width
            attributes only. It is not supported for V1 files. When classes
            are rebuilt, omitted variable attributes retain their class
            defaults; derived properties which depend on them are therefore
            not reliable. With `build_classes=False`, stored derived fields
            remain available directly in the returned dictionaries.
        columnar : bool, default False
            If `True`, expose projected object products in multi-event chunks
            through :meth:`get_columnar`. Event/class loading remains the
            default. Columnar projection is configured by the analysis manager
            before processing begins.
        chunk_size : int, default 1024
            Maximum number of selected events in one columnar chunk.
        skip_unknown_attrs : bool, default False
            If `True`, allow a loaded object to have unrecognized attributes.
            This allows backward compatibility with old files, but use with
            extreme caution, as this might hide a fundamental issue with your code.
        run_info_key : str, default 'run_info'
            Name of the data product which contains the run info of the event
        allow_missing : bool, default False
            If `True`, allows missing entries in the entry or event list
        keep_open : bool, default True
            If `True`, keep one read-only HDF5 handle open per file and per
            process. This avoids reopening files for every event access. If
            `False`, open and close the file on each `get` call.
        swmr : bool, default False
            If `True`, open files in HDF5 single-writer/multiple-reader mode.
            This is only relevant when reading files produced by a writer that
            was configured for SWMR-safe operation.
        ignore_incomplete : bool, default False
            If `True`, allow opening files marked as incomplete. By default,
            files with an explicit `info.attrs["complete"] = False` marker are
            rejected.
        keys : sequence[str], optional
            Data products to load. If omitted, load every product. This is a
            true reader-level projection and avoids reading unrequested data
            in either layout. Source-provenance products remain eligible so
            reader-owned runtime indexes can be reconstructed.
        entry_fraction_range : sequence[float], optional
            Half-open fractional range of the resolved entry order to select
        """
        # Process the list of files
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)
        self.keep_open = keep_open
        self.swmr = swmr
        self.ignore_incomplete = ignore_incomplete
        self.fixed_only = fixed_only
        self.columnar = columnar
        if chunk_size <= 0:
            raise ValueError("`chunk_size` must be a positive integer.")
        self.chunk_size = chunk_size
        self._columnar_requests: dict[str, tuple[tuple[str, ...] | None, bool]] = {}
        self._handle_pid: int | None = None
        self._file_handles: dict[int, h5py.File] = {}

        # Product schemas and process-local dataset handles are owned by the
        # product-group backend and initialized alongside shared file handles.
        self._initialize_product_backend()
        self.requested_keys = set(keys) if keys is not None else None
        self.file_format_versions: list[int] = []

        # If an entry list is requested based on run/subrun/event ID, create map
        if run_event_list is not None or skip_run_event_list is not None:
            create_run_map = True

        # Loop over the input files, build a map from index to file ID
        file_index, run_info = [], []
        self.num_entries = 0
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as in_file:
                # Check that there are events in the file
                assert "events" in in_file, "File does not contain an event tree"
                if (
                    "info" in in_file
                    and "complete" in in_file["info"].attrs
                    and not in_file["info"].attrs["complete"]
                    and not self.ignore_incomplete
                ):
                    raise RuntimeError(
                        f"HDF5 file '{path}' is marked incomplete. "
                        "Pass ignore_incomplete=True to override."
                    )

                events = in_file["events"]
                assert isinstance(
                    events, h5py.Dataset
                ), "'events' is not a dataset in the HDF5 file."
                # Explicit layout metadata was introduced with V2. Its absence
                # is therefore an unambiguous legacy-file marker, not an error.
                format_version = 1
                if "info" in in_file:
                    format_version = int(in_file["info"].attrs.get("format_version", 1))
                if format_version not in (1, 2):
                    raise ValueError(
                        f"Unsupported HDF5 format version {format_version} in '{path}'."
                    )
                self.file_format_versions.append(format_version)

                # If requested, register the (run, subrun, event) information
                if create_run_map:
                    product_root = (
                        in_file
                        if format_version == 1
                        else require_group(in_file, "products")
                    )
                    assert (
                        run_info_key in product_root
                    ), f"Must provide {run_info_key} to create run map"

                    info = product_root[run_info_key]
                    if format_version == 1:
                        # V1 object fields are columns of one compound dataset.
                        assert isinstance(
                            info, h5py.Dataset
                        ), f"{run_info_key} is not a dataset in the HDF5 file."
                        assert all(
                            k in info.dtype.names for k in ["run", "subrun", "event"]
                        ), f"{run_info_key} dataset missing required fields."
                        columns = (info["run"], info["subrun"], info["event"])

                    else:
                        # In V2, derived and fixed-width fields remain directly
                        # queryable in the product's compound `fixed` dataset.
                        assert isinstance(info, h5py.Group)
                        fixed = info["fixed"]
                        assert isinstance(fixed, h5py.Dataset)
                        assert all(
                            k in fixed.dtype.names for k in ["run", "subrun", "event"]
                        ), f"{run_info_key} dataset missing required fields."
                        columns = (fixed["run"], fixed["subrun"], fixed["event"])

                    for r, s, e in zip(*columns):
                        run_info.append((r, s, e))

                # Update the total number of entries
                num_entries = len(events)
                file_index.append(i * np.ones(num_entries, dtype=np.int64))
                self.file_offsets[i] = self.num_entries
                self.num_entries += num_entries

        if self.fixed_only and any(
            version != 2 for version in self.file_format_versions
        ):
            raise ValueError(
                "`fixed_only=True` is supported only for HDF5 format "
                "version 2 files."
            )

        # Dump the number of entries to load
        logger.info("Total number of entries in the file(s): %d\n", self.num_entries)

        # Concatenate the file indexes into one, set run info if needed
        self.file_index = np.concatenate(file_index)
        self.run_info = run_info if create_run_map else None

        # Process the run information
        self.process_run_info()

        # Process the entry list
        self.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            run_event_list,
            skip_run_event_list,
            allow_missing,
            entry_fraction_range,
        )

        # Store other attributes
        self.build_classes = build_classes
        self.skip_unknown_attrs = skip_unknown_attrs

        # Process the configuration used to produce the HDF5 file
        self.cfg = self.process_cfg()

        # Process cumulative post-processing provenance, when available
        self.post_processors = self.process_post_processors()

        # Process the SPINE version used to produced the HDF5 file
        self.version = self.process_version()

    @property
    def num_chunks(self) -> int:
        """Number of chunks exposed by the configured columnar reader."""
        return (len(self) + self.chunk_size - 1) // self.chunk_size

    def configure_columnar(
        self,
        requests: dict[str, tuple[tuple[str, ...] | None, bool]],
    ) -> None:
        """Install the analyzer-derived product projection.

        Parameters
        ----------
        requests : dict
            Mapping from object-product names to requested fixed fields and a
            flag indicating whether the product is required.

        Raises
        ------
        RuntimeError
            If the reader was not initialized in columnar mode.
        """
        if not self.columnar:
            raise RuntimeError("Cannot configure columnar projection in event mode.")
        self._columnar_requests = dict(requests)

    def get_columnar(self, idx: int) -> dict[str, Any]:
        """Load one projected multi-event chunk without rebuilding classes.

        Parameters
        ----------
        idx : int
            Chunk index on the configured columnar event axis.

        Returns
        -------
        dict
            Administrative arrays and projected object fields for the chunk.

        Raises
        ------
        RuntimeError
            If columnar mode or its product projection is not configured.
        IndexError
            If ``idx`` falls outside the available chunks.
        KeyError
            If a required product is absent from an input file.
        """
        if not self.columnar:
            raise RuntimeError("Columnar loading was not enabled for this reader.")
        if idx < 0 or idx >= self.num_chunks:
            raise IndexError(
                f"Chunk {idx} out of bounds for columnar reader with "
                f"{self.num_chunks} chunks."
            )
        if not self._columnar_requests:
            raise RuntimeError("Columnar product projection was not configured.")

        first = idx * self.chunk_size
        last = min(first + self.chunk_size, len(self))
        selected = self.entry_index[first:last]
        file_indices = self.file_index[selected]
        local_entries = selected - self.file_offsets[file_indices]
        data: dict[str, Any] = {
            "index": np.arange(first, last, dtype=np.int64),
            "file_index": file_indices.astype(np.int64, copy=False),
            "file_entry_index": local_entries.astype(np.int64, copy=False),
        }

        # Preserve selected event order while grouping adjacent entries from
        # the same file into one physical access unit.
        file_runs: list[tuple[int, np.ndarray]] = []
        run_start = 0
        for i in range(1, len(selected) + 1):
            if i == len(selected) or file_indices[i] != file_indices[run_start]:
                file_runs.append(
                    (
                        int(file_indices[run_start]),
                        local_entries[run_start:i].astype(np.int64, copy=False),
                    )
                )
                run_start = i

        for key, (requested_fields, required) in self._columnar_requests.items():
            pieces = []
            missing = False
            for file_idx, entries in file_runs:
                in_file, should_close = self._open_file(file_idx)
                try:
                    version = self.file_format_versions[file_idx]
                    product_root = (
                        in_file if version == 1 else require_group(in_file, "products")
                    )
                    if key not in product_root:
                        missing = True
                        continue
                    if version == 1:
                        piece = self._load_region_columnar_objects(
                            in_file, key, entries, requested_fields
                        )
                    else:
                        piece = self._load_product_columnar_objects(
                            product_root, key, entries, requested_fields
                        )
                    pieces.append(piece)
                finally:
                    if should_close:
                        in_file.close()

            if missing:
                if required:
                    raise KeyError(
                        f"Required columnar product `{key}` is missing from "
                        "one or more input files."
                    )
                continue
            if pieces:
                data[key] = self._merge_columnar_products(pieces)

        return data

    @staticmethod
    def _merge_columnar_products(pieces: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate file-local columnar products into one chunk."""
        names = tuple(name for name in pieces[0] if name != "event_offsets")
        result = {
            name: np.concatenate([piece[name] for piece in pieces]) for name in names
        }
        counts = np.concatenate([np.diff(piece["event_offsets"]) for piece in pieces])
        result["event_offsets"] = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )
        return result

    def close(self) -> None:
        """Close any persistent HDF5 handles owned by this reader.

        This only affects handles cached in the current process. It is safe to
        call repeatedly.
        """
        for handle in getattr(self, "_file_handles", {}).values():
            try:
                handle.close()
            except (OSError, RuntimeError, ValueError):
                pass

        self._file_handles = {}
        self._clear_product_handles()
        self._handle_pid = None

    def __del__(self) -> None:
        """Best-effort cleanup of persistent read handles on object teardown."""
        self.close()

    def _check_handle_pid(self) -> None:
        """Ensure cached handles belong to the current process.

        Reader instances may be copied into worker processes by data-loading
        frameworks. When that happens, inherited file handles must not be
        reused. This method drops any cached handles on PID changes and lets
        the caller reopen them lazily in the new process.
        """
        current_pid = _get_reader_pid()
        if self._handle_pid is None:
            self._handle_pid = current_pid
            return

        if self._handle_pid != current_pid:
            self.close()
            self._handle_pid = current_pid

    def _open_file(self, file_idx: int) -> tuple[h5py.File, bool]:
        """Return a readable HDF5 handle for one input file.

        Parameters
        ----------
        file_idx : int
            Position of the target file in `self.file_paths`

        Returns
        -------
        tuple[h5py.File, bool]
            The opened HDF5 file handle and a flag indicating whether the
            caller is responsible for closing it. The flag is `True` only when
            `keep_open=False`.
        """
        if not self.keep_open:
            return h5py.File(self.file_paths[file_idx], "r", swmr=self.swmr), True

        self._check_handle_pid()
        handle = self._file_handles.get(file_idx)
        if handle is None or not handle.id.valid:
            handle = h5py.File(self.file_paths[file_idx], "r", swmr=self.swmr)
            self._file_handles[file_idx] = handle

        return handle, False

    def process_cfg(self) -> dict[str, Any] | None:
        """Fetches the SPINE configuration used to produce the HDF5 file.

        Returns
        -------
        dict
            Configuration dictionary
        """
        # Fetch the string-form configuration
        with h5py.File(self.file_paths[0], "r") as in_file:
            assert "info" in in_file, "HDF5 file missing 'info' group."
            assert (
                "cfg" in in_file["info"].attrs
            ), "HDF5 file 'info' group missing 'cfg' attribute."
            cfg_str = in_file["info"].attrs["cfg"]

        # Attempt to parse it (need try for now for SPINE versions < v0.4.0)
        try:
            assert isinstance(cfg_str, str), "'cfg' attribute is not a string."
            cfg = yaml.safe_load(cfg_str)
        except ParserError:
            warn(
                "Parsing configuration failed, returning None for SPINE versions < v0.4.0"
            )
            return None

        return cfg

    def process_post_processors(self) -> tuple[str, ...] | None:
        """Return cumulative post-processing provenance from the input files.

        Files written before this metadata was introduced return ``None``;
        :class:`IOManager` then falls back to their stored configuration. For
        multi-file inputs, all recorded histories must agree because taking a
        union could claim that products exist in files where they do not.

        Returns
        -------
        tuple[str, ...] or None
            Ordered processor names, or ``None`` for legacy files.

        Raises
        ------
        TypeError
            If the stored attribute is malformed.
        ValueError
            If the input files record different histories.
        """
        histories: list[tuple[str, ...] | None] = []
        for path in self.file_paths:
            with h5py.File(path, "r") as in_file:
                assert "info" in in_file, "HDF5 file missing 'info' group."
                payload = in_file["info"].attrs.get("post_processors")

            if payload is None:
                histories.append(None)
                continue
            if not isinstance(payload, str):
                raise TypeError(
                    "HDF5 file 'post_processors' attribute must be a string."
                )
            decoded = yaml.safe_load(payload)
            if not isinstance(decoded, list) or not all(
                isinstance(name, str) for name in decoded
            ):
                raise TypeError(
                    "HDF5 file 'post_processors' attribute must decode to a "
                    "list of strings."
                )
            histories.append(tuple(decoded))

        history = histories[0]
        if any(candidate != history for candidate in histories[1:]):
            raise ValueError(
                "Input HDF5 files record different post-processing provenance."
            )
        return history

    def process_version(self) -> str:
        """Return the SPINE software release which produced the first file.

        ``spine_version`` identifies software and must not be confused with
        ``format_version``, which selects the physical HDF5 layout. The
        historical ``version`` attribute remains as a fallback for files
        written before these concepts were named separately.

        Returns
        -------
        str
            SPINE release tag
        """
        # Fetch the string-form configuration
        with h5py.File(self.file_paths[0], "r") as in_file:
            assert "info" in in_file, "HDF5 file missing 'info' group."
            attrs = in_file["info"].attrs
            assert (
                "spine_version" in attrs or "version" in attrs
            ), "HDF5 file 'info' group missing a SPINE version attribute."
            version = attrs.get("spine_version", attrs.get("version"))

        assert isinstance(version, str), "'version' attribute is not a string."
        return version

    def get(self, idx: int) -> dict[str, Any]:
        """Return one decoded entry from the HDF5 input files.

        Parameters
        ----------
        idx : int
            Reader entry index to access.

        Returns
        -------
        dict
            Data products and administrative metadata for one event.

        Raises
        ------
        IndexError
            If ``idx`` is outside the configured reader entry list.
        """
        # Resolve the user-facing index onto its physical file and local entry
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}."
            )
        file_idx = self.get_file_index(idx)
        entry_idx = self.get_file_entry_index(idx)

        in_file, should_close = self._open_file(file_idx)
        try:
            return self._load_entry(idx, file_idx, entry_idx, in_file)
        finally:
            if should_close:
                in_file.close()

    def get_many(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Load a batch using contiguous V2 reads where possible.

        Input order, duplicate indexes, entry filtering, and scalar decoding
        semantics are preserved. V2 entries which are adjacent within one
        physical file are decoded as a run; V1 and isolated entries retain the
        scalar path. Each participating file is opened at most once.

        Parameters
        ----------
        indices : sequence[int]
            Reader entry indexes to load. They may be unordered or repeated.

        Returns
        -------
        list[dict]
            Decoded entries in the same order as ``indices``.

        Raises
        ------
        IndexError
            If any requested index is outside the configured entry list. All
            indexes are validated before an event-read handle is opened.
        """
        # Resolve and validate the full request before performing physical I/O
        resolved: list[tuple[int, int, int, int]] = []
        for position, raw_idx in enumerate(indices):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self):
                raise IndexError(
                    f"Index {idx} out of bounds for dataset of size {len(self)}."
                )
            resolved.append(
                (
                    position,
                    idx,
                    self.get_file_index(idx),
                    self.get_file_entry_index(idx),
                )
            )

        # Group by physical file while retaining each result's output position
        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for position, idx, file_idx, entry_idx in resolved:
            grouped.setdefault(file_idx, []).append((position, idx, entry_idx))

        # Reuse one transient or persistent handle for each participating file.
        results: list[dict[str, Any] | None] = [None] * len(resolved)
        for file_idx, entries in grouped.items():
            in_file, should_close = self._open_file(file_idx)
            try:
                run_start = 0
                for index in range(1, len(entries) + 1):
                    if (
                        index < len(entries)
                        and entries[index][2] == entries[index - 1][2] + 1
                    ):
                        continue

                    run = entries[run_start:index]
                    if self.file_format_versions[file_idx] == 2 and len(run) > 1:
                        batch = self._load_v2_run(file_idx, run, in_file)
                        for (position, _, _), event in zip(run, batch):
                            results[position] = event
                    else:
                        for position, idx, entry_idx in run:
                            results[position] = self._load_entry(
                                idx, file_idx, entry_idx, in_file
                            )
                    run_start = index
            finally:
                if should_close:
                    in_file.close()

        # Reads were grouped by file; return them in the caller's original order
        return cast(list[dict[str, Any]], results)

    def _load_v2_run(
        self,
        file_idx: int,
        entries: list[tuple[int, int, int]],
        in_file: h5py.File,
    ) -> list[dict[str, Any]]:
        """Decode one contiguous run of V2 events from an open file.

        Parameters
        ----------
        file_idx : int
            Index of the physical file containing the run.
        entries : list[tuple[int, int, int]]
            Output position, user-facing index, and file-local entry index for
            each event. File-local indexes must form an increasing sequence.
        in_file : h5py.File
            Open readable handle for ``file_idx``.

        Returns
        -------
        list[dict]
            Decoded events in run order.
        """
        first = entries[0][2]
        last = entries[-1][2] + 1

        # Seed administrative metadata before stored provenance is decoded.
        data = []
        for _, idx, entry_idx in entries:
            event = {"file_index": file_idx, "file_entry_index": entry_idx}
            event.update(self.get_source_provenance(file_idx, entry_idx))
            data.append(event)

        products = require_group(in_file, "products")
        for key in products:
            if key is not None and self.should_load_key(key):
                self.load_product_many(products, first, last, data, key)
        self.reconstruct_products_many(products, first, last, data)

        # Match scalar semantics: the exposed index belongs to this reader,
        # regardless of any index value serialized in the source file.
        for (_, idx, _), event in zip(entries, data):
            event["index"] = idx
        return data

    def _load_entry(
        self,
        idx: int,
        file_idx: int,
        entry_idx: int,
        in_file: h5py.File,
    ) -> dict[str, Any]:
        """Decode one resolved entry from an already-open file handle.

        Parameters
        ----------
        idx : int
            User-facing reader index written into the returned metadata.
        file_idx : int
            Index of the physical file containing the event.
        entry_idx : int
            Event index local to that physical file.
        in_file : h5py.File
            Open readable handle for ``file_idx``.

        Returns
        -------
        dict
            Decoded event products and administrative metadata.

        Notes
        -----
        This method does not own ``in_file`` and therefore never closes it.
        Keeping handle ownership in :meth:`get` and :meth:`get_many` lets both
        scalar and batch paths share the same decoding implementation.
        """

        # Use the event tree to find out what needs to be loaded
        data = {"file_index": file_idx, "file_entry_index": entry_idx}
        data.update(self.get_source_provenance(file_idx, entry_idx))
        events = in_file["events"]
        assert isinstance(
            events, h5py.Dataset
        ), "'events' is not a dataset in the HDF5 file."

        # Dispatch on the physical layout of the file containing this entry.
        if self.file_format_versions[file_idx] == 1:
            event = events[entry_idx]
            names = getattr(getattr(event, "dtype", None), "names", None)
            if names is not None:
                for key in names:
                    if self.should_load_key(key):
                        self.load_region_product(in_file, event, data, key)
            else:
                raise ValueError("Event entry does not have named fields.")

        else:
            # V2 products own their implementation datasets below `/products`
            products = require_group(in_file, "products")
            for key in products:
                if key is not None and self.should_load_key(key):
                    self.load_product(products, entry_idx, data, key)
            self.reconstruct_products(products, entry_idx, data)

        # Use the global index, not the one read from file
        data["index"] = idx

        return data

    def should_load_key(self, key: str) -> bool:
        """Return whether a product belongs to the reader projection.

        Source-provenance keys are always admitted when present. They are
        administrative inputs used by :meth:`get_source_provenance`, rather
        than ordinary user-requested products, and are required to preserve
        source entry identity across an HDF5 round trip.

        Parameters
        ----------
        key : str
            Stored data-product name.

        Returns
        -------
        bool
            ``True`` when the product should be read from disk.
        """
        return (
            self.requested_keys is None
            or key in self.requested_keys
            or key
            in {
                "source_file_index",
                "source_file_entry_index",
            }
        )


def _get_reader_pid() -> int:
    """Return the current process ID for HDF5 handle ownership checks."""
    return os.getpid()
