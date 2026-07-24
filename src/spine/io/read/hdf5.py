"""Contains a reader class dedicated to loading data from HDF5 files."""

import os
from dataclasses import dataclass, fields
from typing import Any
from warnings import warn

import h5py
import numpy as np
import yaml
from yaml.parser import ParserError

import spine.data
from spine.utils.logger import logger

from .base import ReaderBase

__all__ = ["HDF5Reader"]


class HDF5Reader(ReaderBase):
    """Read event data from versioned SPINE HDF5 files.

    This class inherits from the :class:`ReaderBase` class. It provides
    methods to load HDF5 files and extract their data products. Two physical
    layouts are supported:

    - Version 1 is the legacy layout. Each row of ``events`` is a compound
      record of HDF5 region references into top-level product datasets.
      Variable object attributes use HDF5 VLEN fields.
    - Version 2 stores each product in a top-level group and uses monotonic
      integer offset arrays to delimit events, objects, and variable fields.
      Its ``events`` dataset is deliberately retained as the authoritative
      event axis, but contains no product references.

    The reader detects the layout independently for every input file. This
    allows one logical dataset to span legacy and V2 files without exposing
    layout differences to callers. Files which predate explicit
    ``info.attrs["format_version"]`` metadata are interpreted as V1.

    Product projection is performed before any product dataset is accessed.
    This is particularly useful for V2 because product names live at the file
    root rather than in the ``events`` compound dtype.
    """

    name: str = "hdf5"

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
        skip_unknown_attrs: bool = False,
        run_info_key: str = "run_info",
        allow_missing: bool = False,
        keep_open: bool = True,
        swmr: bool = False,
        ignore_incomplete: bool = False,
        keys: list[str] | tuple[str, ...] | None = None,
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
        """
        # Process the list of files
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)
        self.keep_open = keep_open
        self.swmr = swmr
        self.ignore_incomplete = ignore_incomplete
        self._handle_pid: int | None = None
        self._file_handles: dict[int, h5py.File] = {}

        # V2 object schemas are immutable for the lifetime of a file. Cache
        # their decoded attribute metadata so event reads do not repeatedly
        # invoke YAML or rediscover logical fields from compound dtypes.
        # Only plain Python values are cached; h5py objects remain tied to the
        # process-local file-handle lifecycle above.
        self._v2_object_schemas: dict[
            tuple[str, str],
            tuple[
                str,
                bool,
                tuple[str, ...],
                tuple[tuple[str, int, bool, tuple[str, ...]], ...],
            ],
        ] = {}
        self._v2_object_handles: dict[
            tuple[str, str],
            tuple[h5py.Dataset, h5py.Dataset, tuple[h5py.Dataset, ...]],
        ] = {}
        self._v2_product_handles: dict[tuple[str, str], _V2ProductHandles] = {}
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
                    assert (
                        run_info_key in in_file
                    ), f"Must provide {run_info_key} to create run map"

                    info = in_file[run_info_key]
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
        )

        # Store other attributes
        self.build_classes = build_classes
        self.skip_unknown_attrs = skip_unknown_attrs

        # Process the configuration used to produce the HDF5 file
        self.cfg = self.process_cfg()

        # Process the SPINE version used to produced the HDF5 file
        self.version = self.process_version()

    def close(self) -> None:
        """Close any persistent HDF5 handles owned by this reader.

        This only affects handles cached in the current process. It is safe to
        call repeatedly.
        """
        for handle in getattr(self, "_file_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

        self._file_handles = {}
        self._v2_object_handles = {}
        self._v2_product_handles = {}
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
        """Returns a specific entry in the file.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        data : dict
            Ditionary of data products corresponding to one event
        """
        # Get the appropriate entry index
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}."
            )
        file_idx = self.get_file_index(idx)
        entry_idx = self.get_file_entry_index(idx)

        # Use the event tree to find out what needs to be loaded
        data = {"file_index": file_idx, "file_entry_index": entry_idx}
        data.update(self.get_source_provenance(file_idx, entry_idx))
        in_file, should_close = self._open_file(file_idx)
        try:
            events = in_file["events"]
            assert isinstance(
                events, h5py.Dataset
            ), "'events' is not a dataset in the HDF5 file."

            # Dispatch on the physical layout of the file containing this
            # entry. `file_format_versions` is parallel to `file_paths`, so a
            # single reader can transparently span V1 and V2 files.
            if self.file_format_versions[file_idx] == 1:
                event = events[entry_idx]
                names = getattr(getattr(event, "dtype", None), "names", None)
                if names is not None:
                    for key in names:
                        if self.should_load_key(key):
                            self.load_key(in_file, event, data, key)
                else:
                    raise ValueError("Event entry does not have named fields.")
            else:
                # V2 product membership is represented by top-level groups,
                # not fields in the events dtype.
                for key in in_file.keys():
                    if key not in {"events", "info"} and self.should_load_key(key):
                        self.load_key_v2(in_file, entry_idx, data, key)
        finally:
            if should_close:
                in_file.close()

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

    def load_key_v2(
        self,
        in_file: h5py.File,
        entry_idx: int,
        data: dict[str, Any],
        key: str,
    ) -> None:
        """Load one event product from an offset-based V2 group.

        Every V2 product advertises a ``kind`` attribute which determines its
        physical schema:

        - ``array`` and ``string`` use ``values`` plus ``event_offsets``.
        - ``objects`` use compound ``fixed`` rows, per-event object offsets,
          and flat variable-field pools.
        - ``list`` represents a variable number of same-width arrays. Its
          ``event_offsets`` delimit elements and ``element_offsets`` delimit
          values.
        - ``multi_list`` represents a fixed number of differently shaped
          arrays using one child group per list position.

        Offset arrays follow the boundary convention that item ``i`` occupies
        ``offsets[i]:offsets[i + 1]``. An offset dataset for ``N`` items
        therefore contains ``N + 1`` entries. Empty items are represented by
        equal adjacent offsets and require no special sentinel.

        Parameters
        ----------
        in_file : h5py.File
            Open file containing the product.
        entry_idx : int
            File-local event index.
        data : dict
            Event dictionary to update with the decoded value.
        key : str
            Top-level V2 product-group name.
        """
        cache_key = (os.fspath(in_file.filename), key)
        product = self._v2_product_handles.get(cache_key) if self.keep_open else None
        if product is None:
            group = in_file[key]
            if not isinstance(group, h5py.Group) or "kind" not in group.attrs:
                raise ValueError(
                    f"V2 product '{key}' is not a recognized product group."
                )
            kind = _decode_string_attribute(group.attrs["kind"], "kind")

            # Resolve the physical datasets once for persistent readers. Name
            # lookup and attribute decoding are surprisingly visible when
            # repeated for every product of every event.
            if kind in {"array", "string"}:
                product = _V2ProductHandles(
                    kind=kind,
                    values=_require_dataset(group, "values"),
                    event_offsets=_require_dataset(group, "event_offsets"),
                    scalar=bool(group.attrs["scalar"]),
                )
            elif kind == "objects":
                product = _V2ProductHandles(kind=kind, object_group=group)
            elif kind == "list":
                product = _V2ProductHandles(
                    kind=kind,
                    values=_require_dataset(group, "values"),
                    element_offsets=_require_dataset(group, "element_offsets"),
                    event_offsets=_require_dataset(group, "event_offsets"),
                )
            elif kind == "multi_list":
                elements = []
                for name in sorted(
                    group.keys(), key=lambda item: int(item.split("_")[-1])
                ):
                    element = _require_group(group, name)
                    elements.append(
                        (
                            _require_dataset(element, "values"),
                            _require_dataset(element, "event_offsets"),
                        )
                    )
                product = _V2ProductHandles(kind=kind, elements=tuple(elements))
            else:
                raise ValueError(
                    f"Unrecognized V2 product kind '{kind}' for key '{key}'."
                )

            if self.keep_open:
                self._v2_product_handles[cache_key] = product

        kind = product.kind

        if kind == "array":
            values = product.values
            offsets = product.event_offsets
            assert values is not None and offsets is not None
            start, stop = (int(v) for v in offsets[entry_idx : entry_idx + 2])
            result = values[start:stop]
            if product.scalar:
                result = result[0]
            data[key] = result
            return

        if kind == "string":
            # Strings are explicit UTF-8 byte spans rather than HDF5 VLEN
            # strings, so their physical representation is predictable.
            values = product.values
            offsets = product.event_offsets
            assert values is not None and offsets is not None
            start, stop = (int(v) for v in offsets[entry_idx : entry_idx + 2])
            data[key] = values[start:stop].tobytes().decode("utf-8")
            return

        if kind == "objects":
            assert product.object_group is not None
            self.load_objects_v2(product.object_group, entry_idx, data, key)
            return

        if kind == "list":
            # First map the event to a range of logical elements, then map
            # every element to its slice in the shared values dataset.
            values = product.values
            element_offsets = product.element_offsets
            event_offsets = product.event_offsets
            assert (
                values is not None
                and element_offsets is not None
                and event_offsets is not None
            )
            first, last = (int(v) for v in event_offsets[entry_idx : entry_idx + 2])
            bounds = element_offsets[first : last + 1]
            result = np.empty(last - first, dtype=object)
            # Read the event's complete span once. The individual arrays below
            # are inexpensive slices of this in-memory block.
            base = int(bounds[0]) if len(bounds) else 0
            terminal = int(bounds[-1]) if len(bounds) else base
            event_values = values[base:terminal]
            for i in range(last - first):
                start = int(bounds[i]) - base
                stop = int(bounds[i + 1]) - base
                result[i] = event_values[start:stop]
            data[key] = result
            return

        if kind == "multi_list":
            result = []
            for values, offsets in product.elements:
                start, stop = (int(v) for v in offsets[entry_idx : entry_idx + 2])
                result.append(values[start:stop])
            data[key] = result
            return

    def load_objects_v2(
        self,
        group: h5py.Group,
        entry_idx: int,
        data: dict[str, Any],
        key: str,
    ) -> None:
        """Load one V2 object collection and optionally rebuild its classes.

        A V2 object product separates attributes by storage behavior:

        - ``fixed`` contains one compound row per object. Scalar, enum,
          fixed-width, and derived properties live here and can be consumed by
          external HDF5 tools without importing SPINE.
        - ``variables/pool_N/values`` concatenates variable-length fields which
          share a physical dtype. Strings use UTF-8 byte pools.
        - ``_var_offsets_N`` in each fixed row contains ``F + 1`` absolute
          boundaries for the ``F`` fields listed in the pool's ``fields``
          attribute.

        Only the object and variable-value spans touched by this event are
        read. Logical dictionaries are then passed to ``DataBase.from_dict``;
        when ``build_classes=False`` those dictionaries are returned directly.

        Parameters
        ----------
        group : h5py.Group
            V2 object product group.
        entry_idx : int
            File-local event index.
        data : dict
            Event dictionary to update.
        key : str
            Name under which to store the reconstructed collection.
        """
        # Dataset dtypes and group attributes do not vary by event. In
        # particular, parsing each pool's YAML-encoded field list here used to
        # be a significant fraction of V2 read time. Cache only decoded Python
        # metadata, rather than h5py handles, so keep_open=False and fork-safe
        # handle reopening continue to work normally.
        file_name = os.fspath(group.file.filename)
        group_name = group.name
        if group_name is None:
            raise ValueError("V2 object group must have a file path.")
        schema_key = (file_name, group_name)
        schema = self._v2_object_schemas.get(schema_key)
        if schema is None:
            fixed = _require_dataset(group, "fixed")
            variables = _require_group(group, "variables")
            class_name = _decode_string_attribute(
                group.attrs["class_name"], "class_name"
            )
            scalar = bool(group.attrs["scalar"])
            fixed_names = tuple(
                name
                for name in fixed.dtype.names or ()
                if not name.startswith("_var_offsets_")
            )
            decoded_pool_specs: list[tuple[str, int, bool, tuple[str, ...]]] = []
            for pool_name, pool in sorted(
                variables.items(),
                key=lambda item: int(item[0].split("_")[-1]),
            ):
                if not isinstance(pool, h5py.Group):
                    raise TypeError(f"V2 variable pool '{pool_name}' must be a group.")
                pool_index = int(pool_name.split("_")[-1])
                kind = _decode_string_attribute(pool.attrs["kind"], "kind")
                fields_value = yaml.safe_load(
                    _decode_string_attribute(pool.attrs["fields"], "fields")
                )
                if not isinstance(fields_value, list) or not all(
                    isinstance(name, str) for name in fields_value
                ):
                    raise TypeError(
                        f"V2 variable pool '{pool_name}' fields must be "
                        "a list of strings."
                    )
                fields = tuple(fields_value)
                decoded_pool_specs.append(
                    (pool_name, pool_index, kind == "string", fields)
                )
            schema = (
                class_name,
                scalar,
                fixed_names,
                tuple(decoded_pool_specs),
            )
            self._v2_object_schemas[schema_key] = schema

        class_name, scalar, fixed_names, pool_specs = schema

        # Persistent readers can also retain direct dataset handles. This
        # removes several group-name lookups per variable pool and event. The
        # cache is cleared whenever the owning file handles are closed or a
        # reader crosses a process boundary.
        handles = self._v2_object_handles.get(schema_key) if self.keep_open else None
        if handles is None:
            fixed = _require_dataset(group, "fixed")
            event_offsets = _require_dataset(group, "event_offsets")
            pool_values = []
            for pool_name, _, _, _ in pool_specs:
                values = _require_dataset(group, f"variables/{pool_name}/values")
                pool_values.append(values)
            handles = (fixed, event_offsets, tuple(pool_values))
            if self.keep_open:
                self._v2_object_handles[schema_key] = handles

        fixed, event_offsets, pool_values = handles

        # Select only the compound object rows owned by this event.
        first, last = (int(v) for v in event_offsets[entry_idx : entry_idx + 2])
        rows = fixed[first:last]
        obj_class = self.resolve_object_class(class_name, rows)

        variable_values: dict[str, list[Any]] = {}
        for values, (_, pool_index, is_string, fields) in zip(pool_values, pool_specs):
            bounds = rows[f"_var_offsets_{pool_index}"]
            # Pool offsets are absolute across the file. Read the enclosing
            # event span once and subtract `base` for local NumPy slicing.
            base = int(bounds[0, 0]) if len(bounds) else 0
            terminal = int(bounds[-1, -1]) if len(bounds) else base
            event_values = values[base:terminal]
            for j, name in enumerate(fields):
                loaded = []
                for i in range(last - first):
                    start = int(bounds[i, j]) - base
                    stop = int(bounds[i, j + 1]) - base
                    value = event_values[start:stop]
                    if is_string:
                        value = value.tobytes().decode("utf-8")
                    loaded.append(value)
                variable_values[name] = loaded

        # Offset helper columns are physical metadata and are intentionally
        # excluded from the logical object dictionaries.
        result = []
        for i, row in enumerate(rows):
            obj_dict = {name: row[name] for name in fixed_names}
            obj_dict.update(
                {name: values[i] for name, values in variable_values.items()}
            )
            if self.build_classes:
                result.append(obj_class.from_dict(obj_dict))
            else:
                result.append(obj_dict)
        if scalar:
            result = result[0]
        data[key] = result

    @staticmethod
    def resolve_object_class(class_name: str, array: np.ndarray) -> type:
        """Resolve an HDF5 object class name to the concrete SPINE class.

        This keeps backward-compatibility quirks localized in the reader.
        In particular, older HDF5 files stored image metadata with
        ``class_name="Meta"``. Newer files store the explicit
        ``ImageMeta2D`` / ``ImageMeta3D`` class names instead.

        Parameters
        ----------
        class_name : str
            Class name stored in the HDF5 dataset metadata
        array : np.ndarray
            Structured array slice containing the serialized objects

        Returns
        -------
        type
            Concrete SPINE data class to reconstruct
        """
        if class_name != "Meta":
            return getattr(spine.data, class_name)

        from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D

        names = getattr(array.dtype, "names", None)
        assert names is not None and "count" in names, (
            "Legacy HDF5 class_name='Meta' requires a structured dtype "
            "with a 'count' field."
        )
        sample = array[0] if len(array) else None
        if sample is None:
            return ImageMeta3D

        dim = len(sample["count"])
        if dim == 2:
            return ImageMeta2D
        if dim == 3:
            return ImageMeta3D

        raise ValueError(
            f"Unsupported legacy Meta dimensionality: {dim}. Expected 2 or 3."
        )

    def load_key(
        self,
        in_file: h5py.File,
        event: dict[str, Any],
        data: dict[str, Any],
        key: str,
    ) -> None:
        """Fetch a specific key for a specific event.

        Parameters
        ----------
        in_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        data : dict
            Dictionary of data products corresponding to one event
        key: str
            Name of the dataset in the entry
        """
        # The event-level information is a region reference: fetch it
        region_ref = event[key]
        dataset = in_file[key]
        if isinstance(dataset, h5py.Dataset):
            names = getattr(getattr(dataset, "dtype", None), "names", None)
            if not names:
                # If the reference points at a simple dataset, return
                data[key] = dataset[region_ref]
                if dataset.attrs["scalar"]:
                    data[key] = data[key][0]
                if len(dataset.shape) > 1:
                    data[key] = data[key].reshape(-1, dataset.shape[1])

            else:
                # If the dataset has multiple attributes, it contains an object.
                # Start by fetching the appropriate class to rebuild
                array = dataset[region_ref]
                class_name = dataset.attrs["class_name"]
                assert isinstance(
                    class_name, str
                ), "Dataset missing 'class_name' attribute."
                obj_class = self.resolve_object_class(class_name, array)

                # If needed, get the list of recognized attributes
                known_attrs = []
                if self.skip_unknown_attrs:
                    known_attrs = [f.name for f in fields(obj_class)]

                # Load the object
                names = array.dtype.names
                data[key] = []
                for i, el in enumerate(array):
                    # Fetch the list of key/value pairs, filter if requested
                    if self.skip_unknown_attrs:
                        obj_dict = {}
                        for i, k in enumerate(names):
                            if k in known_attrs:
                                obj_dict[k] = el[i]
                    else:
                        obj_dict = dict(zip(names, el))

                    # Rebuild an instance of the object class, if requested
                    if self.build_classes:
                        data[key].append(obj_class.from_dict(obj_dict))
                    else:
                        data[key].append(obj_dict)

                if in_file[key].attrs["scalar"]:
                    data[key] = data[key][0]

        elif isinstance(dataset, h5py.Group):
            # If the reference points at a group, unpack
            index = dataset["index"]
            assert isinstance(index, h5py.Dataset), "Dataset 'index' is missing."
            el_refs = index[region_ref].flatten()
            if len(index.shape) == 1:
                elements = dataset["elements"]
                assert isinstance(
                    elements, h5py.Dataset
                ), "Dataset 'elements' is missing."
                ret = np.empty(len(el_refs), dtype=object)
                ret[:] = [elements[r] for r in el_refs]
                if len(elements.shape) > 1:
                    for i in range(len(el_refs)):
                        ret[i] = ret[i].reshape(-1, elements.shape[1])

            else:
                elements = [dataset[f"element_{i}"] for i in range(len(el_refs))]
                ret = []
                for i, el in enumerate(elements):
                    assert isinstance(
                        el, h5py.Dataset
                    ), f"Dataset 'element_{i}' is missing."
                    ret.append(el[el_refs[i]])
                    if len(el.shape) > 1:
                        ret[i] = ret[i].reshape(-1, el.shape[1])

            data[key] = ret

        else:
            raise ValueError(f"Dataset for key '{key}' is neither a group nor dataset.")


@dataclass(frozen=True)
class _V2ProductHandles:
    """Resolved HDF5 objects needed to load one V2 product."""

    kind: str
    values: h5py.Dataset | None = None
    event_offsets: h5py.Dataset | None = None
    scalar: bool = False
    object_group: h5py.Group | None = None
    element_offsets: h5py.Dataset | None = None
    elements: tuple[tuple[h5py.Dataset, h5py.Dataset], ...] = ()


def _require_dataset(parent: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """Return a named child dataset or fail with a schema error."""
    child = parent[name]
    if not isinstance(child, h5py.Dataset):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 dataset.")
    return child


def _require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """Return a named child group or fail with a schema error."""
    child = parent[name]
    if not isinstance(child, h5py.Group):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 group.")
    return child


def _decode_string_attribute(value: Any, name: str) -> str:
    """Normalize one required byte/string-valued HDF5 attribute."""
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise TypeError(f"HDF5 attribute '{name}' must be a string.")
    return value


def _get_reader_pid() -> int:
    """Return the current process ID for HDF5 handle ownership checks."""
    return os.getpid()
